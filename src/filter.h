// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten – main DirectShow CTransformFilter
#pragma once
#include <streams.h>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "ideflatten.h"
#include "depth_estimator.h"
#include "stereo_renderer.h"
#include "logger.h"

// Forward declaration of the setup filter structure defined in filter.cpp
extern const AMOVIESETUP_FILTER sudFilter;

class C3DeflattenFilter
    : public CTransformFilter
    , public I3Deflatten
    , public ISpecifyPropertyPages
{
public:
    static CUnknown* WINAPI CreateInstance(LPUNKNOWN pUnk, HRESULT* phr);

    DECLARE_IUNKNOWN
    STDMETHODIMP NonDelegatingQueryInterface(REFIID riid, void** ppv) override;

    // ── CTransformFilter ──────────────────────────────────────────────────────
    HRESULT CheckInputType(const CMediaType* pmt) override;
    HRESULT GetMediaType(int iPos, CMediaType* pmt) override;
    HRESULT CheckTransform(const CMediaType* pmtIn,
                           const CMediaType* pmtOut) override;
    HRESULT DecideBufferSize(IMemAllocator* pAlloc,
                             ALLOCATOR_PROPERTIES* pProps) override;
    HRESULT Transform(IMediaSample* pIn, IMediaSample* pOut) override;
    HRESULT StartStreaming() override;
    HRESULT StopStreaming()  override;

    // ── I3Deflatten ───────────────────────────────────────────────────────────
    STDMETHODIMP GetConfig(DeflattenConfig* pCfg) override;
    STDMETHODIMP SetConfig(const DeflattenConfig* pCfg) override;
    STDMETHODIMP GetModelPath(LPWSTR buf, UINT cch) override;
    STDMETHODIMP SetModelPath(LPCWSTR path) override;
    STDMETHODIMP GetGPUInfo(LPWSTR buf, UINT cch) override;
    STDMETHODIMP ReloadModel() override;

    // ── ISpecifyPropertyPages ─────────────────────────────────────────────────
    STDMETHODIMP GetPages(CAUUID* pPages) override;

private:
    explicit C3DeflattenFilter(LPUNKNOWN pUnk, HRESULT* phr);
    ~C3DeflattenFilter() override;

    void    OutputDimensions(int inW, int inH, int& outW, int& outH) const;
    HRESULT BuildOutputMediaType(const CMediaType* pmtIn, CMediaType* pmtOut);

    // ── INI persistence ───────────────────────────────────────────────────────
    // Reads/writes 3Deflatten.ini next to the .ax file.
    // LoadIni() is called from the constructor; SaveIni() from SetConfig().
    static std::wstring GetIniPath();
    void LoadIni();
    void SaveIni() const;

    std::wstring     m_iniPath;

    CCritSec         m_csConfig;
    DeflattenConfig  m_cfg{};
    std::wstring     m_modelPath;
    std::wstring     m_gpuInfo;

    std::unique_ptr<DepthEstimator> m_depth;
    std::unique_ptr<StereoRenderer> m_stereo;

    int  m_inW = 0, m_inH = 0, m_inStride = 0;
    bool m_isBGR  = true;
    bool m_isNV12 = false;
    bool m_isYUY2 = false;

    int                m_frameCount = 0;

    // Pre-allocated per-frame buffers — never freed during streaming, eliminating
    // per-frame heap allocs. Render writes directly into the DirectShow output
    // sample (pDst) so there is no intermediate output buffer copy.
    std::vector<BYTE>  m_convBuf;       // NV12/YUY2 → BGRA (srcW*srcH*4 bytes)
    std::vector<float> m_depthRender;   // depth for current render (zero-copy swap from cache)
    bool               m_hadRealDepth = false; // true once we've received a real depth map

    // ── Depth motion compensation ─────────────────────────────────────────────
    // Between inference frames the depth map is stale.  We estimate the global
    // 2D translation between consecutive decoded frames (Lucas-Kanade on a
    // downsampled luma image) and accumulate it as a UV offset that is passed to
    // the stereo shader to warp depth sampling back in sync with the current frame.
    // When new depth arrives (haveDepth), the accumulation resets to (0, 0).
    std::vector<uint8_t> m_prevLumaSmall; // previous frame luma, downsampled
    int                  m_lumaSmW = 0, m_lumaSmH = 0;
    float                m_accumDx = 0.f, m_accumDy = 0.f;

    // Estimate global 2D translation between luma images using Lucas-Kanade.
    // Returns (dx, dy) in SOURCE-RESOLUTION pixels. smallW/H: downsampled dims.
    // scale: srcW / smallW (to convert small-image offsets back to source pixels).
    static void EstimateMotionLK(const uint8_t* prev, const uint8_t* curr,
                                  int w, int h, float scale,
                                  float& outDx, float& outDy);

    // Adaptive frame skipping: when inference is slower than the video frame rate,
    // skip posting frames to the worker so we always infer the LATEST frame rather
    // than a frame that will never be displayed.
    // m_skipEvery = 1 → infer every frame; 2 → every other; 3 → every third, etc.
    int                m_skipEvery    = 1;      // updated each time worker returns
    int                m_skipCounter  = 0;      // counts Transform() calls for skip logic
    double             m_avgInferMs   = 0.0;    // exponential moving average of inference ms

    // ── Async depth worker (single-slot latest-wins) ──────────────────────────
    // Transform() converts the frame to BGRA and posts it here, then returns
    // immediately using the last completed depth map.  The worker thread runs
    // ORT inference in the background so the media-player graph never stalls.
    void StartDepthThread();
    void StopDepthThread();
    void DepthWorkerThread();

    std::thread             m_depthThread;

    // ── Depth-view hotkey thread ──────────────────────────────────────────────
    // Polls GetAsyncKeyState at ~30 ms intervals and toggles m_showDepth when
    // cfg.depthViewKey is pressed.  Runs independently of the graph thread so
    // the toggle is responsive during pause as well as during playback.
    void StartHotkeyThread();
    void StopHotkeyThread();
    void HotkeyThread();

    std::thread             m_hotkeyThread;
    std::atomic<bool>       m_hotkeyStop{false};

    // Pending work slot (written by Transform, read by worker)
    std::mutex              m_pendMtx;
    std::condition_variable m_pendCV;
    bool                    m_pendStop  = false;
    bool                    m_pendReady = false;
    std::vector<BYTE>       m_pendBGRA;
    int                     m_pendW     = 0;
    int                     m_pendH     = 0;

    // Depth result cache (written by worker, read by Transform)
    // We always render the CURRENT input frame with the LATEST cached depth.
    // The old "matched-source" approach (m_cachedBGRA) caused PotPlayer to
    // receive 5-6 identical output frames per inference cycle at slow model
    // speeds (e.g. 250ms/frame) → looked completely frozen. The tiny temporal
    // desync (depth is ~1 frame stale during fast motion) is imperceptible.
    std::mutex              m_cacheMtx;
    std::vector<float>      m_cachedDepth;
    int                     m_cachedW    = 0;
    int                     m_cachedH    = 0;
    bool                    m_cacheReady = false;
    double                  m_lastInferMs = 0.0;  // inference time of most recent result
};
