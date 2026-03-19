// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten – main DirectShow CTransformFilter
#pragma once
#include <streams.h>
#include <memory>
#include <thread>
#include <mutex>
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

    std::vector<BYTE> m_outBuf;
    int               m_frameCount = 0;

    // ── Async depth worker (single-slot latest-wins) ──────────────────────────
    // Transform() converts the frame to BGRA and posts it here, then returns
    // immediately using the last completed depth map.  The worker thread runs
    // ORT inference in the background so the media-player graph never stalls.
    void StartDepthThread();
    void StopDepthThread();
    void DepthWorkerThread();

    std::thread             m_depthThread;

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
};
