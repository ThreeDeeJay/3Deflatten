// SPDX-License-Identifier: GPL-3.0-or-later
// 3Deflatten – main DirectShow CTransformFilter
#pragma once
#include <streams.h>
#include <memory>
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

    CCritSec         m_csConfig;
    DeflattenConfig  m_cfg{};
    std::wstring     m_modelPath;
    std::wstring     m_gpuInfo;

    std::unique_ptr<DepthEstimator> m_depth;
    std::unique_ptr<StereoRenderer> m_stereo;

    int  m_inW = 0, m_inH = 0, m_inStride = 0;
    bool m_isBGR = true;

    std::vector<BYTE> m_outBuf;
};
