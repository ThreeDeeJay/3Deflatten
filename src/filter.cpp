// SPDX-License-Identifier: GPL-3.0-or-later
#include "filter.h"
#include "prop_page.h"
#include "guids.h"
#include <dvdmedia.h>
#include <uuids.h>
#include <algorithm>

// ── Filter registration ───────────────────────────────────────────────────────
static const AMOVIESETUP_MEDIATYPE sudPinTypes[] = {
    { &MEDIATYPE_Video, &MEDIASUBTYPE_RGB32  },
    { &MEDIATYPE_Video, &MEDIASUBTYPE_ARGB32 },
    { &MEDIATYPE_Video, &MEDIASUBTYPE_RGB24  },
    { &MEDIATYPE_Video, &MEDIASUBTYPE_YUY2   },
    { &MEDIATYPE_Video, &MEDIASUBTYPE_NV12   },
};

static const AMOVIESETUP_PIN sudPins[] = {
    { const_cast<LPWSTR>(L"Input"),  FALSE, FALSE, FALSE, FALSE,
      &CLSID_NULL, nullptr, ARRAYSIZE(sudPinTypes), sudPinTypes },
    { const_cast<LPWSTR>(L"Output"), FALSE, TRUE,  FALSE, FALSE,
      &CLSID_NULL, nullptr, ARRAYSIZE(sudPinTypes), sudPinTypes },
};

const AMOVIESETUP_FILTER sudFilter = {
    &CLSID_3Deflatten,
    L"3Deflatten (2D to 3D AI Depth)",
    MERIT_DO_NOT_USE,
    ARRAYSIZE(sudPins),
    sudPins
};

// ── CreateInstance ────────────────────────────────────────────────────────────
CUnknown* WINAPI C3DeflattenFilter::CreateInstance(LPUNKNOWN pUnk,
                                                    HRESULT* phr) {
    return new C3DeflattenFilter(pUnk, phr);
}

C3DeflattenFilter::C3DeflattenFilter(LPUNKNOWN pUnk, HRESULT* phr)
    : CTransformFilter(L"3Deflatten", pUnk, CLSID_3Deflatten)
{
    m_cfg.convergence = 0.5f;
    m_cfg.separation  = 0.03f;
    m_cfg.outputMode  = OutputMode::SideBySide;
    m_cfg.gpuProvider = GPUProvider::Auto;
    m_cfg.depthSmooth = 0.4f;
    m_cfg.flipDepth   = FALSE;

    wchar_t envModel[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"DEFLATTEN_MODEL_PATH", envModel, MAX_PATH))
        m_modelPath = envModel;

    m_depth  = std::make_unique<DepthEstimator>();
    m_stereo = std::make_unique<StereoRenderer>();

    LOG_INFO("C3DeflattenFilter constructed");
    if (phr) *phr = S_OK;
}

C3DeflattenFilter::~C3DeflattenFilter() {
    LOG_INFO("C3DeflattenFilter destroyed");
}

STDMETHODIMP C3DeflattenFilter::NonDelegatingQueryInterface(REFIID riid,
                                                             void** ppv) {
    if (riid == IID_I3Deflatten)
        return GetInterface(static_cast<I3Deflatten*>(this), ppv);
    if (riid == IID_ISpecifyPropertyPages)
        return GetInterface(static_cast<ISpecifyPropertyPages*>(this), ppv);
    return CTransformFilter::NonDelegatingQueryInterface(riid, ppv);
}

// ── CheckInputType ────────────────────────────────────────────────────────────
HRESULT C3DeflattenFilter::CheckInputType(const CMediaType* pmt) {
    if (pmt->majortype != MEDIATYPE_Video) return VFW_E_TYPE_NOT_ACCEPTED;
    static const GUID* allowed[] = {
        &MEDIASUBTYPE_RGB32,  &MEDIASUBTYPE_ARGB32,
        &MEDIASUBTYPE_RGB24,  &MEDIASUBTYPE_YUY2,
        &MEDIASUBTYPE_NV12,
    };
    for (auto* g : allowed)
        if (pmt->subtype == *g) return S_OK;
    return VFW_E_TYPE_NOT_ACCEPTED;
}

// ── GetMediaType ──────────────────────────────────────────────────────────────
HRESULT C3DeflattenFilter::GetMediaType(int iPos, CMediaType* pmt) {
    if (iPos < 0) return E_INVALIDARG;
    if (iPos > 0) return VFW_S_NO_MORE_ITEMS;
    if (!m_pInput || !m_pInput->IsConnected()) return E_UNEXPECTED;

    CMediaType mtIn;
    m_pInput->ConnectionMediaType(&mtIn);
    return BuildOutputMediaType(&mtIn, pmt);
}

HRESULT C3DeflattenFilter::BuildOutputMediaType(const CMediaType* pmtIn,
                                                  CMediaType* pmtOut) {
    *pmtOut = *pmtIn;
    auto* bmiIn = reinterpret_cast<BITMAPINFOHEADER*>(pmtIn->Format());
    if (!bmiIn) return E_FAIL;

    int inW = bmiIn->biWidth, inH = abs(bmiIn->biHeight);
    int outW, outH;
    OutputDimensions(inW, inH, outW, outH);

    auto* bmiOut = reinterpret_cast<BITMAPINFOHEADER*>(pmtOut->Format());
    bmiOut->biWidth     = outW;
    bmiOut->biHeight    = (bmiIn->biHeight < 0) ? -outH : outH;
    pmtOut->subtype     = MEDIASUBTYPE_RGB32;
    bmiOut->biBitCount  = 32;
    bmiOut->biCompression = BI_RGB;
    bmiOut->biSizeImage   = outW * outH * 4;
    pmtOut->SetSampleSize(bmiOut->biSizeImage);

    if (pmtOut->formattype == FORMAT_VideoInfo) {
        auto* vih = reinterpret_cast<VIDEOINFOHEADER*>(pmtOut->Format());
        vih->rcSource = {0, 0, outW, outH};
        vih->rcTarget = vih->rcSource;
        vih->dwBitRate = (DWORD)((double)vih->dwBitRate * outW * outH
                                  / (inW * inH));
    }
    return S_OK;
}

void C3DeflattenFilter::OutputDimensions(int inW, int inH,
                                          int& outW, int& outH) const {
    CAutoLock lk(const_cast<CCritSec*>(&m_csConfig));
    if (m_cfg.outputMode == OutputMode::SideBySide) {
        outW = inW*2; outH = inH;
    } else {
        outW = inW; outH = inH*2;
    }
}

// ── CheckTransform ────────────────────────────────────────────────────────────
HRESULT C3DeflattenFilter::CheckTransform(const CMediaType* pmtIn,
                                           const CMediaType* pmtOut) {
    CMediaType proposed;
    if (FAILED(BuildOutputMediaType(pmtIn, &proposed))) return VFW_E_TYPE_NOT_ACCEPTED;

    auto* b1 = reinterpret_cast<BITMAPINFOHEADER*>(proposed.Format());
    auto* b2 = reinterpret_cast<BITMAPINFOHEADER*>(pmtOut->Format());
    if (!b1 || !b2) return VFW_E_TYPE_NOT_ACCEPTED;
    if (b1->biWidth != b2->biWidth) return VFW_E_TYPE_NOT_ACCEPTED;
    if (abs(b1->biHeight) != abs(b2->biHeight)) return VFW_E_TYPE_NOT_ACCEPTED;
    return S_OK;
}

// ── DecideBufferSize ──────────────────────────────────────────────────────────
HRESULT C3DeflattenFilter::DecideBufferSize(IMemAllocator* pAlloc,
                                             ALLOCATOR_PROPERTIES* pProps) {
    ASSERT(m_pInput->IsConnected());

    CMediaType mtIn;
    m_pInput->ConnectionMediaType(&mtIn);
    auto* bmi = reinterpret_cast<BITMAPINFOHEADER*>(mtIn.Format());
    if (!bmi) return E_FAIL;

    int outW, outH;
    OutputDimensions(bmi->biWidth, abs(bmi->biHeight), outW, outH);
    pProps->cBuffers  = 1;
    pProps->cbBuffer  = outW * outH * 4;
    pProps->cbAlign   = 1;
    pProps->cbPrefix  = 0;

    ALLOCATOR_PROPERTIES actual;
    return pAlloc->SetProperties(pProps, &actual);
}

// ── StartStreaming ────────────────────────────────────────────────────────────
HRESULT C3DeflattenFilter::StartStreaming() {
    LOG_INFO("StartStreaming");

    HRESULT hr = m_stereo->Init();
    if (FAILED(hr)) { LOG_ERR("StereoRenderer::Init failed"); return hr; }

    if (!m_depth->IsLoaded()) {
        hr = m_depth->Load(m_modelPath, m_cfg.gpuProvider, m_gpuInfo);
        if (FAILED(hr))
            LOG_WARN("DepthEstimator::Load failed – will use flat depth map");
    }

    if (m_pInput && m_pInput->IsConnected()) {
        CMediaType mtIn;
        m_pInput->ConnectionMediaType(&mtIn);
        auto* bmi = reinterpret_cast<BITMAPINFOHEADER*>(mtIn.Format());
        if (bmi) {
            m_inW      = bmi->biWidth;
            m_inH      = abs(bmi->biHeight);
            m_inStride = ((m_inW * bmi->biBitCount + 31) / 32) * 4;
            m_isBGR    = (mtIn.subtype == MEDIASUBTYPE_RGB32 ||
                          mtIn.subtype == MEDIASUBTYPE_ARGB32 ||
                          mtIn.subtype == MEDIASUBTYPE_RGB24);
        }
    }

    int outW, outH;
    OutputDimensions(m_inW, m_inH, outW, outH);
    m_outBuf.resize(outW * outH * 4, 0);

    LOG_INFO("Input: ", m_inW, "x", m_inH,
             "  Output: ", outW, "x", outH,
             "  Mode: ",
             m_cfg.outputMode==OutputMode::SideBySide ? "SBS" : "TAB");
    return S_OK;
}

HRESULT C3DeflattenFilter::StopStreaming() {
    LOG_INFO("StopStreaming");
    return S_OK;
}

// ── Transform ────────────────────────────────────────────────────────────────
HRESULT C3DeflattenFilter::Transform(IMediaSample* pIn, IMediaSample* pOut) {
    BYTE* pSrc = nullptr;
    BYTE* pDst = nullptr;
    pIn->GetPointer(&pSrc);
    pOut->GetPointer(&pDst);

    // ── YUV->BGRA conversion ──────────────────────────────────────────────────
    std::vector<BYTE> rgba;
    const BYTE* rgbaPtr = pSrc;
    int         rgbaStride = m_inStride;

    CMediaType mtIn;
    m_pInput->ConnectionMediaType(&mtIn);

    if (mtIn.subtype == MEDIASUBTYPE_YUY2) {
        rgba.resize(m_inW * m_inH * 4);
        auto clamp8 = [](int v) -> BYTE {
            return (BYTE)std::max(0, std::min(255, v));
        };
        for (int y=0; y<m_inH; ++y) {
            const BYTE* row = pSrc + y*m_inStride;
            BYTE* out = rgba.data() + y*m_inW*4;
            for (int x=0; x<m_inW; x+=2) {
                int Y0=row[x*2], Cb=row[x*2+1];
                int Y1=row[x*2+2], Cr=row[x*2+3];
                auto yuv=[&](int Y){
                    int C=Y-16, D=Cb-128, E=Cr-128;
                    return std::tuple{
                        clamp8((298*C+409*E+128)>>8),
                        clamp8((298*C-100*D-208*E+128)>>8),
                        clamp8((298*C+516*D+128)>>8)};
                };
                auto[r0,g0,b0]=yuv(Y0);
                out[0]=b0;out[1]=g0;out[2]=r0;out[3]=255;
                auto[r1,g1,b1]=yuv(Y1);
                out[4]=b1;out[5]=g1;out[6]=r1;out[7]=255;
                out+=8;
            }
        }
        rgbaPtr    = rgba.data();
        rgbaStride = m_inW*4;
    }

    // ── Depth estimation ──────────────────────────────────────────────────────
    DeflattenConfig cfg;
    { CAutoLock lk(&m_csConfig); cfg = m_cfg; }

    DepthResult depthResult;
    bool haveDepth = false;

    if (m_depth->IsLoaded()) {
        haveDepth = SUCCEEDED(
            m_depth->Estimate(rgbaPtr, m_inW, m_inH, rgbaStride,
                              m_isBGR, cfg.flipDepth==TRUE,
                              cfg.depthSmooth, depthResult));
        if (!haveDepth) LOG_WARN("Depth estimation failed for frame");
    }

    if (!haveDepth) {
        depthResult.data.assign(m_inW * m_inH, 0.5f);
        depthResult.width  = m_inW;
        depthResult.height = m_inH;
    }

    // ── Stereo render ─────────────────────────────────────────────────────────
    int outW, outH;
    OutputDimensions(m_inW, m_inH, outW, outH);
    int outStride = outW*4;

    if ((int)m_outBuf.size() < outStride*outH)
        m_outBuf.resize(outStride*outH, 0);

    m_stereo->Render(rgbaPtr, m_inW, m_inH, rgbaStride,
                     depthResult.data.data(), cfg,
                     m_outBuf.data(), outStride);

    LONG needed = outStride * outH;
    if (pOut->GetSize() < needed) {
        LOG_ERR("Output sample too small: ", pOut->GetSize(), " < ", needed);
        return E_FAIL;
    }
    memcpy(pDst, m_outBuf.data(), needed);
    pOut->SetActualDataLength(needed);

    REFERENCE_TIME tStart, tStop;
    if (SUCCEEDED(pIn->GetTime(&tStart, &tStop)))
        pOut->SetTime(&tStart, &tStop);
    pOut->SetSyncPoint(TRUE);
    pOut->SetDiscontinuity(pIn->IsDiscontinuity()==S_OK);

    return S_OK;
}

// ── I3Deflatten ───────────────────────────────────────────────────────────────
STDMETHODIMP C3DeflattenFilter::GetConfig(DeflattenConfig* p) {
    if (!p) return E_POINTER;
    CAutoLock lk(&m_csConfig); *p = m_cfg; return S_OK;
}
STDMETHODIMP C3DeflattenFilter::SetConfig(const DeflattenConfig* p) {
    if (!p) return E_POINTER;
    CAutoLock lk(&m_csConfig);
    m_cfg = *p;
    LOG_INFO("Config: conv=", m_cfg.convergence,
             " sep=", m_cfg.separation,
             " mode=", (int)m_cfg.outputMode);
    return S_OK;
}
STDMETHODIMP C3DeflattenFilter::GetModelPath(LPWSTR buf, UINT cch) {
    if (!buf) return E_POINTER;
    wcsncpy_s(buf, cch, m_modelPath.c_str(), _TRUNCATE);
    return S_OK;
}
STDMETHODIMP C3DeflattenFilter::SetModelPath(LPCWSTR path) {
    if (!path) return E_POINTER;
    m_modelPath = path; return S_OK;
}
STDMETHODIMP C3DeflattenFilter::GetGPUInfo(LPWSTR buf, UINT cch) {
    if (!buf) return E_POINTER;
    wcsncpy_s(buf, cch, m_gpuInfo.c_str(), _TRUNCATE);
    return S_OK;
}
STDMETHODIMP C3DeflattenFilter::ReloadModel() {
    LOG_INFO("ReloadModel requested");
    return m_depth->Load(m_modelPath, m_cfg.gpuProvider, m_gpuInfo);
}

STDMETHODIMP C3DeflattenFilter::GetPages(CAUUID* pPages) {
    if (!pPages) return E_POINTER;
    pPages->cElems = 1;
    pPages->pElems = static_cast<GUID*>(CoTaskMemAlloc(sizeof(GUID)));
    if (!pPages->pElems) return E_OUTOFMEMORY;
    pPages->pElems[0] = CLSID_3DeflattenProp;
    return S_OK;
}
