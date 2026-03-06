// SPDX-License-Identifier: GPL-3.0-or-later
#include "filter.h"
#include "prop_page.h"
#include "guids.h"
#include <dvdmedia.h>
#include <uuids.h>
#include <algorithm>
#include <chrono>
#include <sstream>

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

// ── Helpers ───────────────────────────────────────────────────────────────────
static std::string GuidToSubtypeName(const GUID& g) {
    if (g == MEDIASUBTYPE_RGB32)  return "RGB32";
    if (g == MEDIASUBTYPE_ARGB32) return "ARGB32";
    if (g == MEDIASUBTYPE_RGB24)  return "RGB24";
    if (g == MEDIASUBTYPE_YUY2)   return "YUY2";
    if (g == MEDIASUBTYPE_NV12)   return "NV12";
    char buf[64];
    snprintf(buf, sizeof(buf),
        "{%08lX-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}",
        g.Data1, g.Data2, g.Data3,
        g.Data4[0], g.Data4[1], g.Data4[2], g.Data4[3],
        g.Data4[4], g.Data4[5], g.Data4[6], g.Data4[7]);
    return buf;
}

static std::string HRStr(HRESULT hr) {
    char buf[32];
    snprintf(buf, sizeof(buf), "0x%08X", (unsigned)hr);
    return buf;
}

static std::string MediaTypeDesc(const CMediaType* pmt) {
    if (!pmt) return "(null)";
    std::string s = GuidToSubtypeName(pmt->subtype);
    auto* bmi = reinterpret_cast<const BITMAPINFOHEADER*>(pmt->Format());
    if (bmi)
        s += " " + std::to_string(bmi->biWidth) + "x"
                 + std::to_string(abs(bmi->biHeight))
                 + " bpp=" + std::to_string(bmi->biBitCount);
    return s;
}

// ── CreateInstance ────────────────────────────────────────────────────────────
CUnknown* WINAPI C3DeflattenFilter::CreateInstance(LPUNKNOWN pUnk,
                                                    HRESULT* phr) {
    LOG_INFO("CreateInstance");
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
    if (GetEnvironmentVariableW(L"DEFLATTEN_MODEL_PATH", envModel, MAX_PATH)) {
        m_modelPath = envModel;
        LOG_INFO("Model path from env: ", m_modelPath);
    }

    m_depth  = std::make_unique<DepthEstimator>();
    m_stereo = std::make_unique<StereoRenderer>();

    LOG_INFO("C3DeflattenFilter constructed  convergence=", m_cfg.convergence,
             " separation=", m_cfg.separation,
             " mode=", m_cfg.outputMode==OutputMode::SideBySide?"SBS":"TAB",
             " gpu=", (int)m_cfg.gpuProvider);
    if (phr) *phr = S_OK;
}

C3DeflattenFilter::~C3DeflattenFilter() {
    LOG_INFO("C3DeflattenFilter destroyed  frames_processed=", m_frameCount);
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
    if (pmt->majortype != MEDIATYPE_Video) {
        LOG_DBG("CheckInputType REJECTED (not video): ",
                GuidToSubtypeName(pmt->subtype));
        return VFW_E_TYPE_NOT_ACCEPTED;
    }
    static const GUID* allowed[] = {
        &MEDIASUBTYPE_RGB32,  &MEDIASUBTYPE_ARGB32,
        &MEDIASUBTYPE_RGB24,  &MEDIASUBTYPE_YUY2,
        &MEDIASUBTYPE_NV12,
    };
    for (auto* g : allowed) {
        if (pmt->subtype == *g) {
            LOG_DBG("CheckInputType ACCEPTED: ", MediaTypeDesc(pmt));
            return S_OK;
        }
    }
    LOG_WARN("CheckInputType REJECTED (unsupported subtype): ",
             GuidToSubtypeName(pmt->subtype));
    return VFW_E_TYPE_NOT_ACCEPTED;
}

// ── GetMediaType ──────────────────────────────────────────────────────────────
HRESULT C3DeflattenFilter::GetMediaType(int iPos, CMediaType* pmt) {
    if (iPos < 0) return E_INVALIDARG;
    if (iPos > 0) return VFW_S_NO_MORE_ITEMS;
    if (!m_pInput || !m_pInput->IsConnected()) {
        LOG_WARN("GetMediaType called but input not connected");
        return E_UNEXPECTED;
    }
    CMediaType mtIn;
    m_pInput->ConnectionMediaType(&mtIn);
    HRESULT hr = BuildOutputMediaType(&mtIn, pmt);
    LOG_DBG("GetMediaType[", iPos, "] in=", MediaTypeDesc(&mtIn),
            " -> out=", MediaTypeDesc(pmt), " hr=", HRStr(hr));
    return hr;
}

HRESULT C3DeflattenFilter::BuildOutputMediaType(const CMediaType* pmtIn,
                                                  CMediaType* pmtOut) {
    *pmtOut = *pmtIn;
    auto* bmiIn = reinterpret_cast<BITMAPINFOHEADER*>(pmtIn->Format());
    if (!bmiIn) {
        LOG_ERR("BuildOutputMediaType: no BITMAPINFOHEADER in input type");
        return E_FAIL;
    }

    int inW = bmiIn->biWidth, inH = abs(bmiIn->biHeight);
    int outW, outH;
    OutputDimensions(inW, inH, outW, outH);

    auto* bmiOut = reinterpret_cast<BITMAPINFOHEADER*>(pmtOut->Format());
    bmiOut->biWidth       = outW;
    bmiOut->biHeight      = (bmiIn->biHeight < 0) ? -outH : outH;
    pmtOut->subtype       = MEDIASUBTYPE_RGB32;
    bmiOut->biBitCount    = 32;
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
    if (FAILED(BuildOutputMediaType(pmtIn, &proposed))) {
        LOG_WARN("CheckTransform: BuildOutputMediaType failed for in=",
                 MediaTypeDesc(pmtIn));
        return VFW_E_TYPE_NOT_ACCEPTED;
    }
    auto* b1 = reinterpret_cast<BITMAPINFOHEADER*>(proposed.Format());
    auto* b2 = reinterpret_cast<BITMAPINFOHEADER*>(pmtOut->Format());
    if (!b1 || !b2) return VFW_E_TYPE_NOT_ACCEPTED;
    if (b1->biWidth != b2->biWidth) {
        LOG_WARN("CheckTransform: width mismatch proposed=", b1->biWidth,
                 " offered=", b2->biWidth);
        return VFW_E_TYPE_NOT_ACCEPTED;
    }
    if (abs(b1->biHeight) != abs(b2->biHeight)) {
        LOG_WARN("CheckTransform: height mismatch proposed=", abs(b1->biHeight),
                 " offered=", abs(b2->biHeight));
        return VFW_E_TYPE_NOT_ACCEPTED;
    }
    LOG_DBG("CheckTransform OK  in=", MediaTypeDesc(pmtIn),
            " out=", MediaTypeDesc(pmtOut));
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
    HRESULT hr = pAlloc->SetProperties(pProps, &actual);
    LOG_INFO("DecideBufferSize: ", outW, "x", outH,
             " buf=", pProps->cbBuffer, " hr=", HRStr(hr));
    return hr;
}

// ── StartStreaming ────────────────────────────────────────────────────────────
HRESULT C3DeflattenFilter::StartStreaming() {
    LOG_INFO("===== StartStreaming =====");
    m_frameCount = 0;

    HRESULT hr = m_stereo->Init();
    if (FAILED(hr)) {
        LOG_ERR("StereoRenderer::Init FAILED hr=", HRStr(hr));
        return hr;
    }
    LOG_INFO("StereoRenderer::Init OK  gpuAvailable=",
             m_stereo->IsGPUAvailable() ? "yes" : "no");

    if (!m_depth->IsLoaded()) {
        LOG_INFO("Loading depth model  path='", m_modelPath,
                 "'  provider=", (int)m_cfg.gpuProvider);
        hr = m_depth->Load(m_modelPath, m_cfg.gpuProvider, m_gpuInfo);
        if (FAILED(hr)) {
            LOG_ERR("DepthEstimator::Load FAILED hr=", HRStr(hr),
                    " -- frames will use flat depth map (no 3D effect)");
        } else {
            LOG_INFO("DepthEstimator::Load OK  gpuInfo='", m_gpuInfo, "'");
        }
    } else {
        LOG_INFO("Depth model already loaded  gpuInfo='", m_gpuInfo, "'");
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
            LOG_INFO("Input media type: ", MediaTypeDesc(&mtIn),
                     " stride=", m_inStride,
                     " isBGR=", m_isBGR ? "yes" : "no");
        } else {
            LOG_ERR("StartStreaming: no BITMAPINFOHEADER -- dimensions unknown");
        }
    } else {
        LOG_ERR("StartStreaming: input pin not connected");
    }

    int outW, outH;
    OutputDimensions(m_inW, m_inH, outW, outH);
    m_outBuf.resize(outW * outH * 4, 0);

    LOG_INFO("Pipeline: ", m_inW, "x", m_inH, " -> ", outW, "x", outH,
             " mode=", m_cfg.outputMode==OutputMode::SideBySide?"SBS":"TAB",
             " conv=", m_cfg.convergence,
             " sep=", m_cfg.separation,
             " smooth=", m_cfg.depthSmooth,
             " flip=", m_cfg.flipDepth?"yes":"no");
    LOG_INFO("===== StartStreaming done =====");
    return S_OK;
}

HRESULT C3DeflattenFilter::StopStreaming() {
    LOG_INFO("StopStreaming  frames_processed=", m_frameCount);
    return S_OK;
}

// ── Transform ────────────────────────────────────────────────────────────────
HRESULT C3DeflattenFilter::Transform(IMediaSample* pIn, IMediaSample* pOut) {
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();

    ++m_frameCount;
    const bool logThis = (m_frameCount == 1)
                      || (m_frameCount == 2)
                      || (m_frameCount % 100 == 0);

    if (logThis)
        LOG_DBG("Transform frame #", m_frameCount,
                " inLen=", pIn->GetActualDataLength(),
                " outSize=", pOut->GetSize());

    BYTE* pSrc = nullptr;
    BYTE* pDst = nullptr;
    if (FAILED(pIn->GetPointer(&pSrc)) || !pSrc) {
        LOG_ERR("Frame #", m_frameCount, ": GetPointer(in) failed");
        return E_FAIL;
    }
    if (FAILED(pOut->GetPointer(&pDst)) || !pDst) {
        LOG_ERR("Frame #", m_frameCount, ": GetPointer(out) failed");
        return E_FAIL;
    }

    // ── YUV->BGRA conversion ──────────────────────────────────────────────────
    std::vector<BYTE> rgba;
    const BYTE* rgbaPtr    = pSrc;
    int         rgbaStride = m_inStride;

    CMediaType mtIn;
    m_pInput->ConnectionMediaType(&mtIn);

    if (mtIn.subtype == MEDIASUBTYPE_YUY2) {
        if (logThis) LOG_DBG("Frame #", m_frameCount, ": converting YUY2->BGRA");
        rgba.resize(m_inW * m_inH * 4);
        auto clamp8 = [](int v) -> BYTE {
            return (BYTE)std::max(0, std::min(255, v));
        };
        for (int y = 0; y < m_inH; ++y) {
            const BYTE* row = pSrc + y * m_inStride;
            BYTE* out = rgba.data() + y * m_inW * 4;
            for (int x = 0; x < m_inW; x += 2) {
                int Y0 = row[x*2], Cb = row[x*2+1];
                int Y1 = row[x*2+2], Cr = row[x*2+3];
                auto yuv = [&](int Y) {
                    int C = Y-16, D = Cb-128, E = Cr-128;
                    return std::tuple{
                        clamp8((298*C+409*E+128)>>8),
                        clamp8((298*C-100*D-208*E+128)>>8),
                        clamp8((298*C+516*D+128)>>8)};
                };
                auto [r0,g0,b0] = yuv(Y0);
                out[0]=b0; out[1]=g0; out[2]=r0; out[3]=255;
                auto [r1,g1,b1] = yuv(Y1);
                out[4]=b1; out[5]=g1; out[6]=r1; out[7]=255;
                out += 8;
            }
        }
        rgbaPtr    = rgba.data();
        rgbaStride = m_inW * 4;
    } else if (mtIn.subtype == MEDIASUBTYPE_NV12) {
        LOG_WARN("Frame #", m_frameCount,
                 ": NV12 input not yet converted -- passing raw bytes");
    }

    // ── Depth estimation ──────────────────────────────────────────────────────
    DeflattenConfig cfg;
    { CAutoLock lk(&m_csConfig); cfg = m_cfg; }

    DepthResult depthResult;
    bool haveDepth = false;

    if (m_depth->IsLoaded()) {
        auto td0 = Clock::now();
        haveDepth = SUCCEEDED(
            m_depth->Estimate(rgbaPtr, m_inW, m_inH, rgbaStride,
                              m_isBGR, cfg.flipDepth == TRUE,
                              cfg.depthSmooth, depthResult));
        auto tdMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - td0).count();
        if (logThis)
            LOG_DBG("Frame #", m_frameCount,
                    ": depth inference ", haveDepth ? "OK" : "FAILED",
                    " (", tdMs, " ms)");
        if (!haveDepth)
            LOG_WARN("Frame #", m_frameCount, ": depth estimation failed");
    } else {
        if (logThis)
            LOG_WARN("Frame #", m_frameCount,
                     ": depth model not loaded -- using flat depth (no 3D effect)");
    }

    if (!haveDepth) {
        depthResult.data.assign(m_inW * m_inH, 0.5f);
        depthResult.width  = m_inW;
        depthResult.height = m_inH;
    }

    // ── Stereo render ─────────────────────────────────────────────────────────
    int outW, outH;
    OutputDimensions(m_inW, m_inH, outW, outH);
    int outStride = outW * 4;

    if ((int)m_outBuf.size() < outStride * outH)
        m_outBuf.resize(outStride * outH, 0);

    auto tr0 = Clock::now();
    m_stereo->Render(rgbaPtr, m_inW, m_inH, rgbaStride,
                     depthResult.data.data(), cfg,
                     m_outBuf.data(), outStride);
    auto trMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - tr0).count();

    LONG needed = outStride * outH;
    if (pOut->GetSize() < needed) {
        LOG_ERR("Frame #", m_frameCount,
                ": output sample too small: have=", pOut->GetSize(),
                " need=", needed);
        return E_FAIL;
    }
    memcpy(pDst, m_outBuf.data(), needed);
    pOut->SetActualDataLength(needed);

    REFERENCE_TIME tStart, tStop;
    if (SUCCEEDED(pIn->GetTime(&tStart, &tStop)))
        pOut->SetTime(&tStart, &tStop);
    pOut->SetSyncPoint(TRUE);
    pOut->SetDiscontinuity(pIn->IsDiscontinuity() == S_OK);

    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - t0).count();

    if (logThis)
        LOG_INFO("Frame #", m_frameCount,
                 ": render=", trMs, "ms  total=", totalMs, "ms"
                 "  depth=", haveDepth ? "AI" : "flat",
                 "  gpu=", m_stereo->IsGPUAvailable() ? "yes" : "no");

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
    LOG_INFO("SetConfig: conv=", m_cfg.convergence,
             " sep=", m_cfg.separation,
             " mode=", (int)m_cfg.outputMode,
             " gpu=", (int)m_cfg.gpuProvider,
             " smooth=", m_cfg.depthSmooth,
             " flip=", m_cfg.flipDepth ? "yes" : "no");
    return S_OK;
}
STDMETHODIMP C3DeflattenFilter::GetModelPath(LPWSTR buf, UINT cch) {
    if (!buf) return E_POINTER;
    wcsncpy_s(buf, cch, m_modelPath.c_str(), _TRUNCATE);
    return S_OK;
}
STDMETHODIMP C3DeflattenFilter::SetModelPath(LPCWSTR path) {
    if (!path) return E_POINTER;
    m_modelPath = path;
    LOG_INFO("SetModelPath: '", m_modelPath, "'");
    return S_OK;
}
STDMETHODIMP C3DeflattenFilter::GetGPUInfo(LPWSTR buf, UINT cch) {
    if (!buf) return E_POINTER;
    wcsncpy_s(buf, cch, m_gpuInfo.c_str(), _TRUNCATE);
    return S_OK;
}
STDMETHODIMP C3DeflattenFilter::ReloadModel() {
    LOG_INFO("ReloadModel requested  path='", m_modelPath, "'");
    HRESULT hr = m_depth->Load(m_modelPath, m_cfg.gpuProvider, m_gpuInfo);
    LOG_INFO("ReloadModel ", SUCCEEDED(hr) ? "OK" : "FAILED",
             " hr=", HRStr(hr), " gpuInfo='", m_gpuInfo, "'");
    return hr;
}

STDMETHODIMP C3DeflattenFilter::GetPages(CAUUID* pPages) {
    if (!pPages) return E_POINTER;
    pPages->cElems = 1;
    pPages->pElems = static_cast<GUID*>(CoTaskMemAlloc(sizeof(GUID)));
    if (!pPages->pElems) return E_OUTOFMEMORY;
    pPages->pElems[0] = CLSID_3DeflattenProp;
    return S_OK;
}
