// SPDX-License-Identifier: GPL-3.0-or-later
#include "filter.h"
#include "prop_page.h"
#include "guids.h"
#include <dvdmedia.h>
#include <uuids.h>
#include <algorithm>
#include <chrono>
#include <cmath>

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
    ARRAYSIZE(sudPins), sudPins
};

// ── Format helpers ────────────────────────────────────────────────────────────
// BITMAPINFOHEADER is inside VIDEOINFOHEADER/VIDEOINFOHEADER2, not at offset 0.
static const BITMAPINFOHEADER* GetBMI(const CMediaType* pmt) {
    if (!pmt || !pmt->Format() || pmt->FormatLength() == 0) return nullptr;
    if (pmt->formattype == FORMAT_VideoInfo &&
        pmt->FormatLength() >= sizeof(VIDEOINFOHEADER))
        return &reinterpret_cast<const VIDEOINFOHEADER*>(pmt->Format())->bmiHeader;
    if (pmt->formattype == FORMAT_VideoInfo2 &&
        pmt->FormatLength() >= sizeof(VIDEOINFOHEADER2))
        return &reinterpret_cast<const VIDEOINFOHEADER2*>(pmt->Format())->bmiHeader;
    if (pmt->FormatLength() >= sizeof(BITMAPINFOHEADER))
        return reinterpret_cast<const BITMAPINFOHEADER*>(pmt->Format());
    return nullptr;
}

static std::string GuidName(const GUID& g) {
    if (g == MEDIASUBTYPE_RGB32)  return "RGB32";
    if (g == MEDIASUBTYPE_ARGB32) return "ARGB32";
    if (g == MEDIASUBTYPE_RGB24)  return "RGB24";
    if (g == MEDIASUBTYPE_YUY2)   return "YUY2";
    if (g == MEDIASUBTYPE_NV12)   return "NV12";
    char buf[64];
    snprintf(buf,sizeof(buf),"{%08lX-%04X-%04X-%02X%02X-%02X%02X%02X%02X%02X%02X}",
        g.Data1,g.Data2,g.Data3,
        g.Data4[0],g.Data4[1],g.Data4[2],g.Data4[3],
        g.Data4[4],g.Data4[5],g.Data4[6],g.Data4[7]);
    return buf;
}
static std::string HRStr(HRESULT hr)
    { char b[16]; snprintf(b,sizeof(b),"0x%08X",(unsigned)hr); return b; }
static std::string MTDesc(const CMediaType* pmt) {
    if (!pmt) return "(null)";
    std::string s = GuidName(pmt->subtype);
    auto* b = GetBMI(pmt);
    if (b) s += " " + std::to_string(b->biWidth)+"x"+std::to_string(abs((int)b->biHeight));
    return s;
}

// ── INI persistence ───────────────────────────────────────────────────────────
// Stores/loads settings from 3Deflatten.ini next to the .ax file.
// Using the Win32 private-profile API so the INI is plain, human-editable text.
// File location: resolved from the .ax module's own path so it stays with the
// filter regardless of working directory or COM host.

std::wstring C3DeflattenFilter::GetIniPath() {
    // Resolve HMODULE of this .ax by searching from the address of CreateInstance.
    HMODULE hm = nullptr;
    GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&C3DeflattenFilter::CreateInstance), &hm);
    wchar_t path[MAX_PATH] = {};
    if (hm) GetModuleFileNameW(hm, path, MAX_PATH);
    // Replace the filename part with 3Deflatten.ini
    wchar_t* sl = wcsrchr(path, L'\\');
    if (sl) wcscpy_s(sl + 1, MAX_PATH - (DWORD)(sl - path + 1), L"3Deflatten.ini");
    else     wcscpy_s(path, L"3Deflatten.ini");
    return path;
}

void C3DeflattenFilter::LoadIni() {
    if (m_iniPath.empty()) m_iniPath = GetIniPath();
    const wchar_t* p = m_iniPath.c_str();
    const wchar_t* s = L"3Deflatten";

    auto getF = [&](const wchar_t* key, float def) -> float {
        wchar_t buf[64] = {};
        GetPrivateProfileStringW(s, key, L"", buf, ARRAYSIZE(buf), p);
        if (buf[0] == L'\0') return def;
        return (float)_wtof(buf);
    };
    auto getI = [&](const wchar_t* key, int def) -> int {
        return (int)GetPrivateProfileIntW(s, key, def, p);
    };
    auto getStr = [&](const wchar_t* key) -> std::wstring {
        wchar_t buf[MAX_PATH] = {};
        GetPrivateProfileStringW(s, key, L"", buf, ARRAYSIZE(buf), p);
        return buf;
    };

    m_cfg.convergence  = getF(L"convergence", 0.250f);
    m_cfg.separation   = getF(L"separation",  0.050f);
    m_cfg.depthSmooth  = getF(L"depthSmooth", 0.0f);
    m_cfg.outputMode   = (OutputMode)getI(L"outputMode",  (int)OutputMode::SideBySide);
    m_cfg.gpuProvider  = (GPUProvider)getI(L"gpuProvider", (int)GPUProvider::Auto);
    m_cfg.flipDepth    = getI(L"flipDepth", 0) ? TRUE : FALSE;
    m_cfg.infillMode   = (InfillMode)getI(L"infillMode", (int)InfillMode::Outer);
    m_cfg.showDepth    = getI(L"showDepth", 0) ? TRUE : FALSE;
    m_cfg.depthViewKey = getI(L"depthViewKey", 161);  // 161 = VK_RSHIFT
    m_cfg.inferenceRuntime = (InferenceRuntime)getI(L"inferenceRuntime", 0);
    m_cfg.depthMaxDim      = getI(L"depthMaxDim", 0);
    m_cfg.meshDiv          = getI(L"meshDiv", 2);
    m_cfg.depthDilate      = getI(L"depthDilate", 4);
    m_cfg.depthEdgeThresh  = getF(L"depthEdgeThresh", 0.20f);
    m_cfg.depthJBU         = getI(L"depthJBU", 0) ? TRUE : FALSE;

    std::wstring mp = getStr(L"modelPath");
    if (!mp.empty()) m_modelPath = mp;

    LOG_INFO("LoadIni: '", m_iniPath, "'",
             " conv=", m_cfg.convergence, " sep=", m_cfg.separation,
             " smooth=", m_cfg.depthSmooth, " flip=", m_cfg.flipDepth?"y":"n",
             " mode=", (int)m_cfg.outputMode, " gpu=", (int)m_cfg.gpuProvider);
}

void C3DeflattenFilter::SaveIni() const {
    const wchar_t* p = m_iniPath.c_str();
    const wchar_t* s = L"3Deflatten";

    auto setF = [&](const wchar_t* key, float v) {
        wchar_t buf[32]; swprintf_s(buf, L"%.6g", v);
        WritePrivateProfileStringW(s, key, buf, p);
    };
    auto setI = [&](const wchar_t* key, int v) {
        wchar_t buf[16]; swprintf_s(buf, L"%d", v);
        WritePrivateProfileStringW(s, key, buf, p);
    };

    setF(L"convergence",  m_cfg.convergence);
    setF(L"separation",   m_cfg.separation);
    setF(L"depthSmooth",  m_cfg.depthSmooth);
    setI(L"outputMode",   (int)m_cfg.outputMode);
    setI(L"gpuProvider",  (int)m_cfg.gpuProvider);
    setI(L"flipDepth",    m_cfg.flipDepth ? 1 : 0);
    setI(L"infillMode",   (int)m_cfg.infillMode);
    setI(L"showDepth",         m_cfg.showDepth ? 1 : 0);
    setI(L"depthViewKey",      m_cfg.depthViewKey);
    setI(L"inferenceRuntime",  (int)m_cfg.inferenceRuntime);
    setI(L"depthMaxDim",       m_cfg.depthMaxDim);
    setI(L"meshDiv",           m_cfg.meshDiv);
    setI(L"depthDilate",       m_cfg.depthDilate);
    setF(L"depthEdgeThresh",   m_cfg.depthEdgeThresh);
    setI(L"depthJBU",          m_cfg.depthJBU ? 1 : 0);
    WritePrivateProfileStringW(s, L"modelPath", m_modelPath.c_str(), p);

    LOG_INFO("SaveIni: '", m_iniPath, "'");
}

// ── CreateInstance ────────────────────────────────────────────────────────────
CUnknown* WINAPI C3DeflattenFilter::CreateInstance(LPUNKNOWN pUnk, HRESULT* phr) {
    LOG_INFO("CreateInstance");
    return new C3DeflattenFilter(pUnk, phr);
}

C3DeflattenFilter::C3DeflattenFilter(LPUNKNOWN pUnk, HRESULT* phr)
    : CTransformFilter(L"3Deflatten", pUnk, CLSID_3Deflatten)
{
    m_cfg.convergence   = 0.250f;
    m_cfg.separation    = 0.050f;
    m_cfg.outputMode    = OutputMode::SideBySide;
    m_cfg.gpuProvider   = GPUProvider::Auto;
    m_cfg.depthSmooth   = 0.0f;
    m_cfg.flipDepth     = FALSE;
    m_cfg.infillMode    = InfillMode::Outer;
    m_cfg.showDepth     = FALSE;
    m_cfg.depthViewKey  = 161;  // VK_RSHIFT
    m_cfg.inferenceRuntime = InferenceRuntime::OnnxRuntime;
    m_cfg.depthMaxDim      = 0;
    m_cfg.meshDiv          = 2;
    m_cfg.depthDilate      = 4;
    m_cfg.depthEdgeThresh  = 0.20f;
    m_cfg.depthJBU         = FALSE;
    m_hadRealDepth    = false;
    m_skipEvery       = 1;
    m_skipCounter     = 0;
    m_avgInferMs      = 0.0;
    m_accumDx = 0.f; m_accumDy = 0.f;
    m_lumaSmW = 0; m_lumaSmH = 0;

    wchar_t envModel[MAX_PATH] = {};
    if (GetEnvironmentVariableW(L"DEFLATTEN_MODEL_PATH", envModel, MAX_PATH))
        m_modelPath = envModel;

    // Load persisted settings from INI (overrides defaults above).
    // modelPath from INI is only used if DEFLATTEN_MODEL_PATH env var is not set.
    m_iniPath = GetIniPath();
    LoadIni();
    // Env var wins over INI model path
    if (envModel[0]) m_modelPath = envModel;

    m_depth  = std::make_unique<DepthEstimator>();
    m_stereo = std::make_unique<StereoRenderer>();
    LOG_INFO("C3DeflattenFilter constructed");
    if (phr) *phr = S_OK;
}
C3DeflattenFilter::~C3DeflattenFilter() {
    StopDepthThread();
    LOG_INFO("C3DeflattenFilter destroyed  frames=", m_frameCount);
}

STDMETHODIMP C3DeflattenFilter::NonDelegatingQueryInterface(REFIID riid, void** ppv) {
    if (riid == IID_I3Deflatten)
        return GetInterface(static_cast<I3Deflatten*>(this), ppv);
    if (riid == IID_ISpecifyPropertyPages)
        return GetInterface(static_cast<ISpecifyPropertyPages*>(this), ppv);
    return CTransformFilter::NonDelegatingQueryInterface(riid, ppv);
}

// ── CheckInputType ────────────────────────────────────────────────────────────
HRESULT C3DeflattenFilter::CheckInputType(const CMediaType* pmt) {
    if (pmt->majortype != MEDIATYPE_Video) return VFW_E_TYPE_NOT_ACCEPTED;
    static const GUID* ok[] = {
        &MEDIASUBTYPE_RGB32,&MEDIASUBTYPE_ARGB32,&MEDIASUBTYPE_RGB24,
        &MEDIASUBTYPE_YUY2,&MEDIASUBTYPE_NV12
    };
    for (auto* g : ok) {
        if (pmt->subtype == *g) {
            LOG_DBG("CheckInputType accepted: ", MTDesc(pmt));
            return S_OK;
        }
    }
    LOG_WARN("CheckInputType rejected: ", GuidName(pmt->subtype));
    return VFW_E_TYPE_NOT_ACCEPTED;
}

// ── BuildOutputMediaType ──────────────────────────────────────────────────────
HRESULT C3DeflattenFilter::BuildOutputMediaType(const CMediaType* pmtIn,
                                                  CMediaType* pmtOut) {
    const BITMAPINFOHEADER* bmiIn = GetBMI(pmtIn);
    if (!bmiIn) return E_FAIL;

    int inW = bmiIn->biWidth, inH = abs((int)bmiIn->biHeight);
    int outW, outH;
    OutputDimensions(inW, inH, outW, outH);

    pmtOut->SetType(&MEDIATYPE_Video);
    pmtOut->SetSubtype(&MEDIASUBTYPE_RGB32);
    pmtOut->SetFormatType(&FORMAT_VideoInfo);
    pmtOut->SetTemporalCompression(FALSE);
    pmtOut->SetSampleSize(outW * outH * 4);

    VIDEOINFOHEADER vih = {};
    if (pmtIn->formattype == FORMAT_VideoInfo &&
        pmtIn->FormatLength() >= sizeof(VIDEOINFOHEADER)) {
        auto* vIn = reinterpret_cast<const VIDEOINFOHEADER*>(pmtIn->Format());
        vih.AvgTimePerFrame = vIn->AvgTimePerFrame;
        vih.dwBitRate = inW && inH
            ? (DWORD)((double)vIn->dwBitRate * outW * outH / (inW * inH)) : 0;
    } else if (pmtIn->formattype == FORMAT_VideoInfo2 &&
               pmtIn->FormatLength() >= sizeof(VIDEOINFOHEADER2)) {
        auto* vIn = reinterpret_cast<const VIDEOINFOHEADER2*>(pmtIn->Format());
        vih.AvgTimePerFrame = vIn->AvgTimePerFrame;
        vih.dwBitRate = inW && inH
            ? (DWORD)((double)vIn->dwBitRate * outW * outH / (inW * inH)) : 0;
    }
    vih.rcSource = {0,0,outW,outH};
    vih.rcTarget = vih.rcSource;
    vih.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    vih.bmiHeader.biWidth       = outW;
    vih.bmiHeader.biHeight      = (bmiIn->biHeight < 0) ? -outH : outH;
    vih.bmiHeader.biPlanes      = 1;
    vih.bmiHeader.biBitCount    = 32;
    vih.bmiHeader.biCompression = BI_RGB;
    vih.bmiHeader.biSizeImage   = outW * outH * 4;
    pmtOut->SetFormat(reinterpret_cast<BYTE*>(&vih), sizeof(vih));
    return S_OK;
}

void C3DeflattenFilter::OutputDimensions(int inW, int inH,
                                          int& outW, int& outH) const {
    CAutoLock lk(const_cast<CCritSec*>(&m_csConfig));
    if (m_cfg.outputMode == OutputMode::SideBySide) { outW=inW*2; outH=inH; }
    else { outW=inW; outH=inH*2; }
}

HRESULT C3DeflattenFilter::GetMediaType(int iPos, CMediaType* pmt) {
    if (iPos < 0) return E_INVALIDARG;
    if (iPos > 0) return VFW_S_NO_MORE_ITEMS;
    if (!m_pInput || !m_pInput->IsConnected()) return E_UNEXPECTED;
    CMediaType mtIn; m_pInput->ConnectionMediaType(&mtIn);
    HRESULT hr = BuildOutputMediaType(&mtIn, pmt);
    LOG_DBG("GetMediaType[",iPos,"] ",MTDesc(&mtIn)," -> ",MTDesc(pmt)," ",HRStr(hr));
    return hr;
}

HRESULT C3DeflattenFilter::CheckTransform(const CMediaType* pmtIn,
                                           const CMediaType* pmtOut) {
    CMediaType prop;
    if (FAILED(BuildOutputMediaType(pmtIn, &prop))) return VFW_E_TYPE_NOT_ACCEPTED;
    auto* b1=GetBMI(&prop), *b2=GetBMI(pmtOut);
    if (!b1||!b2) return VFW_E_TYPE_NOT_ACCEPTED;
    if (b1->biWidth != b2->biWidth ||
        abs((int)b1->biHeight) != abs((int)b2->biHeight)) {
        LOG_WARN("CheckTransform: size mismatch");
        return VFW_E_TYPE_NOT_ACCEPTED;
    }
    LOG_DBG("CheckTransform OK in=",MTDesc(pmtIn)," out=",MTDesc(pmtOut));
    return S_OK;
}

HRESULT C3DeflattenFilter::DecideBufferSize(IMemAllocator* pAlloc,
                                             ALLOCATOR_PROPERTIES* pProps) {
    ASSERT(m_pInput->IsConnected());
    CMediaType mtIn; m_pInput->ConnectionMediaType(&mtIn);
    auto* bmi = GetBMI(&mtIn);
    if (!bmi) return E_FAIL;
    int outW, outH;
    OutputDimensions(bmi->biWidth, abs((int)bmi->biHeight), outW, outH);
    pProps->cBuffers=1; pProps->cbBuffer=outW*outH*4;
    pProps->cbAlign=1;  pProps->cbPrefix=0;
    ALLOCATOR_PROPERTIES actual;
    HRESULT hr = pAlloc->SetProperties(pProps, &actual);
    LOG_INFO("DecideBufferSize: ",outW,"x",outH," buf=",pProps->cbBuffer," ",HRStr(hr));
    return hr;
}

// ── Async depth thread ────────────────────────────────────────────────────────
void C3DeflattenFilter::StartDepthThread() {
    StopDepthThread();   // clean up any previous thread
    m_pendStop = m_pendReady = false;
    m_cacheReady = false;
    m_depthThread = std::thread([this]{ DepthWorkerThread(); });
    LOG_INFO("Depth worker thread started");
}

void C3DeflattenFilter::StopDepthThread() {
    {
        std::lock_guard<std::mutex> lk(m_pendMtx);
        m_pendStop  = true;
        m_pendReady = true;   // wake the thread if it's waiting
    }
    m_pendCV.notify_one();
    if (m_depthThread.joinable()) {
        m_depthThread.join();
        LOG_INFO("Depth worker thread stopped");
    }
}

void C3DeflattenFilter::StartHotkeyThread() {
    StopHotkeyThread();
    m_hotkeyStop.store(false);
    m_hotkeyThread = std::thread([this]{ HotkeyThread(); });
}

void C3DeflattenFilter::StopHotkeyThread() {
    m_hotkeyStop.store(true);
    if (m_hotkeyThread.joinable())
        m_hotkeyThread.join();
}

void C3DeflattenFilter::HotkeyThread() {
    // Poll the configured virtual key at ~30 ms intervals.
    // Toggle showDepth on key-down edge (was-up → is-down).
    bool prevDown = false;
    while (!m_hotkeyStop.load()) {
        int vk = 0;
        { CAutoLock lk(&m_csConfig); vk = m_cfg.depthViewKey; }
        if (vk > 0) {
            bool down = (GetAsyncKeyState(vk) & 0x8000) != 0;
            if (down && !prevDown) {
                CAutoLock lk(&m_csConfig);
                m_cfg.showDepth = m_cfg.showDepth ? FALSE : TRUE;
                LOG_INFO("Depth view hotkey: showDepth=", m_cfg.showDepth ? "on" : "off");
            }
            prevDown = down;
        }
        // ~30 ms poll interval — low enough to feel instant, low CPU cost
        for (int i = 0; i < 3 && !m_hotkeyStop.load(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void C3DeflattenFilter::DepthWorkerThread() {
    // bgra is declared OUTSIDE the loop so it retains its 8 MB allocation
    // across iterations.  Swapping with m_pendBGRA (instead of moving) means
    // m_pendBGRA also always keeps its capacity → Transform() never reallocs.
    std::vector<BYTE> bgra;
    for (;;) {
        int w = 0, h = 0, slot = -1;
        {
            std::unique_lock<std::mutex> lk(m_pendMtx);
            m_pendCV.wait(lk, [this]{ return m_pendReady || m_pendStop; });
            if (m_pendStop) break;
            bgra.swap(m_pendBGRA);
            w    = m_pendW;
            h    = m_pendH;
            slot = m_pendSlot;
            m_pendReady = false;
        }
        if (bgra.empty() || w == 0 || h == 0) continue;

        DeflattenConfig cfg;
        { CAutoLock lk(&m_csConfig); cfg = m_cfg; }

        DepthResult result;
        auto t0 = std::chrono::steady_clock::now();
        HRESULT hr = m_depth->Estimate(bgra.data(), w, h, w*4,
                                        true /*isBGR*/,
                                        cfg.flipDepth == TRUE,
                                        cfg.depthSmooth,
                                        cfg.depthDilate,
                                        cfg.depthEdgeThresh,
                                        cfg.depthJBU == TRUE, result);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();

        if (SUCCEEDED(hr)) {
            std::lock_guard<std::mutex> lk(m_cacheMtx);
            m_cachedDepth = std::move(result.data);
            m_cachedW     = result.width;
            m_cachedH     = result.height;
            m_cachedSlot  = slot;   // which ring slot this depth was computed from
            m_cacheReady  = true;
            m_lastInferMs = (double)ms;
            LOG_DBG("Depth worker: inference done in ", ms, " ms");
        } else {
            LOG_WARN("Depth worker: inference failed hr=", HRStr(hr));
        }
    }
}

// ── Global motion estimation (Lucas-Kanade, whole-image translation) ─────────
// Treats the entire image as one patch; valid for global camera motion.
// Inputs are 8-bit luma images of dimensions w×h (small, ~128×72).
// Returns (outDx, outDy) in SOURCE-RESOLUTION pixels (scaled by srcW/w).
// Capped at ±(w*2) to suppress hard-cut failures.
/*static*/ void C3DeflattenFilter::EstimateMotionLK(
        const uint8_t* prev, const uint8_t* curr,
        int w, int h, float scale,
        float& outDx, float& outDy) {

    double sIx2  = 0, sIy2  = 0, sIxIy = 0;
    double sIxIt = 0, sIyIt = 0;
    for (int y = 1; y < h-1; ++y) {
        const uint8_t* pr = prev + y * w;
        const uint8_t* cr = curr + y * w;
        for (int x = 1; x < w-1; ++x) {
            double Ix = ((int)pr[x+1] - (int)pr[x-1]) * 0.5;
            double Iy = ((int)prev[(y+1)*w+x] - (int)prev[(y-1)*w+x]) * 0.5;
            double It = (int)cr[x] - (int)pr[x];
            sIx2  += Ix * Ix;
            sIy2  += Iy * Iy;
            sIxIy += Ix * Iy;
            sIxIt += Ix * It;
            sIyIt += Iy * It;
        }
    }
    double det = sIx2 * sIy2 - sIxIy * sIxIy;
    if (std::abs(det) < 1.0) { outDx = outDy = 0.f; return; }
    double dx = (-sIy2 * sIxIt + sIxIy * sIyIt) / det;
    double dy = ( sIxIy * sIxIt - sIx2 * sIyIt) / det;
    float cap = (float)w * 2.f;
    outDx = std::max(-cap, std::min(cap, (float)(dx * scale)));
    outDy = std::max(-cap, std::min(cap, (float)(dy * scale)));
}

// ── NV12 -> BGRA ─────────────────────────────────────────────────────────────
// Writes into a pre-allocated destination buffer (no resize inside hot path).
static void NV12toBGRA(const BYTE* src, int w, int h, int stride,
                        std::vector<BYTE>& dst) {
    // Caller pre-allocates dst to w*h*4; we only resize on first call / resolution change
    if ((int)dst.size() != w * h * 4)
        dst.resize(w * h * 4);
    const BYTE* yPlane  = src;
    const BYTE* uvPlane = src + stride * h;
    auto clamp8 = [](int v)->BYTE{ return (BYTE)(v<0?0:v>255?255:v); };
    for (int y = 0; y < h; ++y) {
        const BYTE* yr  = yPlane  + y * stride;
        const BYTE* uvr = uvPlane + (y/2) * stride;
        BYTE* out = dst.data() + y * w * 4;
        for (int x = 0; x < w; ++x) {
            int Y=yr[x], U=uvr[x&~1], V=uvr[(x&~1)+1];
            int C=Y-16, D=U-128, E=V-128;
            out[0]=clamp8((298*C+516*D+128)>>8);          // B
            out[1]=clamp8((298*C-100*D-208*E+128)>>8);    // G
            out[2]=clamp8((298*C+409*E+128)>>8);           // R
            out[3]=255;
            out+=4;
        }
    }
}

// ── StartStreaming / StopStreaming ────────────────────────────────────────────
HRESULT C3DeflattenFilter::StartStreaming() {
    LOG_INFO("===== StartStreaming =====");
    m_frameCount  = 0;
    m_skipCounter = 0;
    m_skipEvery   = 1;
    m_avgInferMs  = 0.0;
    m_accumDx = 0.f; m_accumDy = 0.f;
    m_prevLumaSmall.clear();
    m_lumaSmW = 0; m_lumaSmH = 0;

    HRESULT hr = m_stereo->Init();
    if (FAILED(hr)) { LOG_ERR("StereoRenderer::Init FAILED ",HRStr(hr)); return hr; }
    LOG_INFO("StereoRenderer::Init OK  gpu=", m_stereo->IsGPUAvailable()?"yes":"no");

    if (!m_depth->IsLoaded()) {
        LOG_INFO("Loading model path='", m_modelPath,
                 "' provider=", (int)m_cfg.gpuProvider);
        hr = m_depth->Load(m_modelPath, m_cfg.gpuProvider, m_cfg.inferenceRuntime, m_cfg.depthMaxDim, m_gpuInfo);
        if (FAILED(hr))
            LOG_ERR("Model load FAILED ", HRStr(hr), " - flat depth fallback");
        else {
            LOG_INFO("Model loaded OK  ep='", m_gpuInfo, "'");
            if (m_depth->IsStreaming())
                LOG_INFO("  Recurrent context streaming active.");
            if (m_depth->IsDA3Stream())
                LOG_INFO("  DA3-Streaming (sliding-window temporal alignment) active.");
        }
    } else {
        LOG_INFO("Model already loaded  ep='", m_gpuInfo, "'");
    }

    if (m_pInput && m_pInput->IsConnected()) {
        CMediaType mtIn; m_pInput->ConnectionMediaType(&mtIn);
        const auto* bmi = GetBMI(&mtIn);
        if (!bmi) { LOG_ERR("StartStreaming: can't read input format"); return E_FAIL; }
        m_inW  = bmi->biWidth;
        m_inH  = abs((int)bmi->biHeight);
        m_isNV12 = (mtIn.subtype == MEDIASUBTYPE_NV12);
        m_isYUY2 = (mtIn.subtype == MEDIASUBTYPE_YUY2);
        m_isBGR  = !m_isNV12 && !m_isYUY2;
        m_inStride = m_isNV12
            ? (m_inW + 15) & ~15
            : ((m_inW * bmi->biBitCount + 31) / 32) * 4;
        LOG_INFO("Input: ", MTDesc(&mtIn),
                 " stride=", m_inStride,
                 " isNV12=", m_isNV12?"yes":"no");
    }

    // Pre-allocate per-frame buffers so Transform() never calls malloc.
    size_t bgraBytes = (size_t)m_inW * m_inH * 4;
    m_convBuf.assign(bgraBytes, 0);
    m_pendBGRA.assign(bgraBytes, 0);

    // IMPORTANT: do NOT overwrite m_depthRender if it already has real depth
    // from a previous session.  Overwriting with flat causes visible flash on
    // every pause/resume.  Only initialise flat on the very first session.
    size_t depthN = (size_t)m_inW * m_inH;
    if (!m_hadRealDepth || m_depthRender.size() != depthN)
        m_depthRender.assign(depthN, 0.5f);

    int outW, outH;
    OutputDimensions(m_inW, m_inH, outW, outH);
    LOG_INFO("Pipeline: ", m_inW,"x",m_inH," -> ",outW,"x",outH,
             " mode=", m_cfg.outputMode==OutputMode::SideBySide?"SBS":"TAB",
             " conv=",m_cfg.convergence," sep=",m_cfg.separation,
             " smooth=",m_cfg.depthSmooth," flip=",m_cfg.flipDepth?"yes":"no");

    StartDepthThread();
    StartHotkeyThread();
    LOG_INFO("===== StartStreaming done =====");
    return S_OK;
}

HRESULT C3DeflattenFilter::StopStreaming() {
    StopDepthThread();
    StopHotkeyThread();
    if (m_depth) m_depth->ResetStreamingContext();
    LOG_INFO("StopStreaming  frames=", m_frameCount);
    return S_OK;
}

// ── Transform ─────────────────────────────────────────────────────────────────
HRESULT C3DeflattenFilter::Transform(IMediaSample* pIn, IMediaSample* pOut) {
    using Clock = std::chrono::steady_clock;
    auto t0 = Clock::now();
    ++m_frameCount;
    const bool logInfo = (m_frameCount <= 2) || (m_frameCount % 100 == 0);

    BYTE *pSrc=nullptr, *pDst=nullptr;
    if (FAILED(pIn->GetPointer(&pSrc))||!pSrc ||
        FAILED(pOut->GetPointer(&pDst))||!pDst) {
        LOG_ERR("Frame #",m_frameCount,": GetPointer failed"); return E_FAIL;
    }

    // ── Convert to BGRA into pre-allocated m_convBuf (no malloc) ─────────────
    const BYTE* rgbaPtr    = pSrc;
    int         rgbaStride = m_inStride;

    if (m_isNV12) {
        NV12toBGRA(pSrc, m_inW, m_inH, m_inStride, m_convBuf);
        rgbaPtr = m_convBuf.data(); rgbaStride = m_inW*4;
        if (logInfo) LOG_DBG("Frame #",m_frameCount,": NV12->BGRA");
    } else if (m_isYUY2) {
        m_convBuf.resize(m_inW * m_inH * 4);
        auto cl=[](int v)->BYTE{return(BYTE)(v<0?0:v>255?255:v);};
        for (int y=0;y<m_inH;++y) {
            const BYTE* row=pSrc+y*m_inStride; BYTE* out=m_convBuf.data()+y*m_inW*4;
            for (int x=0;x<m_inW;x+=2) {
                int Y0=row[x*2],Cb=row[x*2+1],Y1=row[x*2+2],Cr=row[x*2+3];
                auto yuv=[&](int Y)->std::tuple<BYTE,BYTE,BYTE>{
                    int C=Y-16,D=Cb-128,E=Cr-128;
                    return{cl((298*C+409*E+128)>>8),
                           cl((298*C-100*D-208*E+128)>>8),
                           cl((298*C+516*D+128)>>8)};};
                auto[r0,g0,b0]=yuv(Y0); out[0]=b0;out[1]=g0;out[2]=r0;out[3]=255;
                auto[r1,g1,b1]=yuv(Y1); out[4]=b1;out[5]=g1;out[6]=r1;out[7]=255;
                out+=8;
            }
        }
        rgbaPtr=m_convBuf.data(); rgbaStride=m_inW*4;
    }

    // ── Motion estimation for depth compensation ─────────────────────────────
    // Downsample luma to ~128 pixels wide, run Lucas-Kanade against previous
    // frame, and accumulate the translation.  When new depth arrives the
    // accumulator is reset to (0,0) — the depth is in sync with that frame.
    constexpr int kLumaW = 128;
    int lumaH = (m_inH * kLumaW + m_inW - 1) / m_inW;
    if (lumaH < 2) lumaH = 2;
    size_t lumaN = (size_t)kLumaW * lumaH;

    // Build current luma (box-downsampled from BGRA rgbaPtr)
    std::vector<uint8_t> curLuma(lumaN);
    {
        float scaleX = (float)m_inW  / kLumaW;
        float scaleY = (float)m_inH  / lumaH;
        for (int sy = 0; sy < lumaH; ++sy) {
            int srcY = std::min((int)(sy * scaleY), m_inH - 1);
            for (int sx = 0; sx < kLumaW; ++sx) {
                int srcX = std::min((int)(sx * scaleX), m_inW - 1);
                const BYTE* p = rgbaPtr + srcY * rgbaStride + srcX * 4;
                // BT.601 luma from BGRA: Y = 0.114*B + 0.587*G + 0.299*R
                curLuma[sy * kLumaW + sx] = (uint8_t)(
                    (p[0] * 29 + p[1] * 150 + p[2] * 77) >> 8);
            }
        }
    }

    if (m_prevLumaSmall.size() == lumaN && m_lumaSmW == kLumaW && m_lumaSmH == lumaH
        && m_hadRealDepth) {
        float dx = 0.f, dy = 0.f;
        float scale = (float)m_inW / kLumaW;
        EstimateMotionLK(m_prevLumaSmall.data(), curLuma.data(),
                         kLumaW, lumaH, scale, dx, dy);
        m_accumDx += dx;
        m_accumDy += dy;
    }
    m_prevLumaSmall = curLuma;
    m_lumaSmW = kLumaW; m_lumaSmH = lumaH;
    // When inference is slower than the video frame rate, skip posting most frames
    // so the worker always sees the LATEST frame rather than one that will never
    // be displayed.  m_skipEvery is recomputed each time a new depth result arrives.
    // ── Write source BGRA into the ring buffer ───────────────────────────────
    // Slot index cycles 0..kRingSize-1. Transform() is always called on the
    // graph thread so m_ringWr is only touched here — no mutex needed.
    int thisSlot = m_ringWr;
    m_ringWr = (m_ringWr + 1) % kRingSize;
    {
        auto& slot = m_ring[thisSlot];
        const size_t bgraBytes = (size_t)m_inH * rgbaStride;
        if (slot.bgra.size() != bgraBytes) slot.bgra.resize(bgraBytes);
        memcpy(slot.bgra.data(), rgbaPtr, bgraBytes);
        slot.stride  = rgbaStride;
        slot.frameNo = m_frameCount;
    }

    // ── Post to depth worker ─────────────────────────────────────────────────
    if (m_depth->IsLoaded()) {
        ++m_skipCounter;
        bool shouldPost = (m_skipCounter >= m_skipEvery);
        if (shouldPost) {
            m_skipCounter = 0;
            const size_t bgraBytes = (size_t)m_inH * rgbaStride;
            if (m_pendBGRA.size() != bgraBytes) m_pendBGRA.resize(bgraBytes);
            memcpy(m_pendBGRA.data(), rgbaPtr, bgraBytes);
            {
                std::lock_guard<std::mutex> lk(m_pendMtx);
                m_pendW     = m_inW;
                m_pendH     = m_inH;
                m_pendSlot  = thisSlot;
                m_pendReady = true;
            }
            m_pendCV.notify_one();
        }
    }

    // ── Consume depth result when available, render matched source BGRA ──────
    // When depth for slot S arrives, render m_ring[S].bgra with that depth.
    // Output timestamps still come from pIn — audio stays locked.
    // Fall back to current frame if the ring slot has been overwritten (extreme lag).
    DeflattenConfig cfg;
    { CAutoLock lk(&m_csConfig); cfg = m_cfg; }

    bool haveDepth = false;
    const BYTE* renderBGRA   = rgbaPtr;
    int         renderStride = rgbaStride;
    {
        std::lock_guard<std::mutex> lk(m_cacheMtx);
        if (m_cacheReady && m_cachedW == m_inW && m_cachedH == m_inH) {
            m_depthRender.swap(m_cachedDepth);
            m_cacheReady = false;
            haveDepth    = true;
            m_hadRealDepth = true;

            // Use the source frame the depth was computed from, BUT only if it
            // is at most 1 frame old.  If inference lagged (skipEvery > 1) and
            // the matched slot is already several frames in the past, rendering
            // it would show an old frame out of sequence — visible stutter.
            // In that case fall back to the current live BGRA (original behaviour:
            // no stutter, just a slight depth-vs-RGB offset during fast motion).
            int s = m_cachedSlot;
            if (s >= 0 && s < kRingSize && !m_ring[s].bgra.empty()
                && m_ring[s].stride > 0
                && (m_frameCount - m_ring[s].frameNo) <= 1) {
                renderBGRA   = m_ring[s].bgra.data();
                renderStride = m_ring[s].stride;
            }

            // Depth is matched to its source frame — motion compensation offset
            // should be zero for this matched pair.  Reset accumulator.
            m_accumDx = 0.f;
            m_accumDy = 0.f;

            // Update adaptive skip rate
            double inferMs = m_lastInferMs;
            if (inferMs > 0.0) {
                m_avgInferMs = (m_avgInferMs < 1.0) ? inferMs
                             : 0.85 * m_avgInferMs + 0.15 * inferMs;
                constexpr double kDefaultFrameMs = 41.67;
                double budget = kDefaultFrameMs;
                if (m_pInput && m_pInput->IsConnected()) {
                    CMediaType mt; m_pInput->ConnectionMediaType(&mt);
                    if (mt.formattype == FORMAT_VideoInfo &&
                        mt.FormatLength() >= sizeof(VIDEOINFOHEADER)) {
                        auto* vih = reinterpret_cast<const VIDEOINFOHEADER*>(mt.Format());
                        if (vih->AvgTimePerFrame > 0)
                            budget = (double)vih->AvgTimePerFrame / 10000.0;
                    }
                }
                int newSkip = (budget > 0)
                    ? std::max(1, std::min(4, (int)std::ceil(m_avgInferMs / budget)))
                    : 1;
                if (newSkip != m_skipEvery) {
                    LOG_INFO("Frame skip rate updated: infer every ",
                             newSkip, " frames  (avg=",
                             (int)m_avgInferMs, "ms budget=", (int)budget, "ms)");
                    m_skipEvery = newSkip;
                }
            }
        }
    }
    if (!haveDepth && !m_hadRealDepth) {
        if (logInfo) LOG_DBG("Frame #",m_frameCount,": no depth yet – using flat");
        size_t depthN = (size_t)m_inW * m_inH;
        if (m_depthRender.size() != depthN)
            m_depthRender.assign(depthN, 0.5f);
    }

    // ── Stereo render directly into the DirectShow output sample ─────────────
    int outW, outH;
    OutputDimensions(m_inW, m_inH, outW, outH);
    int outStride = outW * 4;
    LONG needed = outStride * outH;

    if (pOut->GetSize() < needed) {
        LOG_ERR("Frame #",m_frameCount,": out buf too small have=",
                pOut->GetSize()," need=",needed);
        return E_FAIL;
    }

    // Write directly into pDst — no intermediate copy
    auto tr0 = Clock::now();
    if (cfg.showDepth) {
        // Show raw depth map on both views in greyscale.
        // For SBS: left half and right half both get the same depth greyscale.
        // For TAB: top half and bottom half both get the same depth greyscale.
        const float* d = m_depthRender.data();
        int dn = m_inW * m_inH;
        if ((int)m_depthRender.size() >= dn) {
            if (cfg.outputMode == OutputMode::SideBySide) {
                // Fill both halves of the SBS output (outW = 2*inW)
                for (int y = 0; y < m_inH; ++y) {
                    BYTE* row = pDst + y * outStride;
                    for (int x = 0; x < m_inW; ++x) {
                        BYTE v = (BYTE)(std::max(0.f, std::min(1.f, d[y*m_inW+x])) * 255.f + 0.5f);
                        // Left eye
                        row[x*4+0]=v; row[x*4+1]=v; row[x*4+2]=v; row[x*4+3]=255;
                        // Right eye (offset by inW pixels)
                        row[(x+m_inW)*4+0]=v; row[(x+m_inW)*4+1]=v;
                        row[(x+m_inW)*4+2]=v; row[(x+m_inW)*4+3]=255;
                    }
                }
            } else {
                // TAB: top half and bottom half
                for (int y = 0; y < m_inH; ++y) {
                    for (int x = 0; x < m_inW; ++x) {
                        BYTE v = (BYTE)(std::max(0.f, std::min(1.f, d[y*m_inW+x])) * 255.f + 0.5f);
                        // Top eye
                        BYTE* tp = pDst + y*outStride + x*4;
                        tp[0]=v; tp[1]=v; tp[2]=v; tp[3]=255;
                        // Bottom eye
                        BYTE* bp = pDst + (y+m_inH)*outStride + x*4;
                        bp[0]=v; bp[1]=v; bp[2]=v; bp[3]=255;
                    }
                }
            }
        }
    } else {
        m_stereo->Render(renderBGRA, m_inW, m_inH, renderStride,
                         m_depthRender.data(), cfg,
                         m_accumDx, m_accumDy,
                         pDst, outStride);
    }
    auto trMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - tr0).count();

    pOut->SetActualDataLength(needed);

    REFERENCE_TIME tStart, tStop;
    if (SUCCEEDED(pIn->GetTime(&tStart, &tStop)))
        pOut->SetTime(&tStart, &tStop);
    pOut->SetSyncPoint(TRUE);
    pOut->SetDiscontinuity(pIn->IsDiscontinuity() == S_OK);

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now() - t0).count();
    // Log EVERY frame at debug level so slowdowns are visible in the log.
    // Info-level log still only on first 2 + every 100th frame.
    LOG_DBG("Frame #",m_frameCount,
             ": render=",trMs,"ms total=",ms,"ms",
             " depth=",haveDepth?"cached":"flat");
    if (logInfo)
        LOG_INFO("Frame #",m_frameCount,
                 ": render=",trMs,"ms total=",ms,"ms",
                 " depth=",haveDepth?"cached":"flat",
                 " gpu=",m_stereo->IsGPUAvailable()?"yes":"no");
    return S_OK;
}

// ── I3Deflatten ───────────────────────────────────────────────────────────────
STDMETHODIMP C3DeflattenFilter::GetConfig(DeflattenConfig* p) {
    if(!p)return E_POINTER;
    CAutoLock lk(&m_csConfig); *p=m_cfg; return S_OK;
}
STDMETHODIMP C3DeflattenFilter::SetConfig(const DeflattenConfig* p) {
    if(!p)return E_POINTER;
    { CAutoLock lk(&m_csConfig); m_cfg=*p; }
    LOG_INFO("SetConfig: conv=",m_cfg.convergence," sep=",m_cfg.separation,
             " mode=",(int)m_cfg.outputMode," gpu=",(int)m_cfg.gpuProvider,
             " smooth=",m_cfg.depthSmooth," flip=",m_cfg.flipDepth?"y":"n");
    SaveIni();
    return S_OK;
}
STDMETHODIMP C3DeflattenFilter::GetModelPath(LPWSTR buf, UINT cch) {
    if(!buf)return E_POINTER;
    wcsncpy_s(buf,cch,m_modelPath.c_str(),_TRUNCATE); return S_OK;
}
STDMETHODIMP C3DeflattenFilter::SetModelPath(LPCWSTR path) {
    if(!path)return E_POINTER;
    m_modelPath=path;
    LOG_INFO("SetModelPath: '",m_modelPath,"'");
    SaveIni();
    return S_OK;
}
STDMETHODIMP C3DeflattenFilter::GetGPUInfo(LPWSTR buf, UINT cch) {
    if(!buf)return E_POINTER;
    wcsncpy_s(buf,cch,m_gpuInfo.c_str(),_TRUNCATE); return S_OK;
}
STDMETHODIMP C3DeflattenFilter::ReloadModel() {
    LOG_INFO("ReloadModel path='",m_modelPath,"'");
    HRESULT hr = m_depth->Load(m_modelPath, m_cfg.gpuProvider, m_cfg.inferenceRuntime, m_cfg.depthMaxDim, m_gpuInfo);
    LOG_INFO("ReloadModel ",SUCCEEDED(hr)?"OK":"FAILED"," ",HRStr(hr));
    return hr;
}
STDMETHODIMP C3DeflattenFilter::GetPages(CAUUID* pPages) {
    if(!pPages)return E_POINTER;
    pPages->cElems=1;
    pPages->pElems=static_cast<GUID*>(CoTaskMemAlloc(sizeof(GUID)));
    if(!pPages->pElems)return E_OUTOFMEMORY;
    pPages->pElems[0]=CLSID_3DeflattenProp;
    return S_OK;
}
