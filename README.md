# 3Deflatten

![Build](https://github.com/ThreeDeeJay/3Deflatten/actions/workflows/build.yml/badge.svg)

**Real-time AI 2D-to-3D DirectShow video filter for Windows (32-bit & 64-bit)**

<img width="3840" height="1080" alt="z" src="https://github.com/user-attachments/assets/ff186187-5dde-49db-bac8-cc099ff48925" />
<img width="3840" height="1080" alt="Image" src="https://github.com/user-attachments/assets/e5618555-f873-4fa3-b0b3-a4e24dbac04e" />

3Deflatten is a DirectShow `.ax` filter that intercepts any video stream, estimates per-frame depth using a GPU-accelerated ONNX AI model (Depth Anything V2), and outputs a full-resolution **Side-by-Side (SBS)** or **Top-and-Bottom (TAB)** stereoscopic frame suitable for 3D displays, HMDs, and media players.

Licensed under the **GNU General Public License v3**.

---

## Features

| Feature | Detail |
|---|---|
| AI depth model | Depth Anything V2 Small / ViT-S (ONNX) |
| GPU inference | ONNX Runtime: CUDA → DirectML → CPU auto-fallback |
| GPU compositing | DirectX 11 pixel shader stereo warp |
| Output modes | Full-resolution SBS (width×2) or TAB (height×2) |
| Customisation | Convergence, separation, depth-flip, temporal smoothing |
| Configuration | DirectShow property page + `I3Deflatten` COM interface |
| Verbose logging | Activated by `DEFLATTEN_LOG_FILE` environment variable |
| Architectures | x86 (Win32) + x64, independently registered |

---

## Quick Start

### 1. Prerequisites

- **Visual Studio 2022** (MSVC 17+, C++17)
- **CMake 3.20+**
- **Git** — required by CMake's `FetchContent` to auto-download the DirectShow base classes
- **vcpkg** with `VCPKG_ROOT` environment variable set
- **Windows 10 SDK 10.0.19041+** (provides `fxc.exe` for shader compilation)
- **Python 3.8+** (to download the AI model)

> **Note on strmbase:** The DirectShow base classes are fetched from Microsoft's [Windows-classic-samples](https://github.com/microsoft/Windows-classic-samples) repo and compiled automatically by CMake — no manual SDK samples step required.

### 2. Install vcpkg dependencies

```cmd
vcpkg install onnxruntime:x64-windows directxtk:x64-windows
vcpkg install onnxruntime:x86-windows directxtk:x86-windows
```

> **CUDA (optional):** If you have the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) installed, you can get faster inference by instead running:
> ```cmd
> vcpkg install onnxruntime[cuda]:x64-windows
> ```
> Without CUDA, the filter automatically falls back to DirectML (any DX12 GPU) then CPU — no code changes needed.

### 3. Download the depth model

```cmd
pip install huggingface_hub
python scripts\download_model.py
```

The model is saved to `%APPDATA%\3Deflatten\models\depth_anything_v2_small.onnx` and found automatically.

### 4. Build

```cmd
rem 64-bit (recommended for most media players)
scripts\build_x64.bat

rem 32-bit (for legacy 32-bit players / shells)
scripts\build_x86.bat
```

### 5. Register (run as Administrator)

```cmd
scripts\register.bat
```

### 6. Use in a DirectShow graph

In **GraphEdit**, **MPC-HC**, **Zoom Player**, or any DirectShow-capable player:

1. Add your video source filter.
2. Insert **3Deflatten (2D to 3D AI Depth)**.
3. Right-click the filter → **Properties** to adjust convergence/separation.
4. Connect the output to your video renderer or 3D display sink.

---

## Environment Variables

| Variable | Description |
|---|---|
| `DEFLATTEN_LOG_FILE` | Full path for verbose log file (e.g. `C:\logs\deflatten.log`). Logging is completely disabled if this variable is not set. |
| `DEFLATTEN_MODEL_PATH` | Override the ONNX model path. If not set, the filter searches `%APPDATA%\3Deflatten\models\` and the EXE directory. |

---

## Configuration Parameters

| Parameter | Range | Default | Description |
|---|---|---|---|
| `convergence` | 0.0 – 1.0 | 0.5 | Depth value rendered at the screen plane. Objects at this normalised depth appear neutral / flat. |
| `separation` | 0.0 – 0.1 | 0.03 | Stereo strength (disparity scale). Higher values produce a stronger 3D effect. |
| `outputMode` | SBS / TAB | SBS | Side-by-Side doubles output width; Top-and-Bottom doubles output height. |
| `gpuProvider` | Auto / CUDA / DirectML / CPU | Auto | ONNX Runtime execution provider. Auto tries CUDA first, then DirectML, then CPU. |
| `depthSmooth` | 0.0 – 1.0 | 0.4 | Temporal EMA smoothing alpha for the depth map (reduces flickering). |
| `flipDepth` | bool | false | Invert near/far depth polarity (useful for some content). |

---

## COM Interface

Control the filter programmatically from any COM-capable language:

```cpp
#include "src/ideflatten.h"
#include "src/guids.h"

// Obtain the interface from the filter
CComPtr<I3Deflatten> pCfg;
pFilter->QueryInterface(IID_I3Deflatten, (void**)&pCfg);

// Read current settings
DeflattenConfig cfg;
pCfg->GetConfig(&cfg);

// Adjust and apply
cfg.separation  = 0.05f;
cfg.convergence = 0.4f;
cfg.outputMode  = OutputMode::TopAndBottom;
pCfg->SetConfig(&cfg);

// Reload model (e.g. after SetModelPath)
pCfg->SetModelPath(L"C:\\models\\depth_anything_v2_large.onnx");
pCfg->ReloadModel();

// Query active GPU info
wchar_t info[256];
pCfg->GetGPUInfo(info, 256);
```

---

## Architecture Overview

```
IMediaSample (input)
       |
       v
  [YUV -> BGRA conversion]  (CPU, inline, per-frame)
       |
       v
  DepthEstimator
  +------------------------------------------+
  | PreprocessFrame: bilinear resize + CHW   |
  | normalise to ImageNet mean/std           |
  |                                          |
  | ONNX Runtime Session                     |
  |   Execution provider: CUDA / DML / CPU  |
  |                                          |
  | PostprocessDepth: resize back,           |
  | min-max normalise to [0,1]               |
  |                                          |
  | TemporalSmooth: EMA blend with prev      |
  +------------------------------------------+
       | float depth map [0,1] at src resolution
       v
  StereoRenderer
  +------------------------------------------+
  | [GPU path]                               |
  |   Upload src + depth to DX11 textures   |
  |   Draw full-screen quad                  |
  |   PS_StereoWarp.hlsl:                   |
  |     determine eye (SBS left/right half) |
  |     d = separation*(depth-convergence)  |
  |     srcUV = eyeUV +/- (d, 0)           |
  |     output = sample(srcTex, srcUV)      |
  |   Readback via staging texture          |
  |                                          |
  | [CPU fallback]                           |
  |   Per-pixel horizontal disparity warp   |
  |   with bilinear interpolation           |
  +------------------------------------------+
       | SBS or TAB BGRA frame
       v
  IMediaSample (output)
```

---

## Project Structure

```
3Deflatten/
├── CMakeLists.txt          CMake build definition
├── vcpkg.json              vcpkg dependency manifest
├── LICENSE                 GNU GPL v3
├── README.md               This file
├── src/
│   ├── guids.h             CLSID / IID definitions
│   ├── ideflatten.h        Public COM interface (I3Deflatten)
│   ├── logger.h/cpp        Verbose file logger
│   ├── depth_estimator.h/cpp  ONNX depth inference
│   ├── stereo_renderer.h/cpp  DX11 + CPU stereo compositor
│   ├── filter.h/cpp        CTransformFilter implementation
│   ├── prop_page.h/cpp     Win32 property page
│   ├── dllmain.cpp         DLL entry + factory table
│   ├── resource.h          Dialog / string IDs
│   ├── 3Deflatten.rc       Win32 resource script
│   └── 3Deflatten.def      DLL export definitions
├── shaders/
│   └── stereo_warp.hlsl    HLSL stereo warp (VS + PS)
└── scripts/
    ├── build_x64.bat       Configure + build 64-bit
    ├── build_x86.bat       Configure + build 32-bit
    ├── register.bat        Register both .ax files
    └── download_model.py   Download Depth Anything V2 ONNX
```

---

## License

3Deflatten is free software: you can redistribute it and/or modify it under the terms of the **GNU General Public License v3** as published by the Free Software Foundation. See [LICENSE](LICENSE) for the full text.

**Third-party components:**
- **Depth Anything V2** model weights — Apache 2.0 license. See <https://github.com/DepthAnything/Depth-Anything-V2>
- **ONNX Runtime** — MIT License. See <https://github.com/microsoft/onnxruntime>
- **DirectX** — Microsoft proprietary (redistributable)
