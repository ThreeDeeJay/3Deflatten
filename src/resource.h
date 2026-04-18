// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

// ── Dialog / string resource IDs ─────────────────────────────────────────────
#define IDD_PROP_PAGE      101
#define IDS_PROP_TITLE     102

// ── Control IDs ──────────────────────────────────────────────────────────────
#define IDC_CONV_SLIDER    1001
#define IDC_CONV_LABEL     1002
#define IDC_SEP_SLIDER     1003
#define IDC_SEP_LABEL      1004
#define IDC_SMOOTH_SLIDER  1005
#define IDC_SMOOTH_LABEL   1006
#define IDC_FLIP_CHECK     1007
#define IDC_MODE_COMBO     1008
#define IDC_GPU_COMBO      1009
#define IDC_MODEL_COMBO    1010   // lists all .onnx files in the DLL directory
#define IDC_RELOAD_BTN     1011
#define IDC_GPU_INFO       1012
#define IDC_INFILL_COMBO   1013   // occlusion gap infill mode selector
#define IDC_APPLY_BTN      1014   // Apply all settings + reload immediately
#define IDC_STREAM_CHECK   1015   // DA3-Streaming temporal alignment checkbox
#define IDC_DEPTH_CHECK    1016   // Show depth map overlay checkbox
#define IDC_RUNTIME_COMBO  1017   // Inference runtime: ONNXRuntime / TensorRT RTX
#define IDC_PROVIDER_LABEL 1018   // "Provider:" label (hidden when TRT-RTX selected)
#define IDC_MESHDIV_COMBO  1019   // Mesh resolution divisor: 1/2/4
#define IDC_DEPTHDIM_COMBO 1020   // Max depth tensor dim: Auto/518/720/1022

// ── Trackbar style constants (commctrl.h equivalents for rc.exe) ──────────────
// rc.exe does not automatically include commctrl.h, so we define what we need.
#ifndef TBS_AUTOTICKS
#define TBS_AUTOTICKS  0x0001
#endif
#ifndef TBS_HORZ
#define TBS_HORZ       0x0000
#endif
#ifndef TBS_NOTICKS
#define TBS_NOTICKS    0x0010
#endif
