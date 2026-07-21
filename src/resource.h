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
#define IDC_DILATE_SLIDER  1021   // Edge dilation radius (0-16 px)
#define IDC_DILATE_LABEL   1022   // "4 px" label
#define IDC_EDGETHRESH_SLIDER 1023 // Edge contrast threshold
#define IDC_EDGETHRESH_LABEL  1024 // "0.20" label
#define IDC_UPSCALE_COMBO  1025   // Depth upscale algorithm: Bilinear / JBU / Weighted Mode
#define IDC_DISCTHRESH_SLIDER 1026 // Mesh edge-cut threshold (gap creation)
#define IDC_DISCTHRESH_LABEL  1027 // "0.10" label
#define IDC_WMFINSET_SLIDER   1028
#define IDC_WMFINSET_LABEL    1029
#define IDC_INSP_TX_SLIDER    1030
#define IDC_INSP_TX_LABEL     1031
#define IDC_INSP_TY_SLIDER    1032
#define IDC_INSP_TY_LABEL     1033
#define IDC_INSP_TZ_SLIDER    1034
#define IDC_INSP_TZ_LABEL     1035
#define IDC_INSP_RY_SLIDER    1036
#define IDC_INSP_RY_LABEL     1037
#define IDC_INSP_RX_SLIDER    1038
#define IDC_INSP_RX_LABEL     1039
#define IDC_INSP_RZ_SLIDER    1040
#define IDC_INSP_RZ_LABEL     1041

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
