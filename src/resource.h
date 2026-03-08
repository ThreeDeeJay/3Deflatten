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
