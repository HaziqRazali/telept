"""
sync_and_calibrate.py

Interactive tool for loading and synchronising:
  - optitrack.avi  (optitrack MJPEG video)
  - optitrack.c3d  (C3D motion-capture data)
  - orbbec.mp4     (Orbbec camera video)

Workflow
--------
1. Click the three Load buttons (bottom row) to open each file.
2. Once a video is loaded its frame appears in the panel.
   - Scroll wheel to zoom in / out on the panel.
   - Click + drag to draw a bounding box around the LED / blink region.
   - Release → blink signal is computed automatically from the box.
   - Drag again on the same panel to redraw the box at any time.
3. Adjust the threshold sliders to clean the ON/OFF signal.
4. Use the frame sliders, sync slider, and Play / Step buttons to align
   the two video streams.

Controls
--------
  Scroll on video panel  : zoom in / out (centred on cursor)
  Left-drag on panel     : draw / redraw bounding box
  ← / →                  : step −1 / +1 (both streams)
  q                      : quit
"""

import os
import time

import numpy as np
import cv2

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.proj3d import proj_transform

import tkinter as tk
from tkinter import filedialog

try:
    import ezc3d
    HAS_EZC3D = True
except ImportError:
    HAS_EZC3D = False
    print("WARNING: ezc3d not found – C3D loading will be unavailable")

# ── constants ──────────────────────────────────────────────────────────────────
FULL_SCALE   = 4        # load full frames at 1/FULL_SCALE resolution
INSET_MARGIN = 8        # px gap from panel edge for the zoomed inset
INIT_THRESH  = 75
ZOOM_FACTOR  = 0.80     # per scroll step (< 1 = zoom in)

COL_OPTI = "#ffaa33"
COL_ORB  = "#33cc66"
COL_C3D  = "#4499ff"
COL_BBOX = (255, 80, 80)   # BGR for cv2 drawing
COL_BBOX_MPL = "#ff5050"   # for matplotlib patch


# ── file/data helpers ──────────────────────────────────────────────────────────

def _ask_file(title_str, filetypes):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title_str, filetypes=filetypes)
    root.destroy()
    return path or None


def _load_video_frames(path, scale=FULL_SCALE):
    """Load all frames of a video at 1/scale resolution.

    Returns
    -------
    frames   : list of (H, W, 3) RGB uint8 arrays
    fps      : float
    n        : int  total frame count
    disp_w   : int  display width
    disp_h   : int  display height
    orig_w   : int  original width
    orig_h   : int  original height
    """
    cap    = cv2.VideoCapture(path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dw     = max(1, orig_w // scale)
    dh     = max(1, orig_h // scale)

    frames = []
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            blank = np.zeros((dh, dw, 3), np.uint8)
            frames.extend([blank] * (n - i))
            break
        small = cv2.resize(f, (dw, dh), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        if i % 60 == 0:
            print(f"  frame {i}/{n}", end="\r", flush=True)
    cap.release()
    return frames, fps, n, dw, dh, orig_w, orig_h


def _compute_bbox_medians(frames, bbox):
    """Per-frame median intensity of the bbox region (display pixel coords).

    bbox : (x1, y1, x2, y2)  integers in display-pixel space
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    h, w    = frames[0].shape[:2]
    x1, y1  = max(0, x1), max(0, y1)
    x2, y2  = min(w, x2), min(h, y2)

    meds = np.empty(len(frames), dtype=np.float32)
    for i, f in enumerate(frames):
        crop = f[y1:y2, x1:x2]
        if crop.size == 0:
            meds[i] = 0.0
        else:
            gray     = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            meds[i]  = float(np.median(gray))
    return meds


def _load_c3d(path):
    """Load a C3D file.

    Returns
    -------
    led    : (N,) float32  marker-visible signal (1 = any marker visible)
    t      : (N,) float64  time axis
    fps    : float
    n      : int
    xyz    : (3, M, N) float64
    bounds : (3, 2) stable axis limits for the 3-D scatter
    """
    c   = ezc3d.c3d(path)
    fps = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    xyz = c["data"]["points"][:3]          # (3, M, N)
    n   = xyz.shape[2]
    n_vis = np.all(np.isfinite(xyz), axis=0).sum(axis=0)   # (N,)
    led = (n_vis > 0).astype(np.float32)
    t   = np.arange(n) / fps

    all_pts = xyz.reshape(3, -1)
    mask    = np.all(np.isfinite(all_pts), axis=0)
    if mask.any():
        fp  = all_pts[:, mask]
        lo  = np.percentile(fp, 1,  axis=1)
        hi  = np.percentile(fp, 99, axis=1)
        pad = np.maximum((hi - lo) * 0.20, 0.05)
        bounds = np.stack([lo - pad, hi + pad], axis=1)
    else:
        bounds = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=float)

    return led, t, fps, n, xyz, bounds


# ── compose helpers ────────────────────────────────────────────────────────────

def _compose_frame(frames, bbox, frame_idx):
    """Return an RGB image: full frame with zoomed inset of bbox (no rect on full frame)."""
    img = frames[min(frame_idx, len(frames) - 1)].copy()
    h, w = img.shape[:2]

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        x1c = max(0, min(x1, x2));  x2c = min(w, max(x1, x2))
        y1c = max(0, min(y1, y2));  y2c = min(h, max(y1, y2))

        # crop from original (no drawing on full frame)
        crop = img[y1c:y2c, x1c:x2c]
        if crop.size > 0:
            iw = max(1, min(w // 3, w - 2 * INSET_MARGIN))
            ih = max(1, min(h // 3, h - 2 * INSET_MARGIN))
            ins = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_NEAREST)
            px  = w - iw - INSET_MARGIN
            py  = h - ih - INSET_MARGIN
            img[py:py + ih, px:px + iw] = ins
            # thin red border around the inset only
            cv2.rectangle(img, (px, py), (px + iw, py + ih), COL_BBOX, 1)

    return img


# ── shared mutable state ───────────────────────────────────────────────────────

D = {
    # optitrack video
    "opti_frames": None, "opti_fps": 30.0, "opti_n": 1,
    "opti_w": 1, "opti_h": 1,
    "opti_bbox": None,   # (x1, y1, x2, y2) in display pixels, ints
    "opti_meds": None,

    # orbbec video
    "orb_frames": None, "orb_fps": 30.0, "orb_n": 1,
    "orb_w": 1, "orb_h": 1,
    "orb_bbox": None,
    "orb_meds": None,

    # C3D
    "c3d_led": None, "c3d_t": None, "c3d_fps": 1.0, "c3d_n": 1,
    "c3d_xyz": None, "c3d_bounds": None,
    "c3d_led_roi": None,          # per-frame LED signal filtered by ROI marker selection
    "c3d_sel_markers": None,      # array of marker indices selected by ROI

    # file paths / folders
    "opti_dir": None,   # folder of optitrack.avi
    "orb_dir":  None,   # folder of orbbec.mp4

    # playback
    "playing": False,
    "_last_tick": 0.0,
    "_opti_frac": 0.0,
    "_orb_frac": 0.0,
    "_sync_updating": False,
    "_sync_offsets": [0, 0],

    # bbox rubber-band (video panels – data coords)
    "opti_drawing": False, "opti_draw_start": None,
    "orb_drawing":  False, "orb_draw_start":  None,

    # C3D ROI rubber-band (display pixel coords)
    "c3d_drawing": False, "c3d_draw_start_disp": None,
}

# imshow handles (set on first load)
_im_opti = None
_im_orb  = None
_sc_c3d  = None


# ── figure layout ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 11), facecolor="#111111")
fig.canvas.manager.set_window_title("Sync & Calibrate")

# ── signal plot ──────────────────────────────────────────────────────────────
ax_sig = fig.add_axes([0.05, 0.605, 0.92, 0.36])
ax_sig.set_facecolor("#1a1a1a")
ax_sig.set_xlabel("time (s)", color="#aaaaaa", fontsize=8)
ax_sig.set_ylabel("LED on / off", color="#aaaaaa", fontsize=8)
ax_sig.set_title(
    "Blink signals — C3D (blue) | Optitrack bbox (orange) | Orbbec bbox (green)",
    color="white", fontsize=10)
ax_sig.tick_params(colors="#888888", labelsize=7)
for sp in ax_sig.spines.values():
    sp.set_edgecolor("#444444")
ax_sig.set_ylim(-0.2, 1.45)
_hint = ax_sig.text(0.5, 0.5,
                    "Load files and draw bounding boxes to see blink signals",
                    transform=ax_sig.transAxes, ha="center", va="center",
                    color="#555555", fontsize=12)

_dummy = np.array([0.0, 1.0])
line_c3d,  = ax_sig.step(_dummy, _dummy * 0, where="post",
                          color=COL_C3D,  lw=1.0, label="C3D",      visible=False)
line_opti, = ax_sig.step(_dummy, _dummy * 0, where="post",
                          color=COL_OPTI, lw=1.0, label="Optitrack", visible=False)
line_orb,  = ax_sig.step(_dummy, _dummy * 0, where="post",
                          color=COL_ORB,  lw=1.0, label="Orbbec",    visible=False)
vline_opti = ax_sig.axvline(0, color=COL_OPTI, lw=1.2, ls="--", alpha=0.7, visible=False)
vline_orb  = ax_sig.axvline(0, color=COL_ORB,  lw=1.2, ls="-.", alpha=0.7, visible=False)
vline_c3d  = ax_sig.axvline(0, color=COL_C3D,  lw=1.2, ls=":",  alpha=0.7, visible=False)

# ── threshold sliders ──────────────────────────────────────────────────────────
ax_thr_opti = fig.add_axes([0.10, 0.548, 0.38, 0.026], facecolor="#2a2a2a")
thresh_opti_sl = mwidgets.Slider(ax_thr_opti, "Opti thr", 0, 255,
                                  valinit=INIT_THRESH, valstep=1, color="#dd8833")
thresh_opti_sl.label.set_color("white")
thresh_opti_sl.valtext.set_color("white")

ax_thr_orb = fig.add_axes([0.55, 0.548, 0.38, 0.026], facecolor="#2a2a2a")
thresh_orb_sl = mwidgets.Slider(ax_thr_orb, "Orbbec thr", 0, 255,
                                 valinit=INIT_THRESH, valstep=1, color="#226633")
thresh_orb_sl.label.set_color("white")
thresh_orb_sl.valtext.set_color("white")

# ── info bar ──────────────────────────────────────────────────────────────────
ax_info = fig.add_axes([0.05, 0.510, 0.92, 0.030])
ax_info.axis("off")
info_text = ax_info.text(0.5, 0.5,
                          "Load optitrack_mjpeg, optitrack_c3d, and orbbec to begin",
                          ha="center", va="center", color="#666666", fontsize=9,
                          transform=ax_info.transAxes, fontfamily="monospace")

# ── video panels ──────────────────────────────────────────────────────────────
ax_opti = fig.add_axes([0.02, 0.17, 0.27, 0.31])
ax_opti.set_facecolor("#0d0d0d")
ax_opti.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for sp in ax_opti.spines.values():
    sp.set_edgecolor("#333333")
_ph_opti = ax_opti.text(0.5, 0.5,
                         "Click 'Load Optitrack mjpeg (.avi)",
                         transform=ax_opti.transAxes, ha="center", va="center",
                         color="#555555", fontsize=10)
ax_opti.set_title("optitrack.avi", color="white", fontsize=9, pad=3)

ax_c3d = fig.add_axes([0.32, 0.17, 0.27, 0.31], projection="3d")
ax_c3d.set_facecolor("#0a0a0a")
for _pane in (ax_c3d.xaxis.pane, ax_c3d.yaxis.pane, ax_c3d.zaxis.pane):
    _pane.fill = False
    _pane.set_edgecolor("#333333")
ax_c3d.grid(True, color="#222222", linewidth=0.4)
ax_c3d.tick_params(colors="#555555", labelsize=5)
ax_c3d.set_xlabel("X", color="#666666", fontsize=7, labelpad=1)
ax_c3d.set_ylabel("Y", color="#666666", fontsize=7, labelpad=1)
ax_c3d.set_zlabel("Z", color="#666666", fontsize=7, labelpad=1)
title_c3d = ax_c3d.set_title("optitrack.c3d", color="#666666", fontsize=9, pad=3)

ax_orb = fig.add_axes([0.63, 0.17, 0.35, 0.31])
ax_orb.set_facecolor("#0d0d0d")
ax_orb.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for sp in ax_orb.spines.values():
    sp.set_edgecolor("#333333")
_ph_orb = ax_orb.text(0.5, 0.5,
                       "Click 'Load RGB.mp4",
                       transform=ax_orb.transAxes, ha="center", va="center",
                       color="#555555", fontsize=10)
ax_orb.set_title("rgb.mp4", color="white", fontsize=9, pad=3)

# ── rubber-band rectangle patches over each video panel ───────────────────────
_opti_rect = mpatches.Rectangle((0, 0), 0, 0, linewidth=1.5,
                                  edgecolor=COL_BBOX_MPL, facecolor="none",
                                  linestyle="--", visible=False)
_orb_rect  = mpatches.Rectangle((0, 0), 0, 0, linewidth=1.5,
                                  edgecolor=COL_BBOX_MPL, facecolor="none",
                                  linestyle="--", visible=False)
ax_opti.add_patch(_opti_rect)
ax_orb.add_patch(_orb_rect)

# C3D ROI rubber-band lives in figure display-pixel space
_c3d_rect = mpatches.Rectangle((0, 0), 0, 0, linewidth=1.5,
                                 edgecolor=COL_BBOX_MPL, facecolor="none",
                                 linestyle="--", visible=False,
                                 transform=fig.dpi_scale_trans)
fig.patches.append(_c3d_rect)
_c3d_roi_active = [False]

# ── frame sliders (initially greyed out) ──────────────────────────────────────
ax_opti_sl = fig.add_axes([0.02, 0.120, 0.57, 0.026], facecolor="#1a1a1a")
opti_slider = mwidgets.Slider(ax_opti_sl, "Opti / C3D frame", 0, 1,
                               valinit=0, valstep=1, color="#555555")
opti_slider.label.set_color("#555555")
opti_slider.valtext.set_color("#555555")

ax_orb_sl  = fig.add_axes([0.63, 0.120, 0.35, 0.026], facecolor="#1a1a1a")
orb_slider  = mwidgets.Slider(ax_orb_sl, "Orbbec frame", 0, 1,
                               valinit=0, valstep=1, color="#555555")
orb_slider.label.set_color("#555555")
orb_slider.valtext.set_color("#555555")

ax_sync_sl = fig.add_axes([0.05, 0.084, 0.92, 0.026], facecolor="#1a1a1a")
sync_slider = mwidgets.Slider(ax_sync_sl, "Sync (s)", 0.0, 1.0,
                               valinit=0.0, color="#555555")
sync_slider.label.set_color("#555555")
sync_slider.valtext.set_color("#555555")

# ── control buttons ───────────────────────────────────────────────────────────
ax_load_opti = fig.add_axes([0.02, 0.048, 0.11, 0.030])
btn_load_opti = mwidgets.Button(ax_load_opti, "Load Optitrack",
                                 color="#1a2233", hovercolor="#2a3a55")
btn_load_opti.label.set_color("#aaaaff")
btn_load_opti.label.set_fontsize(8)

ax_load_c3d = fig.add_axes([0.15, 0.048, 0.09, 0.030])
btn_load_c3d = mwidgets.Button(ax_load_c3d, "Load C3D",
                                color="#1a1a33", hovercolor="#2a2a55")
btn_load_c3d.label.set_color("#8888ff")
btn_load_c3d.label.set_fontsize(8)

ax_load_orb = fig.add_axes([0.26, 0.048, 0.10, 0.030])
btn_load_orb = mwidgets.Button(ax_load_orb, "Load Orbbec",
                                color="#1a3322", hovercolor="#2a5533")
btn_load_orb.label.set_color("#aaffaa")
btn_load_orb.label.set_fontsize(8)

ax_play = fig.add_axes([0.40, 0.048, 0.09, 0.030])
btn_play = mwidgets.Button(ax_play, "▶  Play",
                            color="#225522", hovercolor="#338833")
btn_play.label.set_color("white")
btn_play.label.set_fontsize(9)

ax_reset = fig.add_axes([0.51, 0.048, 0.07, 0.030])
btn_reset = mwidgets.Button(ax_reset, "⟳ Reset",
                             color="#222255", hovercolor="#333388")
btn_reset.label.set_color("white")
btn_reset.label.set_fontsize(9)

ax_minus = fig.add_axes([0.60, 0.048, 0.055, 0.030])
btn_minus = mwidgets.Button(ax_minus, "◀  −1",
                             color="#333333", hovercolor="#555555")
btn_minus.label.set_color("white")
btn_minus.label.set_fontsize(9)

ax_plus = fig.add_axes([0.66, 0.048, 0.055, 0.030])
btn_plus = mwidgets.Button(ax_plus, "+1  ▶",
                            color="#333333", hovercolor="#555555")
btn_plus.label.set_color("white")
btn_plus.label.set_fontsize(9)

ax_save_off = fig.add_axes([0.74, 0.048, 0.11, 0.030])
btn_save_off = mwidgets.Button(ax_save_off, "💾 Save Offset",
                                color="#2a2200", hovercolor="#443300")
btn_save_off.label.set_color("#ffdd55")
btn_save_off.label.set_fontsize(8)

ax_load_off = fig.add_axes([0.86, 0.048, 0.11, 0.030])
btn_load_off = mwidgets.Button(ax_load_off, "📂 Load Offset",
                                color="#1a2a1a", hovercolor="#2a4a2a")
btn_load_off.label.set_color("#88ee88")
btn_load_off.label.set_fontsize(8)

# ── C3D ROI toggle button — own row between frame sliders and panels ──────────
ax_c3d_roi_btn = fig.add_axes([0.38, 0.010, 0.14, 0.030])
btn_c3d_roi = mwidgets.Button(ax_c3d_roi_btn, "☐  Set C3D ROI",
                               color="#1a1a33", hovercolor="#2a2a55")
btn_c3d_roi.label.set_color("#8888ff")
btn_c3d_roi.label.set_fontsize(8)


# ── slider activation helper ──────────────────────────────────────────────────

def _activate_slider(slider, ax, max_val, poly_color, label_color):
    ax.set_facecolor("#2a2a2a")
    slider.valmax = float(max_val)
    slider.ax.set_xlim(0, float(max_val))
    slider.set_val(0)
    slider.poly.set_facecolor(poly_color)
    slider.label.set_color(label_color)
    slider.valtext.set_color(label_color)


def _try_enable_sync():
    """Enable / update the sync slider once both videos are known."""
    if D["opti_frames"] is None or D["orb_frames"] is None:
        return
    of    = D["opti_fps"] or 1.0
    rf    = D["orb_fps"]  or 1.0
    t_max = min((D["opti_n"] - 1) / of, (D["orb_n"] - 1) / rf)
    _activate_slider(sync_slider, ax_sync_sl, max(t_max, 1.0), "#aa44aa", "white")


# ── signal-plot refresh ───────────────────────────────────────────────────────

def _refresh_info():
    parts = []
    thr_o = int(thresh_opti_sl.val)
    thr_r = int(thresh_orb_sl.val)
    if D["c3d_led"] is not None:
        led = D["c3d_led_roi"] if D["c3d_led_roi"] is not None else D["c3d_led"]
        n_sel = len(D["c3d_sel_markers"]) if D["c3d_sel_markers"] is not None else "all"
        parts.append(f"C3D ON:{led.sum()/D['c3d_fps']:.2f}s "
                     f"trans:{int(np.abs(np.diff(led)).sum())} "
                     f"markers={n_sel}")
    if D["opti_meds"] is not None:
        led = (D["opti_meds"] > thr_o).astype(np.float32)
        parts.append(f"Opti ON:{led.sum()/D['opti_fps']:.2f}s "
                     f"trans:{int(np.abs(np.diff(led)).sum())}")
    if D["orb_meds"] is not None:
        led = (D["orb_meds"] > thr_r).astype(np.float32)
        parts.append(f"Orbbec ON:{led.sum()/D['orb_fps']:.2f}s "
                     f"trans:{int(np.abs(np.diff(led)).sum())}")
    if parts:
        info_text.set_text("   |   ".join(parts))
        info_text.set_color("#cccccc")


def _refresh_signal_plot():
    thr_o = int(thresh_opti_sl.val)
    thr_r = int(thresh_orb_sl.val)
    xmax  = 1.0

    if D["c3d_led"] is not None:
        led_c3d_plot = D["c3d_led_roi"] if D["c3d_led_roi"] is not None else D["c3d_led"]
        line_c3d.set_xdata(D["c3d_t"])
        line_c3d.set_ydata(led_c3d_plot)
        n_sel = len(D["c3d_sel_markers"]) if D["c3d_sel_markers"] is not None else "all"
        line_c3d.set_label(f"C3D (markers={n_sel})")
        line_c3d.set_visible(True)
        vline_c3d.set_visible(True)
        xmax = max(xmax, D["c3d_t"][-1])

    if D["opti_meds"] is not None:
        t_op  = np.arange(len(D["opti_meds"])) / D["opti_fps"]
        led   = (D["opti_meds"] > thr_o).astype(np.float32)
        line_opti.set_xdata(t_op)
        line_opti.set_ydata(led * 1.06)
        line_opti.set_label(f"Optitrack (thr={thr_o})")
        line_opti.set_visible(True)
        vline_opti.set_visible(True)
        xmax  = max(xmax, t_op[-1])
        _hint.set_visible(False)

    if D["orb_meds"] is not None:
        t_ob  = np.arange(len(D["orb_meds"])) / D["orb_fps"]
        led   = (D["orb_meds"] > thr_r).astype(np.float32)
        line_orb.set_xdata(t_ob)
        line_orb.set_ydata(led * 1.12)
        line_orb.set_label(f"Orbbec (thr={thr_r})")
        line_orb.set_visible(True)
        vline_orb.set_visible(True)
        xmax  = max(xmax, t_ob[-1])
        _hint.set_visible(False)

    ax_sig.set_xlim(0, xmax)
    ax_sig.legend(fontsize=8, facecolor="#222222", labelcolor="white",
                  loc="upper right")
    _refresh_info()
    fig.canvas.draw_idle()


# ── per-frame draw helpers ────────────────────────────────────────────────────

def _project_c3d_medians_to_display():
    """Return (N_markers, 2) display-pixel coords of each marker's median 3D position."""
    xyz   = D["c3d_xyz"]          # (3, M, F)
    M     = xyz.shape[1]
    pts   = np.empty((M, 2))
    pts[:] = np.nan
    proj  = ax_c3d.get_proj()
    ax_pos = ax_c3d.get_position()    # figure-fraction Bbox
    fw, fh = fig.get_size_inches() * fig.dpi
    for m in range(M):
        traj = xyz[:, m, :].T       # (F, 3)
        ok   = np.all(np.isfinite(traj), axis=1)
        if not ok.any():
            continue
        med  = np.median(traj[ok], axis=0)
        x2, y2, _ = proj_transform(med[0], med[1], med[2], proj)
        # proj_transform returns values in [-1,1] normalised axes data space
        # convert to display pixels via the axes transform
        xd, yd = ax_c3d.transData.transform((x2, y2))
        pts[m] = [xd, yd]
    return pts


def _compute_c3d_led_roi(bbox_disp):
    """Compute per-frame LED signal counting only markers whose median position
    projects inside bbox_disp = (x1, y1, x2, y2) in display pixels."""
    x1d, y1d, x2d, y2d = [float(v) for v in bbox_disp]
    x1d, x2d = min(x1d, x2d), max(x1d, x2d)
    y1d, y2d = min(y1d, y2d), max(y1d, y2d)

    med_disp = _project_c3d_medians_to_display()   # (M, 2)
    inside   = (med_disp[:, 0] >= x1d) & (med_disp[:, 0] <= x2d) & \
               (med_disp[:, 1] >= y1d) & (med_disp[:, 1] <= y2d)
    sel      = np.where(inside)[0]
    print(f"  C3D ROI: {len(sel)} markers selected: {sel}")
    D["c3d_sel_markers"] = sel

    if len(sel) == 0:
        print("  WARNING: no markers in ROI – keeping full signal")
        D["c3d_led_roi"] = None
        D["c3d_sel_markers"] = None
        return

    xyz = D["c3d_xyz"]   # (3, M, F)
    n_vis_roi = np.all(np.isfinite(xyz[:, sel, :]), axis=0).sum(axis=0)  # (F,)
    D["c3d_led_roi"] = (n_vis_roi > 0).astype(np.float32)


def _redraw_c3d(f):
    global _sc_c3d
    if D["c3d_xyz"] is None:
        return
    fc  = min(f, D["c3d_n"] - 1)
    t   = fc / D["c3d_fps"]
    pts = D["c3d_xyz"][:, :, fc].T
    mk  = np.all(np.isfinite(pts), axis=1)

    # highlight ROI-selected markers in a different colour if ROI is active
    sel = D["c3d_sel_markers"]
    if sel is not None and len(sel) > 0:
        sel_mask = np.zeros(pts.shape[0], dtype=bool)
        sel_mask[sel] = True
        colors = np.where(sel_mask & mk, "#ff5555", "#444444")
    else:
        colors = np.where(mk, "#ffee00", "#444444")

    vis_pts = pts[mk]
    vis_col = colors[mk]
    if _sc_c3d is None:
        if len(vis_pts):
            _sc_c3d = ax_c3d.scatter(vis_pts[:, 0], vis_pts[:, 1], vis_pts[:, 2],
                                      c=vis_col, s=30, alpha=0.95, depthshade=False)
        else:
            _sc_c3d = ax_c3d.scatter([], [], [], c="#ffee00", s=30, depthshade=False)
    else:
        if len(vis_pts):
            _sc_c3d._offsets3d = (vis_pts[:, 0], vis_pts[:, 1], vis_pts[:, 2])
            _sc_c3d.set_facecolor(vis_col)
        else:
            _sc_c3d._offsets3d = (np.array([]), np.array([]), np.array([]))
    led = D["c3d_led_roi"] if D["c3d_led_roi"] is not None else D["c3d_led"]
    title_c3d.set_text(
        f"C3D  t={t:.3f}s  frame {fc}  "
        f"LED={'ON' if led[fc] else 'off'}")


def _redraw_opti(f):
    global _im_opti
    if D["opti_frames"] is None:
        return
    img = _compose_frame(D["opti_frames"], D["opti_bbox"], f)
    if _im_opti is None:
        _im_opti = ax_opti.imshow(img, aspect="auto", interpolation="bilinear")
    else:
        _im_opti.set_data(img)
    t       = f / D["opti_fps"]
    med_str = (f"  bbox med={D['opti_meds'][min(f, len(D['opti_meds'])-1)]:.1f}"
               if D["opti_meds"] is not None else "  (draw bbox to compute signal)")
    ax_opti.set_title(
        f"optitrack.avi  t={t:.3f}s  frame {f}{med_str}",
        color="white", fontsize=9, pad=3)
    vline_opti.set_xdata([t, t])
    if D["c3d_t"] is not None:
        vline_c3d.set_xdata([t, t])
    _redraw_c3d(f)


def _redraw_orb(f):
    global _im_orb
    if D["orb_frames"] is None:
        return
    img = _compose_frame(D["orb_frames"], D["orb_bbox"], f)
    if _im_orb is None:
        _im_orb = ax_orb.imshow(img, aspect="auto", interpolation="bilinear")
    else:
        _im_orb.set_data(img)
    t       = f / D["orb_fps"]
    med_str = (f"  bbox med={D['orb_meds'][min(f, len(D['orb_meds'])-1)]:.1f}"
               if D["orb_meds"] is not None else "  (draw bbox to compute signal)")
    ax_orb.set_title(
        f"orbbec.mp4  t={t:.3f}s  frame {f}{med_str}",
        color="white", fontsize=9, pad=3)
    vline_orb.set_xdata([t, t])


# ── load callbacks ────────────────────────────────────────────────────────────

def on_load_opti(event):
    path = _ask_file("Open optitrack MJPEG video",
                     [("Video files", "*.avi *.mp4 *.mov"), ("All", "*.*")])
    if not path:
        return
    print(f"\nLoading {os.path.basename(path)} …")
    btn_load_opti.label.set_text("Loading…")
    fig.canvas.draw_idle(); fig.canvas.flush_events()

    frames, fps, n, dw, dh, ow, oh = _load_video_frames(path)
    D.update({"opti_frames": frames, "opti_fps": fps, "opti_n": n,
               "opti_w": dw, "opti_h": dh,
               "opti_bbox": None, "opti_meds": None,
               "opti_dir": os.path.dirname(os.path.abspath(path))})
    print(f"\n  Loaded {n} frames @ {fps:.4f} fps  display {dw}×{dh}")

    _ph_opti.set_visible(False)
    _redraw_opti(0)
    ax_opti.set_xlim(-0.5, dw - 0.5)
    ax_opti.set_ylim(dh - 0.5, -0.5)

    n_shared = min(n, D["c3d_n"]) if D["c3d_n"] > 1 else n
    _activate_slider(opti_slider, ax_opti_sl, n_shared - 1, COL_OPTI, COL_OPTI)

    btn_load_opti.label.set_text("✓ Optitrack")
    btn_load_opti.ax.set_facecolor("#223322")
    _try_enable_sync()
    fig.canvas.draw_idle()


def on_load_c3d(event):
    if not HAS_EZC3D:
        info_text.set_text("ezc3d not installed – install with: pip install ezc3d")
        info_text.set_color("#ff6666")
        fig.canvas.draw_idle()
        return
    path = _ask_file("Open optitrack C3D",
                     [("C3D files", "*.c3d"), ("All", "*.*")])
    if not path:
        return
    print(f"\nLoading {os.path.basename(path)} …")
    btn_load_c3d.label.set_text("Loading…")
    fig.canvas.draw_idle(); fig.canvas.flush_events()

    led, t, fps, n, xyz, bounds = _load_c3d(path)
    D.update({"c3d_led": led, "c3d_t": t, "c3d_fps": fps, "c3d_n": n,
               "c3d_xyz": xyz, "c3d_bounds": bounds})
    print(f"\n  C3D: {n} frames @ {fps:.4f} fps  ({t[-1]:.3f} s)")

    ax_c3d.set_xlim(bounds[0]); ax_c3d.set_ylim(bounds[1]); ax_c3d.set_zlim(bounds[2])
    _redraw_c3d(0)
    _refresh_signal_plot()

    btn_load_c3d.label.set_text("✓ C3D")
    btn_load_c3d.ax.set_facecolor("#222238")
    _try_enable_sync()
    fig.canvas.draw_idle()


def on_load_orb(event):
    path = _ask_file("Open Orbbec video",
                     [("Video files", "*.mp4 *.avi *.mov"), ("All", "*.*")])
    if not path:
        return
    print(f"\nLoading {os.path.basename(path)} …")
    btn_load_orb.label.set_text("Loading…")
    fig.canvas.draw_idle(); fig.canvas.flush_events()

    frames, fps, n, dw, dh, ow, oh = _load_video_frames(path)
    D.update({"orb_frames": frames, "orb_fps": fps, "orb_n": n,
               "orb_w": dw, "orb_h": dh,
               "orb_bbox": None, "orb_meds": None,
               "orb_dir": os.path.dirname(os.path.abspath(path))})
    print(f"\n  Loaded {n} frames @ {fps:.4f} fps  display {dw}×{dh}")

    _ph_orb.set_visible(False)
    _redraw_orb(0)
    ax_orb.set_xlim(-0.5, dw - 0.5)
    ax_orb.set_ylim(dh - 0.5, -0.5)

    _activate_slider(orb_slider, ax_orb_sl, n - 1, COL_ORB, COL_ORB)

    btn_load_orb.label.set_text("✓ Orbbec")
    btn_load_orb.ax.set_facecolor("#1a3322")
    _try_enable_sync()
    fig.canvas.draw_idle()


# ── bbox drawing (scroll + drag) ──────────────────────────────────────────────

def _on_scroll(event):
    ax = event.inaxes
    if ax not in (ax_opti, ax_orb):
        return
    which = "opti" if ax is ax_opti else "orb"
    if D[f"{which}_frames"] is None:
        return

    w  = D[f"{which}_w"]
    h  = D[f"{which}_h"]
    xl = list(ax.get_xlim())
    yl = list(ax.get_ylim())            # yl[0] > yl[1] because image y is flipped
    cx = event.xdata if event.xdata is not None else (xl[0] + xl[1]) / 2
    cy = event.ydata if event.ydata is not None else (yl[0] + yl[1]) / 2

    factor = ZOOM_FACTOR if event.button == "up" else 1.0 / ZOOM_FACTOR
    xl = [cx + (x - cx) * factor for x in xl]
    yl = [cy + (y - cy) * factor for y in yl]

    # clamp to image extent
    xl[0] = max(-0.5, xl[0])
    xl[1] = min(w - 0.5, xl[1])
    yl[0] = min(h - 0.5, yl[0])
    yl[1] = max(-0.5, yl[1])

    ax.set_xlim(xl)
    ax.set_ylim(yl)
    fig.canvas.draw_idle()


def _on_press(event):
    if event.button != 1:
        return
    ax = event.inaxes

    # ── C3D ROI drag (display-pixel coords) ──
    if _c3d_roi_active[0] and D["c3d_xyz"] is not None:
        # check if click is within the C3D axes bounding box
        ax_pos = ax_c3d.get_position()
        fw, fh = fig.get_size_inches() * fig.dpi
        bx0 = ax_pos.x0 * fw;  bx1 = ax_pos.x1 * fw
        by0 = ax_pos.y0 * fh;  by1 = ax_pos.y1 * fh
        ex, ey = event.x, event.y
        if bx0 <= ex <= bx1 and by0 <= ey <= by1:
            D["c3d_drawing"] = True
            D["c3d_draw_start_disp"] = (ex, ey)
            _c3d_rect.set_xy((ex, ey))
            _c3d_rect.set_width(0)
            _c3d_rect.set_height(0)
            _c3d_rect.set_visible(True)
            fig.canvas.draw_idle()
            return

    # ── video panel drag (data coords) ──
    if ax not in (ax_opti, ax_orb):
        return
    which = "opti" if ax is ax_opti else "orb"
    if D[f"{which}_frames"] is None:
        return
    D[f"{which}_drawing"]    = True
    D[f"{which}_draw_start"] = (event.xdata, event.ydata)
    patch = _opti_rect if which == "opti" else _orb_rect
    patch.set_xy((event.xdata, event.ydata))
    patch.set_width(0)
    patch.set_height(0)
    patch.set_visible(True)
    fig.canvas.draw_idle()


def _on_motion(event):
    # C3D ROI rubber-band
    if D["c3d_drawing"]:
        sx, sy = D["c3d_draw_start_disp"]
        ex, ey = event.x, event.y
        _c3d_rect.set_xy((min(sx, ex), min(sy, ey)))
        _c3d_rect.set_width(abs(ex - sx))
        _c3d_rect.set_height(abs(ey - sy))
        fig.canvas.draw_idle()

    # video panel rubber-bands
    for which, ax, patch in (("opti", ax_opti, _opti_rect),
                              ("orb",  ax_orb,  _orb_rect)):
        if not D[f"{which}_drawing"]:
            continue
        if event.inaxes is not ax:
            continue
        sx, sy = D[f"{which}_draw_start"]
        ex = event.xdata if event.xdata is not None else sx
        ey = event.ydata if event.ydata is not None else sy
        patch.set_xy((min(sx, ex), min(sy, ey)))
        patch.set_width(abs(ex - sx))
        patch.set_height(abs(ey - sy))
        fig.canvas.draw_idle()


def _on_release(event):
    # ── C3D ROI release ──
    if D["c3d_drawing"] and event.button == 1:
        D["c3d_drawing"] = False
        sx, sy = D["c3d_draw_start_disp"]
        ex, ey = event.x, event.y
        _c3d_rect.set_visible(False)
        if abs(ex - sx) > 5 and abs(ey - sy) > 5:
            bbox_disp = (sx, sy, ex, ey)
            print(f"  C3D ROI disp bbox → {bbox_disp}  identifying markers…", flush=True)
            _compute_c3d_led_roi(bbox_disp)
            _redraw_c3d(int(opti_slider.val))
            _refresh_signal_plot()
        # re-enable 3D rotation and deactivate ROI mode
        ax_c3d.mouse_init()
        _c3d_roi_active[0] = False
        btn_c3d_roi.label.set_text("☐  Set C3D ROI")
        btn_c3d_roi.ax.set_facecolor("#1a1a33")
        btn_c3d_roi.label.set_color("#8888ff")
        fig.canvas.draw_idle()
        return

    # ── video panel release ──
    for which, ax, patch in (("opti", ax_opti, _opti_rect),
                              ("orb",  ax_orb,  _orb_rect)):
        if not D[f"{which}_drawing"]:
            continue
        D[f"{which}_drawing"] = False
        if event.button != 1:
            continue
        sx, sy = D[f"{which}_draw_start"]
        ex = event.xdata if event.xdata is not None else sx
        ey = event.ydata if event.ydata is not None else sy

        if abs(ex - sx) < 2 or abs(ey - sy) < 2:
            patch.set_visible(False)
            fig.canvas.draw_idle()
            continue

        bbox = (int(min(sx, ex)), int(min(sy, ey)),
                int(max(sx, ex)), int(max(sy, ey)))
        D[f"{which}_bbox"] = bbox
        patch.set_visible(False)    # bbox is now drawn inside the frame itself

        print(f"\n  {which} bbox → {bbox}  computing medians…", flush=True)
        D[f"{which}_meds"] = _compute_bbox_medians(D[f"{which}_frames"], bbox)
        print(f"  done ({len(D[f'{which}_meds'])} frames)")

        f = int(opti_slider.val) if which == "opti" else int(orb_slider.val)
        if which == "opti":
            _redraw_opti(f)
        else:
            _redraw_orb(f)
        _refresh_signal_plot()
        fig.canvas.draw_idle()


fig.canvas.mpl_connect("scroll_event",         _on_scroll)
fig.canvas.mpl_connect("button_press_event",   _on_press)
fig.canvas.mpl_connect("motion_notify_event",  _on_motion)
fig.canvas.mpl_connect("button_release_event", _on_release)


def on_c3d_roi(event):
    """Toggle C3D ROI draw mode on/off."""
    if D["c3d_xyz"] is None:
        return
    _c3d_roi_active[0] = not _c3d_roi_active[0]
    if _c3d_roi_active[0]:
        ax_c3d.disable_mouse_rotation()
        btn_c3d_roi.label.set_text("✏  Draw ROI now…")
        btn_c3d_roi.ax.set_facecolor("#442244")
        btn_c3d_roi.label.set_color("#ffaaff")
    else:
        ax_c3d.mouse_init()
        btn_c3d_roi.label.set_text("☐  Set C3D ROI")
        btn_c3d_roi.ax.set_facecolor("#1a1a33")
        btn_c3d_roi.label.set_color("#8888ff")
    fig.canvas.draw_idle()


btn_c3d_roi.on_clicked(on_c3d_roi)


# ── save / load sync offset ───────────────────────────────────────────────────

def _offset_path():
    """Return path to offset.txt, preferring the optitrack folder."""
    folder = D["opti_dir"] or D["orb_dir"]
    if folder is None:
        return None
    return os.path.join(folder, "offset.txt")


def on_save_offset(event):
    path = _offset_path()
    if path is None:
        print("Load at least one file before saving offset.")
        return
    if D["opti_frames"] is None or D["orb_frames"] is None:
        print("Both optitrack and orbbec must be loaded before saving offset.")
        return

    f_o  = int(opti_slider.val)
    f_r  = int(orb_slider.val)
    t_o  = f_o / (D["opti_fps"] or 1.0)
    t_r  = f_r / (D["orb_fps"]  or 1.0)

    def _fmt_bbox(b):
        return ",".join(str(int(v)) for v in b) if b is not None else "none"

    def _fmt_markers(sel):
        return ",".join(str(int(i)) for i in sel) if sel is not None and len(sel) else "none"

    lines = [
        f"# Sync offset saved by sync_and_calibrate.py",
        f"optitrack_frame={f_o}",
        f"optitrack_time_s={t_o:.6f}",
        f"optitrack_fps={D['opti_fps']:.6f}",
        f"orbbec_frame={f_r}",
        f"orbbec_time_s={t_r:.6f}",
        f"orbbec_fps={D['orb_fps']:.6f}",
        f"# bounding boxes (x1,y1,x2,y2 in image pixel coords)",
        f"opti_bbox={_fmt_bbox(D['opti_bbox'])}",
        f"orb_bbox={_fmt_bbox(D['orb_bbox'])}",
        f"# C3D ROI: selected marker indices",
        f"c3d_sel_markers={_fmt_markers(D['c3d_sel_markers'])}",
        f"# threshold slider values",
        f"opti_thresh={int(thresh_opti_sl.val)}",
        f"orb_thresh={int(thresh_orb_sl.val)}",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Offset saved → {path}")
    print(f"  optitrack : frame {f_o}  t={t_o:.6f}s")
    print(f"  orbbec    : frame {f_r}  t={t_r:.6f}s")
    print(f"  opti_bbox={D['opti_bbox']}  orb_bbox={D['orb_bbox']}")
    print(f"  c3d_sel_markers={D['c3d_sel_markers']}")
    print(f"  opti_thresh={int(thresh_opti_sl.val)}  orb_thresh={int(thresh_orb_sl.val)}")
    btn_save_off.label.set_text("✓ Saved")
    btn_save_off.ax.set_facecolor("#334400")
    fig.canvas.draw_idle()


def on_load_offset(event):
    # First try auto-detect next to the already loaded files
    path = _offset_path()
    if path is None or not os.path.exists(path):
        # Fall back to a file dialog
        path = _ask_file("Open offset.txt",
                         [("Text files", "*.txt"), ("All", "*.*")])
    if not path or not os.path.exists(path):
        print("offset.txt not found.")
        return

    cfg = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()

    f_o = int(cfg.get("optitrack_frame", 0))
    f_r = int(cfg.get("orbbec_frame",   0))

    if D["opti_frames"] is not None:
        f_o = max(0, min(f_o, D["opti_n"] - 1))
        D["_sync_updating"] = True
        opti_slider.set_val(f_o)
        D["_sync_updating"] = False
        _redraw_opti(f_o)
        D["_sync_offsets"][0] = f_o
        D["_opti_frac"] = float(f_o)

    if D["orb_frames"] is not None:
        f_r = max(0, min(f_r, D["orb_n"] - 1))
        D["_sync_updating"] = True
        orb_slider.set_val(f_r)
        D["_sync_updating"] = False
        _redraw_orb(f_r)
        D["_sync_offsets"][1] = f_r
        D["_orb_frac"] = float(f_r)

    D["_sync_updating"] = True
    sync_slider.set_val(0.0)
    D["_sync_updating"] = False

    # ── restore bounding boxes ────────────────────────────────────────────────
    def _parse_bbox(s):
        if not s or s.strip().lower() == "none":
            return None
        parts = [int(v) for v in s.strip().split(",")]
        return tuple(parts) if len(parts) == 4 else None

    opti_bbox_new = _parse_bbox(cfg.get("opti_bbox", "none"))
    if opti_bbox_new is not None and D["opti_frames"] is not None:
        D["opti_bbox"] = opti_bbox_new
        D["opti_meds"] = _compute_bbox_medians(D["opti_frames"], opti_bbox_new)
        _redraw_opti(f_o)
        print(f"  opti_bbox restored: {opti_bbox_new}")

    orb_bbox_new = _parse_bbox(cfg.get("orb_bbox", "none"))
    if orb_bbox_new is not None and D["orb_frames"] is not None:
        D["orb_bbox"] = orb_bbox_new
        D["orb_meds"] = _compute_bbox_medians(D["orb_frames"], orb_bbox_new)
        _redraw_orb(f_r)
        print(f"  orb_bbox restored: {orb_bbox_new}")

    # ── restore C3D ROI marker selection ──────────────────────────────────────
    sel_str = cfg.get("c3d_sel_markers", "none")
    if sel_str.strip().lower() != "none" and D["c3d_xyz"] is not None:
        try:
            sel = np.array([int(i) for i in sel_str.strip().split(",")], dtype=int)
            sel = sel[sel < D["c3d_xyz"].shape[1]]   # guard against stale indices
            if len(sel):
                D["c3d_sel_markers"] = sel
                xyz       = D["c3d_xyz"]
                n_vis_roi = np.all(np.isfinite(xyz[:, sel, :]), axis=0).sum(axis=0)
                D["c3d_led_roi"] = (n_vis_roi > 0).astype(np.float32)
                _redraw_c3d(f_o)
                print(f"  c3d_sel_markers restored: {sel}")
        except Exception as e:
            print(f"  WARNING: could not restore c3d_sel_markers: {e}")

    # ── restore threshold sliders ─────────────────────────────────────────────
    # Set values directly on the slider knob (bypasses on_changed to avoid
    # double-refresh); then call _refresh_signal_plot once at the end.
    try:
        thr_o = int(cfg.get("opti_thresh", int(thresh_opti_sl.val)))
        thresh_opti_sl.eventson = False
        thresh_opti_sl.set_val(thr_o)
        thresh_opti_sl.eventson = True
        print(f"  opti_thresh restored: {thr_o}")
    except Exception:
        pass
    try:
        thr_r = int(cfg.get("orb_thresh", int(thresh_orb_sl.val)))
        thresh_orb_sl.eventson = False
        thresh_orb_sl.set_val(thr_r)
        thresh_orb_sl.eventson = True
        print(f"  orb_thresh restored: {thr_r}")
    except Exception:
        pass

    _refresh_signal_plot()
    _refresh_info()

    print(f"Offset loaded ← {path}")
    print(f"  optitrack : frame {f_o}")
    print(f"  orbbec    : frame {f_r}")
    btn_load_off.label.set_text("✓ Loaded")
    btn_load_off.ax.set_facecolor("#223322")
    fig.canvas.draw_idle()


btn_save_off.on_clicked(on_save_offset)
btn_load_off.on_clicked(on_load_offset)


# ── threshold callbacks ───────────────────────────────────────────────────────

def on_thresh_opti(val):
    if D["opti_meds"] is None:
        return
    thr = int(val)
    led = (D["opti_meds"] > thr).astype(np.float32)
    line_opti.set_ydata(led * 1.06)
    line_opti.set_label(f"Optitrack (thr={thr})")
    ax_sig.legend(fontsize=8, facecolor="#222222", labelcolor="white", loc="upper right")
    _refresh_info()
    fig.canvas.draw_idle()


def on_thresh_orb(val):
    if D["orb_meds"] is None:
        return
    thr = int(val)
    led = (D["orb_meds"] > thr).astype(np.float32)
    line_orb.set_ydata(led * 1.12)
    line_orb.set_label(f"Orbbec (thr={thr})")
    ax_sig.legend(fontsize=8, facecolor="#222222", labelcolor="white", loc="upper right")
    _refresh_info()
    fig.canvas.draw_idle()


thresh_opti_sl.on_changed(on_thresh_opti)
thresh_orb_sl.on_changed(on_thresh_orb)


# ── frame slider callbacks ────────────────────────────────────────────────────

def update_opti(val):
    if D["_sync_updating"]:
        return
    f = int(opti_slider.val)
    _redraw_opti(f)
    D["_sync_offsets"][0] = f
    D["_sync_updating"] = True
    sync_slider.set_val(0.0)
    D["_sync_updating"] = False
    fig.canvas.draw_idle()


def update_orb(val):
    if D["_sync_updating"]:
        return
    f = int(orb_slider.val)
    _redraw_orb(f)
    D["_sync_offsets"][1] = f
    D["_sync_updating"] = True
    sync_slider.set_val(0.0)
    D["_sync_updating"] = False
    fig.canvas.draw_idle()


def update_sync(val):
    if D["_sync_updating"]:
        return
    D["_sync_updating"] = True
    t    = float(sync_slider.val)
    of   = D["opti_fps"] or 1.0
    rf   = D["orb_fps"]  or 1.0
    n_o  = D["opti_n"]   or 1
    n_r  = D["orb_n"]    or 1
    f_o  = int(np.clip(D["_sync_offsets"][0] + round(t * of), 0, n_o - 1))
    f_r  = int(np.clip(D["_sync_offsets"][1] + round(t * rf), 0, n_r - 1))
    opti_slider.set_val(f_o)
    orb_slider.set_val(f_r)
    D["_opti_frac"] = float(f_o)
    D["_orb_frac"]  = float(f_r)
    _redraw_opti(f_o)
    _redraw_orb(f_r)
    D["_sync_updating"] = False
    fig.canvas.draw_idle()


opti_slider.on_changed(update_opti)
orb_slider.on_changed(update_orb)
sync_slider.on_changed(update_sync)


# ── playback ──────────────────────────────────────────────────────────────────

def on_play(event):
    D["playing"] = not D["playing"]
    if D["playing"]:
        btn_play.label.set_text("⏸  Stop")
        btn_play.ax.set_facecolor("#552222")
        D["_last_tick"]      = time.perf_counter()
        D["_opti_frac"]      = float(opti_slider.val)
        D["_orb_frac"]       = float(orb_slider.val)
        D["_sync_offsets"]   = [int(opti_slider.val), int(orb_slider.val)]
        D["_sync_updating"]  = True
        sync_slider.set_val(0.0)
        D["_sync_updating"]  = False
    else:
        btn_play.label.set_text("▶  Play")
        btn_play.ax.set_facecolor("#225522")
    fig.canvas.draw_idle()


def on_reset(event):
    D["playing"] = False
    btn_play.label.set_text("▶  Play")
    btn_play.ax.set_facecolor("#225522")
    D["_sync_updating"] = True
    opti_slider.set_val(0)
    orb_slider.set_val(0)
    sync_slider.set_val(0.0)
    D["_sync_updating"] = False
    D["_sync_offsets"]  = [0, 0]
    D["_opti_frac"]     = 0.0
    D["_orb_frac"]      = 0.0
    fig.canvas.draw_idle()


def _tick():
    if not D["playing"]:
        return
    if D["opti_frames"] is None or D["orb_frames"] is None:
        return

    now  = time.perf_counter()
    dt   = now - D["_last_tick"]
    D["_last_tick"] = now

    of  = D["opti_fps"] or 1.0
    rf  = D["orb_fps"]  or 1.0
    n_o = D["opti_n"]   or 1
    n_r = D["orb_n"]    or 1

    D["_opti_frac"] += dt * of
    D["_orb_frac"]  += dt * rf

    new_o   = int(D["_opti_frac"])
    new_r   = int(D["_orb_frac"])
    stopped = False

    if new_o >= n_o - 1:
        new_o = n_o - 1; D["_opti_frac"] = float(new_o); stopped = True
    if new_r >= n_r - 1:
        new_r = n_r - 1; D["_orb_frac"]  = float(new_r); stopped = True

    o_chg = new_o != int(opti_slider.val)
    r_chg = new_r != int(orb_slider.val)

    if o_chg:
        opti_slider.eventson = False
        opti_slider.set_val(new_o)
        opti_slider.eventson = True
        _redraw_opti(new_o)

    if r_chg:
        orb_slider.eventson = False
        orb_slider.set_val(new_r)
        orb_slider.eventson = True
        _redraw_orb(new_r)

    if o_chg or r_chg:
        elapsed = (new_o - D["_sync_offsets"][0]) / of
        D["_sync_updating"]  = True
        sync_slider.eventson = False
        sync_slider.set_val(np.clip(elapsed, 0.0, sync_slider.valmax))
        sync_slider.eventson = True
        D["_sync_updating"]  = False
        fig.canvas.draw_idle()

    if stopped:
        D["playing"] = False
        btn_play.label.set_text("▶  Play")
        btn_play.ax.set_facecolor("#225522")
        fig.canvas.draw_idle()


def on_step(direction):
    if D["opti_frames"] is None or D["orb_frames"] is None:
        return
    of  = D["opti_fps"] or 1.0
    rf  = D["orb_fps"]  or 1.0
    n_o = D["opti_n"]   or 1
    n_r = D["orb_n"]    or 1
    ratio = max(1, round(rf / of))
    new_o = int(np.clip(opti_slider.val + direction,         0, n_o - 1))
    new_r = int(np.clip(orb_slider.val  + direction * ratio, 0, n_r - 1))
    opti_slider.set_val(new_o)
    orb_slider.set_val(new_r)
    D["_opti_frac"] = float(new_o)
    D["_orb_frac"]  = float(new_r)


# ── wire up buttons + keys ────────────────────────────────────────────────────

btn_load_opti.on_clicked(on_load_opti)
btn_load_c3d.on_clicked(on_load_c3d)
btn_load_orb.on_clicked(on_load_orb)
btn_play.on_clicked(on_play)
btn_reset.on_clicked(on_reset)
btn_minus.on_clicked(lambda e: on_step(-1))
btn_plus.on_clicked(lambda e: on_step(+1))


def _on_key(event):
    if event.key == "q":
        plt.close(fig)
    elif event.key == "right":
        on_step(+1); fig.canvas.draw_idle()
    elif event.key == "left":
        on_step(-1); fig.canvas.draw_idle()


fig.canvas.mpl_connect("key_press_event", _on_key)

timer = fig.canvas.new_timer(interval=33)
timer.add_callback(_tick)
timer.start()

plt.show()
timer.stop()
