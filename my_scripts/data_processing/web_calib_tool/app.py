"""Gradio UI for the iPad<->Mocap calibration tool.

Run:  python3 app.py          (then open http://localhost:7860)

Stages:
  Tab 0  Data            - set the iPad video + C3D paths and load them
  Tab 1  Marker layout   - click where the 6 reflective markers sit on the board
  Tab 2  Sync            - LED blink: ROI + threshold (video), 3D box (mocap),
                           cross-correlation offset + fine-tune
  Tab 3  Trim            - shared start/end trim for both streams
  Tab 4  Calibrate       - solvePnP + Umeyama -> mocap->camera transform
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

import config
import marker_layout as ml
from data_loader import load_c3d, pre_extract_frames
from sync import (
    auto_find_led,
    compute_mocap_trace,
    compute_video_trace,
    cross_correlate,
    mocap_index_for_video_time,
    save_sync,
    threshold_trace,
)
from mocap_view import render_mocap_2d, render_mocap_3d
from trim import apply_trim, save_trim
from calibrate import run_calibration

# ----------------------------------------------------------------------
# Data globals -- filled by load_data() (from the "0. Data" tab, or on app
# start with the configured / last-saved paths).  Handlers read these at
# call time, so switching data in the UI re-targets every tab.
# ----------------------------------------------------------------------
FRAMES: list = []
TIMES: np.ndarray = np.array([])
MOCAP: dict = {}
N_VID = 0
N_MOC = 0
FPS_M = 100.0
IMG_H, IMG_W = 720, 1280
VIDEO_DUR = 0.0
CURRENT_VIDEO_PATH = str(config.VIDEO_PATH)
CURRENT_C3D_PATH = str(config.C3D_PATH)
FRAME_ARR: np.ndarray | None = None  # (N, H, W, 3) uint8 BGR - all frames in RAM


# ----------------------------------------------------------------------
# Tab 0 - Data loading
# ----------------------------------------------------------------------
def load_data(video_path: str, c3d_path: str):
    """Load the iPad video + C3D mocap file into the module globals.

    Returns GUI updates in this order: (status, video_scrub, mocap_scrub,
    sync_scrub, trim_start, trim_end, video_view, mocap3d, mocap2d).
    """
    global FRAMES, TIMES, MOCAP, N_VID, N_MOC, FPS_M, VIDEO_DUR
    global CURRENT_VIDEO_PATH, CURRENT_C3D_PATH, FRAME_ARR, IMG_H, IMG_W
    video_path = str(Path(video_path).expanduser()).strip()
    c3d_path = str(Path(c3d_path).expanduser()).strip()
    try:
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not Path(c3d_path).is_file():
            raise FileNotFoundError(f"C3D not found: {c3d_path}")
        frames, times = pre_extract_frames(video_path)
        mocap = load_c3d(c3d_path)
    except Exception as e:
        return (f"ERROR: {e}", *([gr.update()] * 6), None, None, None)

    FRAMES, TIMES, MOCAP = frames, times, mocap
    N_VID = len(FRAMES)
    N_MOC = MOCAP["n_frames"]
    FPS_M = MOCAP["fps"]
    VIDEO_DUR = float(TIMES[-1])
    CURRENT_VIDEO_PATH = video_path
    CURRENT_C3D_PATH = c3d_path
    config.save_settings(video_path, c3d_path)

    # Preload every frame into RAM (BGR uint8) for seamless scrubbing.
    FRAME_ARR = _preload_frames(FRAMES)
    IMG_H, IMG_W = FRAME_ARR.shape[1], FRAME_ARR.shape[2]

    img_v = _render_video(0, None, 1.0)
    img3 = render_mocap_3d(MOCAP, 0, None)
    img2 = render_mocap_2d(MOCAP, 0, "top", None)
    msg = (f"Loaded:\n  video: {Path(video_path).name}  "
           f"({N_VID} frames, {VIDEO_DUR:.1f} s, "
           f"{FRAME_ARR.nbytes / 1e6:.0f} MB in RAM)\n"
           f"  mocap: {Path(c3d_path).name}  "
           f"({N_MOC} frames @ {FPS_M:.0f} Hz)\n\n"
           "Proceed to tabs 1-4.")
    return (msg,
            gr.update(maximum=max(N_VID - 1, 0), value=0),
            gr.update(maximum=max(N_MOC - 1, 0), value=0),
            gr.update(maximum=VIDEO_DUR, value=0),
            gr.update(maximum=VIDEO_DUR, value=0),
            gr.update(maximum=VIDEO_DUR, value=VIDEO_DUR),
            img_v, img3, img2,
            gr.update(value=CURRENT_VIDEO_PATH))

DEFAULT_STATE = {
    "placed": [],          # list of [x_mm, y_mm]
    "video_roi": None,     # [x0, y0, x1, y1]
    "roi_click": None,     # first of two ROI clicks
    "box_lo": None,
    "box_hi": None,
    "video_trace": None,
    "video_bin": None,
    "mocap_bin": None,
    "threshold": 128.0,
    "offset": 0.0,
    "zoom": 1.0,
}


def _frame_bgr(i: int) -> np.ndarray:
    """BGR frame; served from the in-RAM array after load_data (seamless scrub)."""
    if FRAME_ARR is not None:
        return FRAME_ARR[i]
    return cv2.imread(str(FRAMES[i]))


def _rgb(img: np.ndarray) -> np.ndarray:
    """BGR numpy image -> RGB for gr.Image display."""
    return img[..., ::-1]


def _preload_frames(frames: list) -> np.ndarray:
    """Decode all cached JPEGs into one (N, H, W, 3) uint8 BGR array."""
    first = cv2.imread(str(frames[0]))
    H, W = first.shape[:2]
    arr = np.empty((len(frames), H, W, 3), np.uint8)
    arr[0] = first
    for i in range(1, len(frames)):
        arr[i] = cv2.imread(str(frames[i]))
    return arr


def _frame_size() -> tuple[int, int]:
    """Return (W, H) of the video frames."""
    if FRAME_ARR is not None:
        return FRAME_ARR.shape[2], FRAME_ARR.shape[1]
    img = _frame_bgr(0)
    return img.shape[1], img.shape[0]


# ----------------------------------------------------------------------
# Tab 1 - Marker layout
# ----------------------------------------------------------------------
def tab1_render_canvas(placed) -> np.ndarray:
    return _rgb(ml.render_canvas(placed))


def tab1_on_click(evt: gr.SelectData, state):
    x_px, y_px = evt.index
    x_mm, y_mm = ml.px_to_mm(x_px, y_px)
    x_mm, y_mm = ml.snap_to_grid(x_mm, y_mm)
    if len(state["placed"]) >= config.NUM_MARKERS:
        return (_rgb(ml.render_canvas(state["placed"])), state,
                f"Already have {config.NUM_MARKERS} markers - use Undo to remove one.")
    state["placed"].append([x_mm, y_mm])
    named = [(f"{i+1}", p[0], p[1]) for i, p in enumerate(state["placed"])]
    msg = "Placed: " + ", ".join(f"{i+1}@({x:.0f},{y:.0f})mm"
                                 for i, (x, y) in enumerate(state["placed"]))
    return _rgb(ml.render_canvas(named)), state, msg


def tab1_undo(state):
    if state["placed"]:
        state["placed"].pop()
    named = [(f"{i+1}", p[0], p[1]) for i, p in enumerate(state["placed"])]
    return (_rgb(ml.render_canvas(named)), state,
            f"{len(state['placed'])} markers placed")


def tab1_clear(state):
    state["placed"] = []
    return _rgb(ml.render_canvas([])), state, "cleared"


def tab1_verify(state):
    if len(state["placed"]) != config.NUM_MARKERS:
        return (f"Need exactly {config.NUM_MARKERS} markers, have "
                f"{len(state['placed'])}.")
    markers_mm = np.array([[x, y, 0.0] for x, y in state["placed"]], float)
    v = ml.verify_markers(markers_mm, MOCAP)
    if not v.get("ok"):
        return f"Verification failed: {v.get('error')}"
    order = ", ".join(v["assigned"])
    return (f"OK. Click order maps to C3D labels: {order}\n"
            f"mean distance error: {v['mean_dist_err_mm']:.1f} mm, "
            f"max: {v['max_dist_err_mm']:.1f} mm")


def tab1_save(state):
    if len(state["placed"]) != config.NUM_MARKERS:
        return f"Need exactly {config.NUM_MARKERS} markers, have {len(state['placed'])}."
    markers_mm = np.array([[x, y, 0.0] for x, y in state["placed"]], float)
    v = ml.verify_markers(markers_mm, MOCAP)
    ml.save_markers(markers_mm, v)
    return "Saved output/markers.json" + (f" (labels: {', '.join(v['assigned'])})"
                                          if v.get("ok") else " (unverified order)")


# ----------------------------------------------------------------------
# Tab 2 - Sync (video ROI editor)
# ----------------------------------------------------------------------
def _zoom_crop_region(roi, zoom: float) -> tuple[int, int, int, int]:
    """Return (x0, y0, cw, ch) of the zoom crop in full-frame pixels."""
    W, H = _frame_size()
    if not zoom or zoom <= 1.0:
        return 0, 0, W, H
    cx, cy = W // 2, H // 2
    if roi is not None:
        cx, cy = (roi[0] + roi[2]) // 2, (roi[1] + roi[3]) // 2
    cw = max(int(W / zoom), 1)
    ch = max(int(H / zoom), 1)
    x0 = max(0, min(cx - cw // 2, W - cw))
    y0 = max(0, min(cy - ch // 2, H - ch))
    return x0, y0, cw, ch


def _render_video(idx: int, roi, zoom: float = 1.0) -> np.ndarray:
    """Full (or zoom-cropped) frame with the ROI rect drawn, returned as RGB.

    Drawn on a copy so the shared in-RAM frame array is never painted on.
    """
    full = _frame_bgr(int(idx))
    x0, y0, cw, ch = _zoom_crop_region(roi, zoom)
    if zoom and zoom > 1.0:
        img = full[y0:y0 + ch, x0:x0 + cw].copy()
    else:
        img = full.copy()

    if roi is not None:
        rx0, ry0, rx1, ry1 = (int(v) for v in roi)
        bx0 = max(rx0 - x0, 0)
        by0 = max(ry0 - y0, 0)
        bx1 = min(rx1 - x0, img.shape[1])
        by1 = min(ry1 - y0, img.shape[0])
        if bx1 > bx0 and by1 > by0:
            cv2.rectangle(img, (bx0, by0), (bx1, by1), (0, 255, 0), 2)
            cv2.putText(img, "LED ROI", (bx0, max(by0 - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                        cv2.LINE_AA)
    return _rgb(img)


def _roi_crop(idx: int, roi) -> np.ndarray | None:
    """Upscaled RGB crop of the ROI box contents (for the ROI zoom pane)."""
    if roi is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in roi)
    if x1 <= x0 or y1 <= y0:
        return None
    img = _frame_bgr(int(idx))[y0:y1, x0:x1]
    h = img.shape[0]
    if h > 0:
        scale = min(4.0, 220.0 / max(h, 1))
        if scale > 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_NEAREST)
    return _rgb(img)


def tab2_show_video(idx, state):
    return _render_video(int(idx), state["video_roi"], state.get("zoom", 1.0))


def tab2_show_zoom(idx, state):
    return _roi_crop(int(idx), state["video_roi"])


def tab2_render(idx, state):
    """Combined scrub handler: main view + ROI pane in ONE event (half the
    round-trips -> less lag)."""
    i = int(idx)
    return (_render_video(i, state["video_roi"], state.get("zoom", 1.0)),
            _roi_crop(i, state["video_roi"]))


def tab2_zoom(z, idx, state):
    """Zoom slider: re-render main view + ROI pane at the current frame."""
    state["zoom"] = float(z)
    return (_render_video(int(idx), state["video_roi"], state["zoom"]),
            _roi_crop(int(idx), state["video_roi"]))


def tab2_nudge(corner, axis, delta, idx, state):
    """Fine-tune one corner of the ROI box by (+/-)delta px."""
    roi = state["video_roi"]
    if roi is None:
        return (state, "Draw the ROI box first (2 clicks on the video).",
                None, None)
    x0, y0, x1, y1 = (int(v) for v in roi)
    if corner == "tl":
        if axis == "x":
            x0 += delta
        else:
            y0 += delta
    elif corner == "tr":
        if axis == "x":
            x1 += delta
        else:
            y0 += delta
    elif corner == "bl":
        if axis == "x":
            x0 += delta
        else:
            y1 += delta
    else:  # br
        if axis == "x":
            x1 += delta
        else:
            y1 += delta
    W, H = _frame_size()
    x0 = max(0, min(x0, W))
    x1 = max(0, min(x1, W))
    y0 = max(0, min(y0, H))
    y1 = max(0, min(y1, H))
    if x0 == x1 or y0 == y1:
        return (state, "Box collapsed - keep a positive size.", None, None)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    state["video_roi"] = [x0, y0, x1, y1]
    state["roi_click"] = None
    return (state,
            f"ROI = {state['video_roi']}",
            _render_video(int(idx), state["video_roi"], state.get("zoom", 1.0)),
            _roi_crop(int(idx), state["video_roi"]))


def tab2_video_click(evt: gr.SelectData, state, idx):
    """2-click ROI draw; maps click coords back to full-frame px when zoomed."""
    cx, cy = evt.index
    x0, y0, _, _ = _zoom_crop_region(state["video_roi"], state.get("zoom", 1.0))
    x, y = min(int(cx) + x0, max(_frame_size()[0] - 1, 0)), \
           min(int(cy) + y0, max(_frame_size()[1] - 1, 0))
    if state["roi_click"] is None:
        state["roi_click"] = (x, y)
        img = _render_video(int(idx), state["video_roi"], state.get("zoom", 1.0))
        return (state, "ROI: click opposite corner", img,
                _roi_crop(int(idx), state["video_roi"]))
    (x0, y0), (x1, y1) = state["roi_click"], (x, y)
    state["video_roi"] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    state["roi_click"] = None
    img = _render_video(int(idx), state["video_roi"], state.get("zoom", 1.0))
    return (state, "ROI set. Press 'Compute traces'.", img,
            _roi_crop(int(idx), state["video_roi"]))


def tab2_reset_roi(idx, state):
    state["video_roi"] = None
    state["roi_click"] = None
    return (state, "ROI cleared",
            _render_video(int(idx), None, state.get("zoom", 1.0)), None)


def _box_arrays(state):
    if state["box_lo"] is None:
        return None
    return (np.array(state["box_lo"]), np.array(state["box_hi"]))


def tab2_show_mocap(midx, state):
    box = _box_arrays(state)
    img3 = render_mocap_3d(MOCAP, int(midx), box)
    img2 = render_mocap_2d(MOCAP, int(midx), "top", box)
    return img3, img2


def tab2_auto_led(state):
    led = auto_find_led(MOCAP)
    if led is None:
        return (state, "No stationary blinking marker found - set the box manually.")
    pos = led["position_mm"]
    half = 50.0
    state["box_lo"] = (pos - half).tolist()
    state["box_hi"] = (pos + half).tolist()
    return (state,
            f"Auto box around '{led['name']}' at ({pos[0]:.0f}, {pos[1]:.0f}, "
            f"{pos[2]:.0f}) mm, size +/-{half:.0f} mm")


def tab2_set_box(x, y, z, half, state):
    state["box_lo"] = [x - half, y - half, z - half]
    state["box_hi"] = [x + half, y + half, z + half]
    return state


def tab2_compute_traces(state):
    if state["video_roi"] is None:
        return (state, None, None, "Set the video ROI first (2 clicks).")
    trace, times = compute_video_trace(tuple(state["video_roi"]),
                                       video_path=CURRENT_VIDEO_PATH)
    state["video_trace"] = trace.tolist()
    state["video_bin"] = threshold_trace(trace, state["threshold"]).tolist()
    box = _box_arrays(state)
    if box is None:
        state["mocap_bin"] = None
        msg = "Video trace done. Set the mocap 3D box to compute its trace."
    else:
        state["mocap_bin"] = compute_mocap_trace(MOCAP, box[0], box[1]).tolist()
        msg = "Both traces computed."
    fig = _traces_figure(state)
    return state, fig, state["mocap_bin"] is not None, msg


def tab2_threshold(thr, state):
    state["threshold"] = float(thr)
    if state["video_trace"] is not None:
        state["video_bin"] = threshold_trace(
            np.array(state["video_trace"]), thr).tolist()
    fig = _traces_figure(state)
    return state, fig


def tab2_auto_sync(state):
    if state["video_bin"] is None or state["mocap_bin"] is None:
        return (state, None, "Compute both traces first.")
    off, agree = cross_correlate(
        np.array(state["video_bin"]), TIMES, np.array(state["mocap_bin"]),
        FPS_M, max_offset=60.0)
    state["offset"] = float(off)
    fig = _traces_figure(state)
    return (state, fig,
            f"Proposed offset = {off:.3f} s (video_time = mocap_time + offset), "
            f"agreement = {agree:.3f}. Fine-tune below, then Save.")


def tab2_offset(off, state):
    state["offset"] = float(off)
    fig = _traces_figure(state)
    return state, fig


def tab2_save_sync(state):
    if state["video_bin"] is None or state["mocap_bin"] is None:
        return "Compute both traces first."
    save_sync(state["offset"], 0.0, state["video_roi"], state["box_lo"],
              state["box_hi"], state["threshold"])
    return (f"Saved output/sync.json  offset={state['offset']:.3f} s")


def tab2_sync_scrub(t_video, state):
    """Move both viewers to the synchronized instant."""
    vi = int(np.argmin(np.abs(TIMES - t_video)))
    mi = mocap_index_for_video_time(TIMES[vi], state["offset"], FPS_M)
    mi = max(0, min(mi, N_MOC - 1))
    img_v = _render_video(vi, state["video_roi"], state.get("zoom", 1.0))
    zv = _roi_crop(vi, state["video_roi"])
    box = _box_arrays(state)
    img3 = render_mocap_3d(MOCAP, mi, box)
    img2 = render_mocap_2d(MOCAP, mi, "top", box)
    return vi, mi, img_v, zv, img3, img2


def _traces_figure(state):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    has_v = state["video_trace"] is not None
    has_b = state["video_bin"] is not None and state["mocap_bin"] is not None

    if has_v:
        t = np.array(state["video_trace"])
        norm = (t - t.min()) / max(t.max() - t.min(), 1e-9)
        ax.plot(TIMES, norm, color="tab:blue", lw=1, label="video intensity (norm)")
    if state["video_bin"] is not None:
        ax.step(TIMES, np.array(state["video_bin"]), color="tab:green", lw=1.2,
                where="post", label="video binary")
    if state["mocap_bin"] is not None:
        m = np.array(state["mocap_bin"])
        t_m = np.arange(len(m)) / FPS_M
        ax.step(t_m + state["offset"], m + 1.5, color="tab:red", lw=1.2,
                where="post", label="mocap binary (+1.5)")
    ax.set_xlabel("video time (s)")
    ax.set_ylabel("signal")
    ax.set_title(f"offset = {state['offset']:.3f} s   "
                 "(video_time = mocap_time + offset)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Tab 3 - Trim
# ----------------------------------------------------------------------
def tab3_apply(start_s, end_s, state):
    res = apply_trim(TIMES, MOCAP, state["offset"], float(start_s), float(end_s))
    return (f"start {res['start_s']:.2f}s  end {res['end_s']:.2f}s  "
            f"duration {res['duration_s']:.2f}s\n"
            f"video frames {res['video_first']}..{res['video_last']} "
            f"({res['video_frames']})\n"
            f"mocap frames {res['mocap_first']}..{res['mocap_last']} "
            f"({res['mocap_frames']})\nvalid: {res['valid']}")


def tab3_save(start_s, end_s):
    save_trim(float(start_s), float(end_s))
    return f"Saved output/trim.json  {start_s:.2f}s .. {end_s:.2f}s"


# ----------------------------------------------------------------------
# Tab 4 - Calibrate
# ----------------------------------------------------------------------
def tab4_run():
    try:
        r = run_calibration(video_path=CURRENT_VIDEO_PATH,
                            c3d_path=CURRENT_C3D_PATH)
        lines = [
            f"Used {r['n_frames_used']} frames, {r['n_correspondences']} "
            "marker correspondences",
            f"mean residual: {r['mean_residual_mm']:.2f} mm",
            f"median: {r['median_residual_mm']:.2f} mm   "
            f"p95: {r['p95_residual_mm']:.2f} mm   max: {r['max_residual_mm']:.2f} mm",
            "R =",
            np.array2string(np.round(np.array(r["rotation"]), 5), precision=5),
            "t (mm) = " + np.array2string(np.round(np.array(r["translation_mm"]), 2)),
            "Saved output/transform.json",
        ]
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}"


# ----------------------------------------------------------------------
# Build the UI
# ----------------------------------------------------------------------
def build_app():
    with gr.Blocks(title="iPad <-> Mocap Calibration") as demo:
        state = gr.State(DEFAULT_STATE.copy())

        with gr.Tab("0. Data"):
            gr.Markdown("Set the paths to your **iPad video (.mp4)** and "
                        "**mocap C3D (.c3d)** file, then click **Load data**. "
                        "Your choice is remembered in `output/settings.json` "
                        "and pre-filled next time.")
            _saved = config.load_settings()
            vid_input = gr.Textbox(label="iPad video (.mp4)",
                                   value=_saved.get("video_path",
                                                    str(config.VIDEO_PATH)))
            c3d_input = gr.Textbox(label="Mocap C3D (.c3d)",
                                   value=_saved.get("c3d_path",
                                                    str(config.C3D_PATH)))
            load_btn = gr.Button("Load data", variant="primary")
            load_status = gr.Textbox(label="Status", lines=5, interactive=False)

        with gr.Tab("1. Marker layout"):
            gr.Markdown("Click on the **orange dots** where the reflective "
                        "markers sit (grid = 40 mm). Click 6 in total.")
            with gr.Row():
                canvas = gr.Image(value=_rgb(ml.render_canvas([])), type="numpy",
                                  height=560, interactive=False)
                with gr.Column():
                    ml_info = gr.Textbox(label="Placement", lines=8,
                                         interactive=False)
                    gr.Markdown("### Actions")
                    ml_undo = gr.Button("Undo last")
                    ml_clear = gr.Button("Clear all")
                    ml_verify = gr.Button("Verify vs C3D")
                    ml_save = gr.Button("Save markers")
                    ml_result = gr.Textbox(label="Result", lines=6,
                                           interactive=False)
            canvas.select(tab1_on_click, [state], [canvas, state, ml_info])
            ml_undo.click(tab1_undo, [state], [canvas, state, ml_info])
            ml_clear.click(tab1_clear, [state], [canvas, state, ml_info])
            ml_verify.click(tab1_verify, [state], [ml_result])
            ml_save.click(tab1_save, [state], [ml_result])

        with gr.Tab("2. Sync"):
            gr.Markdown("**Video:** click 2 points to define the LED ROI box, "
                        "then *Compute traces*.  **Mocap:** set the 3D box "
                        "(or *Auto-find LED*), then *Compute traces*, "
                        "*Auto-sync*, fine-tune, *Save sync*.")
            with gr.Row():
                with gr.Column():
                    video_view = gr.Image(type="numpy", height=480,
                                          interactive=False)
                    zoom_view = gr.Image(type="numpy", height=200,
                                         interactive=False,
                                         label="ROI zoom (box contents)")
                    video_scrub = gr.Slider(0, max(N_VID - 1, 1), value=0,
                                            step=1, label="Video scrub (frame)")
                    zoom_slider = gr.Slider(1, 8, value=1, step=0.5,
                                            label="Video zoom (crop around ROI)")
                    roi_status = gr.Textbox(value="Click 2 points on the video "
                                                  "to set ROI", lines=2,
                                            interactive=False)
                    with gr.Row():
                        reset_roi = gr.Button("Reset ROI")
                    gr.Markdown("### Fine-tune ROI corners (±5 px)")
                    with gr.Row():
                        for _corner, _clabel in [("tl", "TL"), ("tr", "TR"),
                                                 ("bl", "BL"), ("br", "BR")]:
                            with gr.Column():
                                gr.Markdown(f"**{_clabel}**")
                                for _axis in ["x", "y"]:
                                    with gr.Row():
                                        for _delta, _sym in [(-1, "−"), (1, "+")]:
                                            _b = gr.Button(f"{_axis}{_sym}")
                                            _b.click(
                                                lambda idx, s, c=_corner,
                                                       a=_axis, d=_delta:
                                                tab2_nudge(c, a, d, idx, s),
                                                [video_scrub, state],
                                                [state, roi_status,
                                                 video_view, zoom_view],
                                                show_progress="hidden")
                with gr.Column():
                    mocap3d = gr.Image(type="numpy", height=360,
                                       interactive=False)
                    mocap2d = gr.Image(type="numpy", height=280,
                                       interactive=False)
                    mocap_scrub = gr.Slider(0, max(N_MOC - 1, 1), value=0,
                                            step=1,
                                            label="Mocap scrub (frame, review)")
                    with gr.Row():
                        auto_led = gr.Button("Auto-find LED")
                    with gr.Row():
                        bx = gr.Slider(-3000, 1000, value=0, step=5, label="box X")
                        by = gr.Slider(-2000, 1000, value=0, step=5, label="box Y")
                        bz = gr.Slider(-1000, 1000, value=0, step=5, label="box Z")
                    bh = gr.Slider(10, 200, value=50, step=5, label="box half-size (mm)")

            gr.Markdown("### Seamless scrub (HTML5 video player)")
            gr.Markdown("This video runs entirely in your browser, so scrubbing "
                        "and playback have **no loading delay**. Use it to find "
                        "the LED blink, then set the ROI box on the still frame "
                        "above (the frame image still updates through the server "
                        "but without the loading spinner).")
            video_player = gr.Video(value=str(config.VIDEO_PATH), height=300,
                                    interactive=False,
                                    label="Raw video (instant scrub)")

            traces_plot = gr.Plot()
            with gr.Row():
                compute_traces = gr.Button("Compute traces", variant="primary")
                auto_sync = gr.Button("Auto-sync")
                save_sync_btn = gr.Button("Save sync")
            threshold = gr.Slider(0, 255, value=128, step=0.5,
                                  label="Video intensity threshold")
            offset = gr.Slider(-60, 60, value=0, step=0.01,
                               label="Offset fine-tune (s)")
            sync_scrub = gr.Slider(0, max(VIDEO_DUR, 1), value=0, step=0.01,
                                   label="Synchronized scrub (video seconds)")
            sync_info = gr.Textbox(label="Sync status", lines=3, interactive=False)

            video_scrub.change(tab2_render, [video_scrub, state],
                               [video_view, zoom_view], show_progress="hidden")
            zoom_slider.change(tab2_zoom, [zoom_slider, video_scrub, state],
                               [video_view, zoom_view], show_progress="hidden")
            video_view.select(tab2_video_click, [state, video_scrub],
                              [state, roi_status, video_view, zoom_view],
                              show_progress="hidden")
            reset_roi.click(tab2_reset_roi, [video_scrub, state],
                            [state, roi_status, video_view, zoom_view],
                            show_progress="hidden")
            mocap_scrub.change(tab2_show_mocap, [mocap_scrub, state],
                               [mocap3d, mocap2d])
            auto_led.click(tab2_auto_led, [state], [state, sync_info])
            bx.change(tab2_set_box, [bx, by, bz, bh, state], [state])
            by.change(tab2_set_box, [bx, by, bz, bh, state], [state])
            bz.change(tab2_set_box, [bx, by, bz, bh, state], [state])
            bh.change(tab2_set_box, [bx, by, bz, bh, state], [state])
            compute_traces.click(tab2_compute_traces, [state],
                                 [state, traces_plot, sync_info, sync_info])
            threshold.change(tab2_threshold, [threshold, state],
                             [state, traces_plot])
            auto_sync.click(tab2_auto_sync, [state], [state, traces_plot, sync_info])
            offset.change(tab2_offset, [offset, state], [state, traces_plot])
            save_sync_btn.click(tab2_save_sync, [state], [sync_info])
            sync_scrub.change(tab2_sync_scrub, [sync_scrub, state],
                              [video_scrub, mocap_scrub, video_view, zoom_view,
                               mocap3d, mocap2d], show_progress="hidden")

        with gr.Tab("3. Trim"):
            gr.Markdown("Set a shared start/end (in video seconds). Both streams "
                        "get the same duration.")
            start_s = gr.Slider(0, max(VIDEO_DUR, 1), value=0, step=0.01,
                                label="Trim start (s)")
            end_s = gr.Slider(0, max(VIDEO_DUR, 1), value=max(VIDEO_DUR, 1),
                              step=0.01, label="Trim end (s)")
            with gr.Row():
                trim_apply = gr.Button("Preview trim")
                trim_save = gr.Button("Save trim")
            trim_info = gr.Textbox(label="Trim summary", lines=6, interactive=False)
            trim_apply.click(tab3_apply, [start_s, end_s, state], [trim_info])
            trim_save.click(tab3_save, [start_s, end_s], [trim_info])

        with gr.Tab("4. Calibrate"):
            gr.Markdown("Runs solvePnP + Umeyama using saved intrinsics, "
                        "markers, sync and trim. Saves output/transform.json.")
            cal_run = gr.Button("Run calibration", variant="primary")
            cal_out = gr.Textbox(label="Calibration result", lines=14,
                                 interactive=False)
            cal_run.click(tab4_run, None, [cal_out])

        # --- Tab 0 data-loading wiring --------------------------------
        _data_outputs = [load_status, video_scrub, mocap_scrub, sync_scrub,
                         start_s, end_s, video_view, mocap3d, mocap2d,
                         video_player]
        load_btn.click(load_data, [vid_input, c3d_input], _data_outputs)
        # Auto-load on start so the tool works out of the box (uses the
        # configured / last-saved paths).
        demo.load(load_data, [vid_input, c3d_input], _data_outputs)

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name=config.GRADIO_HOST, server_port=config.GRADIO_PORT)
