"""Gradio UI for the iPad<->Mocap calibration tool.

Run:  python3 app.py          (then open http://localhost:7860)

Stages:
  Tab 1  Marker layout   - click where the 6 reflective markers sit on the board
  Tab 2  Sync            - LED blink: ROI + threshold (video), 3D box (mocap),
                           cross-correlation offset + fine-tune
  Tab 3  Trim            - shared start/end trim for both streams
  Tab 4  Calibrate       - solvePnP + Umeyama -> mocap->camera transform
"""

from __future__ import annotations

import json

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
# Load data once (cached; fast after first run)
# ----------------------------------------------------------------------
FRAMES, TIMES = pre_extract_frames(config.VIDEO_PATH)
MOCAP = load_c3d(config.C3D_PATH)
N_VID = len(FRAMES)
N_MOC = MOCAP["n_frames"]
FPS_M = MOCAP["fps"]
IMG_H, IMG_W = 1280, 720
VIDEO_DUR = float(TIMES[-1])

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
}


def _frame_bgr(i: int) -> np.ndarray:
    return cv2.imread(str(FRAMES[i]))


# ----------------------------------------------------------------------
# Tab 1 - Marker layout
# ----------------------------------------------------------------------
def tab1_render_canvas(placed) -> np.ndarray:
    return ml.render_canvas(placed)


def tab1_on_click(evt: gr.SelectData, state):
    x_px, y_px = evt.index
    x_mm, y_mm = ml.px_to_mm(x_px, y_px)
    x_mm, y_mm = ml.snap_to_grid(x_mm, y_mm)
    if len(state["placed"]) >= config.NUM_MARKERS:
        return (ml.render_canvas(state["placed"]), state,
                f"Already have {config.NUM_MARKERS} markers - use Undo to remove one.")
    state["placed"].append([x_mm, y_mm])
    named = [(f"{i+1}", p[0], p[1]) for i, p in enumerate(state["placed"])]
    msg = "Placed: " + ", ".join(f"{i+1}@({x:.0f},{y:.0f})mm"
                                 for i, (x, y) in enumerate(state["placed"]))
    return ml.render_canvas(named), state, msg


def tab1_undo(state):
    if state["placed"]:
        state["placed"].pop()
    named = [(f"{i+1}", p[0], p[1]) for i, p in enumerate(state["placed"])]
    return ml.render_canvas(named), state, f"{len(state['placed'])} markers placed"


def tab1_clear(state):
    state["placed"] = []
    return ml.render_canvas([]), state, "cleared"


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
# Tab 2 - Sync
# ----------------------------------------------------------------------
def _draw_roi_on_frame(idx: int, roi) -> np.ndarray:
    img = _frame_bgr(idx)
    if roi is not None:
        x0, y0, x1, y1 = roi
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(img, "LED ROI", (x0, max(y0 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return img


def tab2_show_video(idx, state):
    img = _draw_roi_on_frame(int(idx), state["video_roi"])
    return img


def tab2_show_zoom(idx, state):
    roi = state["video_roi"]
    if roi is None:
        return None
    x0, y0, x1, y1 = roi
    img = _frame_bgr(int(idx))[y0:y1, x0:x1]
    return img


def tab2_video_click(evt: gr.SelectData, state, idx):
    x, y = evt.index
    if state["roi_click"] is None:
        state["roi_click"] = (x, y)
        img = _draw_roi_on_frame(int(idx), state["video_roi"])
        return (state, "ROI: click opposite corner", img)
    (x0, y0), (x1, y1) = state["roi_click"], (x, y)
    state["video_roi"] = [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]
    state["roi_click"] = None
    img = _draw_roi_on_frame(int(idx), state["video_roi"])
    return state, "ROI set. Press 'Compute traces'.", img


def tab2_reset_roi(state):
    state["video_roi"] = None
    state["roi_click"] = None
    return state, "ROI cleared"


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
    trace, times = compute_video_trace(tuple(state["video_roi"]))
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
    img_v = _draw_roi_on_frame(vi, state["video_roi"])
    box = _box_arrays(state)
    img3 = render_mocap_3d(MOCAP, mi, box)
    img2 = render_mocap_2d(MOCAP, mi, "top", box)
    return vi, mi, img_v, img3, img2


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
        r = run_calibration()
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
def _on_load():
    img_v = _draw_roi_on_frame(0, None)
    img3 = render_mocap_3d(MOCAP, 0, None)
    img2 = render_mocap_2d(MOCAP, 0, "top", None)
    return img_v, img3, img2


def build_app():
    with gr.Blocks(title="iPad <-> Mocap Calibration") as demo:
        state = gr.State(DEFAULT_STATE.copy())

        with gr.Tab("1. Marker layout"):
            gr.Markdown("Click on the **orange dots** where the reflective "
                        "markers sit (grid = 40 mm). Click 6 in total.")
            with gr.Row():
                canvas = gr.Image(value=ml.render_canvas([]), type="numpy",
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
                    zoom_view = gr.Image(type="numpy", height=180,
                                         interactive=False, label="ROI zoom")
                    video_scrub = gr.Slider(0, N_VID - 1, value=0, step=1,
                                            label="Video scrub (frame)")
                    roi_status = gr.Textbox(value="Click 2 points on the video "
                                                  "to set ROI", lines=2,
                                            interactive=False)
                    with gr.Row():
                        reset_roi = gr.Button("Reset ROI")
                with gr.Column():
                    mocap3d = gr.Image(type="numpy", height=360,
                                       interactive=False)
                    mocap2d = gr.Image(type="numpy", height=280,
                                       interactive=False)
                    mocap_scrub = gr.Slider(0, N_MOC - 1, value=0, step=1,
                                            label="Mocap scrub (frame, review)")
                    with gr.Row():
                        auto_led = gr.Button("Auto-find LED")
                    with gr.Row():
                        bx = gr.Slider(-3000, 1000, value=0, step=5, label="box X")
                        by = gr.Slider(-2000, 1000, value=0, step=5, label="box Y")
                        bz = gr.Slider(-1000, 1000, value=0, step=5, label="box Z")
                    bh = gr.Slider(10, 200, value=50, step=5, label="box half-size (mm)")

            traces_plot = gr.Plot()
            with gr.Row():
                compute_traces = gr.Button("Compute traces", variant="primary")
                auto_sync = gr.Button("Auto-sync")
                save_sync_btn = gr.Button("Save sync")
            threshold = gr.Slider(0, 255, value=128, step=0.5,
                                  label="Video intensity threshold")
            offset = gr.Slider(-60, 60, value=0, step=0.01,
                               label="Offset fine-tune (s)")
            sync_scrub = gr.Slider(0, VIDEO_DUR, value=0, step=0.01,
                                   label="Synchronized scrub (video seconds)")
            sync_info = gr.Textbox(label="Sync status", lines=3, interactive=False)

            video_scrub.change(tab2_show_video, [video_scrub, state], [video_view])
            video_scrub.change(tab2_show_zoom, [video_scrub, state], [zoom_view])
            video_view.select(tab2_video_click, [state, video_scrub],
                              [state, roi_status, video_view])
            reset_roi.click(tab2_reset_roi, [state], [state, roi_status])
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
                              [video_scrub, mocap_scrub, video_view, mocap3d, mocap2d])

        demo.load(_on_load, None, [video_view, mocap3d, mocap2d])

        with gr.Tab("3. Trim"):
            gr.Markdown("Set a shared start/end (in video seconds). Both streams "
                        "get the same duration.")
            start_s = gr.Slider(0, VIDEO_DUR, value=0, step=0.01,
                                label="Trim start (s)")
            end_s = gr.Slider(0, VIDEO_DUR, value=VIDEO_DUR, step=0.01,
                              label="Trim end (s)")
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

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name=config.GRADIO_HOST, server_port=config.GRADIO_PORT)
