"""
Interactive C3D marker viewer with frame scrubber.
Usage:
    python view_c3d.py [path/to/file.c3d]

Controls:
    Slider   – scrub to any frame
    ← / →    – step one frame at a time
    Space     – play / pause
"""

import sys
import os
import argparse

import numpy as np
import ezc3d
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" if TkAgg not available
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Default file
# ---------------------------------------------------------------------------
DEFAULT_C3D = (
    "/home/haziq/datasets/telept/data/mocap_recordings/"
    "Take 2014-10-01 07.50.08 AM.c3d"
)


def load_c3d(path):
    print(f"Loading: {path}")
    c = ezc3d.c3d(path)

    # points: shape [4, n_markers, n_frames]  (X, Y, Z, residual)
    pts = c["data"]["points"]          # float64
    xyz = pts[:3]                      # [3, M, F]
    res = pts[3]                       # [M, F]  residual; <0 = invalid

    frame_rate  = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    units       = c["parameters"]["POINT"]["UNITS"]["value"][0]
    labels_raw  = c["parameters"]["POINT"]["LABELS"]["value"]

    # Collect all LABELS* blocks in case markers spill over
    label_keys = sorted(
        k for k in c["parameters"]["POINT"]
        if k.startswith("LABELS")
    )
    all_labels = []
    for k in label_keys:
        all_labels.extend(c["parameters"]["POINT"][k]["value"])
    all_labels = all_labels[: xyz.shape[1]]   # trim to actual marker count

    print(f"  Markers : {xyz.shape[1]}")
    print(f"  Frames  : {xyz.shape[2]}")
    print(f"  Rate    : {frame_rate} fps")
    print(f"  Units   : {units}")
    print(f"  Duration: {xyz.shape[2]/frame_rate:.1f} s")

    return xyz, res, all_labels, frame_rate, units


def get_valid(xyz, res, frame_idx):
    """Return [N_valid, 3] for markers that have a non-negative residual."""
    r = res[:, frame_idx]
    x = xyz[:, :, frame_idx].T   # [M, 3]
    mask = r >= 0
    return x[mask], mask


def compute_scene_bounds(xyz, res):
    """Compute axis limits from all finite, valid points."""
    valid = xyz[:, res >= 0]     # [3, K]  broadcast-friendly
    if valid.size == 0:
        return np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=float)
    lo = np.nanpercentile(valid, 1, axis=1)
    hi = np.nanpercentile(valid, 99, axis=1)
    pad = np.maximum((hi - lo) * 0.1, 0.1)
    return np.stack([lo - pad, hi + pad], axis=1)   # [3, 2]


def interpolate_markers(xyz, res, labels, targets=("HandMark1",)):
    """
    Linearly interpolate missing marker positions (res < 0) only for markers
    whose label is in `targets`.
    Returns:
        xyz_interp : [3, M, F]  – same as xyz but gaps filled for target markers
        was_missing: [M, F] bool – True where the original frame was invalid
    """
    M, F = xyz.shape[1], xyz.shape[2]
    xyz_interp = xyz.copy().astype(float)
    was_missing = res < 0  # [M, F]

    # Set all originally-invalid frames to NaN so they don't bleed into interp
    xyz_interp[:, was_missing] = np.nan

    frames = np.arange(F, dtype=float)
    filled = 0
    for m, label in enumerate(labels):
        if label not in targets and not any(label.startswith(t) for t in targets):
            continue  # skip markers not in the target list

        # A frame is "usable" only when residual is valid AND all coords are finite
        finite_mask = np.all(np.isfinite(xyz_interp[:, m, :]), axis=0)  # [F]
        usable_mask = (~was_missing[m]) & finite_mask
        valid_frames = frames[usable_mask]

        if valid_frames.size == 0:
            print(f"  [{label}] no valid frames — cannot interpolate")
            continue
        elif valid_frames.size == 1:
            for axis in range(3):
                xyz_interp[axis, m, was_missing[m]] = xyz_interp[axis, m, usable_mask][0]
            filled += int(was_missing[m].sum())
            print(f"  [{label}] filled {int(was_missing[m].sum())} frames (constant fill)")
        else:
            n_missing = int(was_missing[m].sum())
            for axis in range(3):
                valid_vals = xyz_interp[axis, m, usable_mask]
                xyz_interp[axis, m] = np.interp(frames, valid_frames, valid_vals)
            filled += n_missing
            print(f"  [{label}] interpolated {n_missing} missing frames")

    return xyz_interp, was_missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("c3d_file", nargs="?", default=DEFAULT_C3D)
    parser.add_argument("--fps_play", type=float, default=0,
                        help="Playback FPS (0 = use file rate)")
    args = parser.parse_args()

    if not os.path.isfile(args.c3d_file):
        sys.exit(f"[ERROR] File not found: {args.c3d_file}")

    xyz, res, labels, frame_rate, units = load_c3d(args.c3d_file)
    xyz_interp, was_missing = interpolate_markers(xyz, res, labels, targets=("HandMark1",))
    n_frames = xyz.shape[2]
    play_fps = args.fps_play if args.fps_play > 0 else frame_rate

    # Pre-compute bounds from the first 500 frames (fast)
    sample = min(500, n_frames)
    bounds = compute_scene_bounds(
        xyz[:, :, :sample].reshape(3, -1),
        res[:, :sample].reshape(-1)
    )

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(11, 9), facecolor="#1a1a1a")
    fig.canvas.manager.set_window_title("C3D Viewer")

    # 3D axis (leave room at bottom for widgets)
    ax = fig.add_axes([0.02, 0.15, 0.96, 0.83], projection="3d")
    ax.set_facecolor("#1a1a1a")
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor("#444444")
    ax.grid(True, color="#333333", linewidth=0.5)
    ax.tick_params(colors="#888888", labelsize=7)
    ax.xaxis.label.set_color("#888888")
    ax.yaxis.label.set_color("#888888")
    ax.zaxis.label.set_color("#888888")

    ax.set_xlabel(f"X ({units})", fontsize=8)
    ax.set_ylabel(f"Y ({units})", fontsize=8)
    ax.set_zlabel(f"Z ({units})", fontsize=8)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.set_zlim(bounds[2])

    pts_valid, _ = get_valid(xyz, res, 0)
    scatter = ax.scatter(
        pts_valid[:, 0], pts_valid[:, 1], pts_valid[:, 2],
        c="#ffee00", s=6, alpha=0.85, depthshade=False, linewidths=0
    )

    # Scatter for interpolated (gap-filled) points – cyan, initially hidden
    scatter_interp = ax.scatter(
        [], [], [],
        c="#00ccff", s=6, alpha=0.6, depthshade=False, linewidths=0,
        visible=False
    )

    # Create one Text3D object per marker (hidden by default until toggled)
    label_texts = []
    for lbl in labels:
        t = ax.text(0, 0, 0, lbl, fontsize=9, color="#aaffaa",
                    ha="left", va="bottom", visible=False)
        label_texts.append(t)

    title = ax.set_title("", color="white", fontsize=10, pad=6)

    def update_title(frame_idx):
        t = frame_idx / frame_rate
        title.set_text(
            f"Frame {frame_idx}/{n_frames-1}   "
            f"t = {t:.3f} s   "
            f"valid = {len(get_valid(xyz, res, frame_idx)[0])}/{xyz.shape[1]}"
        )

    update_title(0)

    # ------------------------------------------------------------------
    # Frame slider
    # ------------------------------------------------------------------
    ax_slider = fig.add_axes([0.12, 0.06, 0.76, 0.03], facecolor="#2a2a2a")
    slider = mwidgets.Slider(
        ax_slider, "Frame", 0, n_frames - 1,
        valinit=0, valstep=1, color="#5588ff"
    )
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    # Time text below slider
    ax_time = fig.add_axes([0.12, 0.02, 0.76, 0.03])
    ax_time.axis("off")
    time_text = ax_time.text(
        0.5, 0.5, "", ha="center", va="center",
        color="#aaaaaa", fontsize=9,
        transform=ax_time.transAxes
    )

    # Play/pause button
    ax_btn = fig.add_axes([0.01, 0.04, 0.07, 0.05], facecolor="#2a2a2a")
    btn = mwidgets.Button(ax_btn, "▶ Play", color="#2a2a2a", hovercolor="#3a3a3a")
    btn.label.set_color("white")

    # Labels toggle button
    ax_lbl_btn = fig.add_axes([0.80, 0.04, 0.09, 0.05], facecolor="#2a2a2a")
    lbl_btn = mwidgets.Button(ax_lbl_btn, "Labels: OFF", color="#2a2a2a", hovercolor="#3a3a3a")
    lbl_btn.label.set_color("white")

    # Interpolation toggle button
    ax_interp_btn = fig.add_axes([0.90, 0.04, 0.09, 0.05], facecolor="#2a2a2a")
    interp_btn = mwidgets.Button(ax_interp_btn, "Interp: OFF", color="#2a2a2a", hovercolor="#3a3a3a")
    interp_btn.label.set_color("#00ccff")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    state = {"frame": 0, "playing": False, "timer": None,
             "show_labels": False, "show_interp": False}

    def update_label_positions(frame_idx):
        """Move each label text to its marker's current position (or hide if invalid)."""
        r = res[:, frame_idx]
        # Use interpolated xyz when interp is enabled so labels track filled gaps
        src = xyz_interp if state["show_interp"] else xyz
        x = src[:, :, frame_idx].T   # [M, 3]
        for i, t in enumerate(label_texts):
            visible_pos = (r[i] >= 0) or (state["show_interp"] and was_missing[i, frame_idx]
                                           and np.all(np.isfinite(x[i])))
            if visible_pos and state["show_labels"]:
                t.set_position((x[i, 0], x[i, 1]))
                t.set_3d_properties(x[i, 2], zdir="z")
                t.set_visible(True)
            else:
                t.set_visible(False)

    def draw_frame(frame_idx):
        pts, _ = get_valid(xyz, res, frame_idx)
        if pts.shape[0] > 0:
            scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])

        # Interpolated (gap-filled) points
        if state["show_interp"]:
            missing_mask = was_missing[:, frame_idx]
            pts_i = xyz_interp[:, missing_mask, frame_idx].T
            # Only show where interpolation produced finite values
            finite = np.all(np.isfinite(pts_i), axis=1)
            pts_i = pts_i[finite]
            if pts_i.shape[0] > 0:
                scatter_interp._offsets3d = (pts_i[:, 0], pts_i[:, 1], pts_i[:, 2])
                scatter_interp.set_visible(True)
            else:
                scatter_interp._offsets3d = ([], [], [])
                scatter_interp.set_visible(False)
        else:
            scatter_interp._offsets3d = ([], [], [])
            scatter_interp.set_visible(False)

        update_label_positions(frame_idx)
        update_title(frame_idx)
        time_text.set_text(
            f"t = {frame_idx / frame_rate:.3f} s  /  "
            f"{(n_frames-1) / frame_rate:.3f} s"
        )
        fig.canvas.draw_idle()

    def on_slider(val):
        f = int(slider.val)
        if f != state["frame"]:
            state["frame"] = f
            draw_frame(f)

    slider.on_changed(on_slider)

    def step(delta):
        f = int(np.clip(state["frame"] + delta, 0, n_frames - 1))
        state["frame"] = f
        slider.set_val(f)          # triggers on_slider → draw_frame

    def on_key(event):
        if event.key == "right":
            step(1)
        elif event.key == "left":
            step(-1)
        elif event.key == "right_shift" or event.key == "shift+right":
            step(10)
        elif event.key == "left_shift" or event.key == "shift+left":
            step(-10)
        elif event.key == " ":
            toggle_play(None)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # ------------------------------------------------------------------
    # Playback timer
    # ------------------------------------------------------------------
    interval_ms = max(1, int(1000 / play_fps))

    def tick(event=None):
        if not state["playing"]:
            return
        next_f = state["frame"] + 1
        if next_f >= n_frames:
            next_f = 0          # loop
        state["frame"] = next_f
        slider.set_val(next_f)

    def toggle_play(event):
        state["playing"] = not state["playing"]
        if state["playing"]:
            btn.label.set_text("⏸ Pause")
            state["timer"] = fig.canvas.new_timer(interval=interval_ms)
            state["timer"].add_callback(tick)
            state["timer"].start()
        else:
            btn.label.set_text("▶ Play")
            if state["timer"]:
                state["timer"].stop()
        fig.canvas.draw_idle()

    btn.on_clicked(toggle_play)

    def toggle_labels(event):
        state["show_labels"] = not state["show_labels"]
        lbl_btn.label.set_text("Labels: ON" if state["show_labels"] else "Labels: OFF")
        draw_frame(state["frame"])

    lbl_btn.on_clicked(toggle_labels)

    def toggle_interp(event):
        state["show_interp"] = not state["show_interp"]
        interp_btn.label.set_text("Interp: ON" if state["show_interp"] else "Interp: OFF")
        draw_frame(state["frame"])

    interp_btn.on_clicked(toggle_interp)

    # draw initial frame
    draw_frame(0)

    plt.show()


if __name__ == "__main__":
    main()
