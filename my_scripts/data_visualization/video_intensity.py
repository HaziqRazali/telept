"""
Video grayscale viewer with frame slider and live mean intensity printing.
Usage:
    python video_intensity.py <video.mp4>

Controls:
    Slider   – scrub to any frame (prints mean intensity to terminal)
    ← / →    – step one frame at a time
    Space    – play / pause
    q        – quit
"""

import sys
import os
import argparse

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets


def load_video_frames(path):
    cap = cv2.VideoCapture(path)
    fps         = cap.get(cv2.CAP_PROP_FPS)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Loading: {path}")
    print(f"  Size   : {vid_w}x{vid_h}")
    print(f"  FPS    : {fps:.4f}")
    print(f"  Frames : {total}")
    print(f"  Duration: {total/fps:.3f} s")
    print("  Reading frames...", end="", flush=True)

    gray_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if len(gray_frames) % 50 == 0:
            print(".", end="", flush=True)
    cap.release()
    print(f" done ({len(gray_frames)} frames)")

    return np.stack(gray_frames, axis=0), fps   # [F, H, W]  uint8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", nargs="?",
                        default="/home/haziq/datasets/telept/data/mocap_recordings/LED1_cropped.mp4")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        sys.exit(f"[ERROR] File not found: {args.video}")

    frames, fps = load_video_frames(args.video)
    n_frames    = frames.shape[0]

    # Pre-compute mean intensity per frame (fast, avoids recalculation while scrubbing)
    mean_intensity = frames.reshape(n_frames, -1).mean(axis=1)   # [F]  float64

    # ── Build figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 8), facecolor="#1a1a1a")
    fig.canvas.manager.set_window_title("Grayscale Intensity Viewer")

    ax = fig.add_axes([0.02, 0.15, 0.96, 0.83])
    ax.set_facecolor("#1a1a1a")
    ax.axis("off")

    im_obj = ax.imshow(frames[0], cmap="gray", vmin=0, vmax=255,
                       interpolation="nearest", aspect="auto")
    title  = ax.set_title("", color="white", fontsize=10, pad=4)

    # Frame slider
    ax_slider = fig.add_axes([0.10, 0.06, 0.78, 0.03], facecolor="#2a2a2a")
    slider = mwidgets.Slider(
        ax_slider, "Frame", 0, n_frames - 1,
        valinit=0, valstep=1, color="#5588ff"
    )
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    # Time text below slider
    ax_time = fig.add_axes([0.10, 0.01, 0.78, 0.04])
    ax_time.axis("off")
    time_text = ax_time.text(
        0.5, 0.5, "", ha="center", va="center",
        color="#aaaaaa", fontsize=9, transform=ax_time.transAxes
    )

    # Play/pause button
    ax_btn = fig.add_axes([0.01, 0.04, 0.07, 0.05], facecolor="#2a2a2a")
    btn = mwidgets.Button(ax_btn, "▶ Play", color="#2a2a2a", hovercolor="#3a3a3a")
    btn.label.set_color("white")

    state = {"frame": 0, "playing": False, "timer": None, "last_printed": -1}

    def show_frame(idx):
        idx = int(np.clip(idx, 0, n_frames - 1))
        im_obj.set_data(frames[idx])
        ms     = idx / fps * 1000
        t_tot  = (n_frames - 1) / fps * 1000
        mi     = mean_intensity[idx]
        title.set_text(
            f"Frame {idx}/{n_frames-1}   "
            f"t = {ms:,.1f} ms   "
            f"Mean intensity = {mi:.2f}"
        )
        time_text.set_text(
            f"t = {ms:,.1f} ms  /  {t_tot:,.1f} ms"
        )
        fig.canvas.draw_idle()

        # Print to terminal only when frame actually changes
        if idx != state["last_printed"]:
            print(f"Frame {idx:5d}  |  t = {ms:8.1f} ms  |  mean intensity = {mi:.4f}")
            state["last_printed"] = idx

    show_frame(0)

    def on_slider(val):
        f = int(slider.val)
        if f != state["frame"]:
            state["frame"] = f
            show_frame(f)

    slider.on_changed(on_slider)

    def step(delta):
        f = int(np.clip(state["frame"] + delta, 0, n_frames - 1))
        state["frame"] = f
        slider.set_val(f)   # triggers on_slider -> show_frame

    def on_key(event):
        if event.key == "right":
            step(1)
        elif event.key == "left":
            step(-1)
        elif event.key == "shift+right":
            step(10)
        elif event.key == "shift+left":
            step(-10)
        elif event.key == " ":
            toggle_play(None)
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    interval_ms = max(1, int(1000 / fps))

    def tick(event=None):
        if not state["playing"]:
            return
        nxt = state["frame"] + 1
        if nxt >= n_frames:
            nxt = 0
        state["frame"] = nxt
        slider.set_val(nxt)

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

    plt.show()


if __name__ == "__main__":
    main()
