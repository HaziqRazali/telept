"""
Review a recorded IR + RGB session side-by-side.

Usage:
    python review_recording.py /home/haziq/datasets/telept/data/mocap/20141017/take1/ir_20260402_175610
    python review_recording.py ir_20260402_175610 --fps 15

Controls:
    Space / Right arrow  : next frame
    Left arrow           : previous frame
    f                    : jump forward 10 frames
    b                    : jump back 10 frames
    g                    : go to frame number (type in terminal)
    p                    : toggle auto-play
    +/-                  : increase/decrease playback speed
    q                    : quit
"""

import sys
import os
import glob
import argparse
import cv2
import numpy as np


# ── display constants ──────────────────────────────────────────────────────────
IR_DISPLAY_SIZE  = (512, 512)   # IR panel size in the side-by-side window
RGB_DISPLAY_SIZE = (768, 432)   # RGB panel size (16:9 of ~768)
PANEL_HEIGHT     = 512          # unified height for both panels (RGB letterboxed)


def load_ir(path: str) -> np.ndarray:
    """Load a 16-bit IR frame and convert to 8-bit BGR for display."""
    ir16 = np.load(path)
    ir8  = cv2.normalize(ir16, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)


def make_side_by_side(ir_bgr: np.ndarray, rgb_bgr: np.ndarray,
                      frame_idx: int, total: int, fps_label: str) -> np.ndarray:
    # Resize IR
    ir_panel = cv2.resize(ir_bgr, IR_DISPLAY_SIZE)
    # Resize RGB maintaining aspect ratio, pad to PANEL_HEIGHT
    rgb_panel = cv2.resize(rgb_bgr, RGB_DISPLAY_SIZE)
    pad = PANEL_HEIGHT - RGB_DISPLAY_SIZE[1]
    top, bot = pad // 2, pad - pad // 2
    rgb_panel = cv2.copyMakeBorder(rgb_panel, top, bot, 0, 0,
                                   cv2.BORDER_CONSTANT, value=(30, 30, 30))

    # Labels
    cv2.putText(ir_panel,  "Active IR", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(rgb_panel, "RGB",        (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0),   2)

    # Status bar
    bar = np.zeros((36, IR_DISPLAY_SIZE[0] + RGB_DISPLAY_SIZE[0], 3), dtype=np.uint8)
    cv2.putText(bar, f"Frame {frame_idx+1}/{total}   {fps_label}   "
                     f"[Space/Arrow=step  p=play  +/-=speed  q=quit]",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    canvas = np.vstack([np.hstack([ir_panel, rgb_panel]), bar])
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ir_dir", help="IR recording directory, e.g. ir_20260402_171419")
    ap.add_argument("--fps", type=float, default=15.0, help="Auto-play FPS (default 15)")
    args = ap.parse_args()

    ir_dir = args.ir_dir
    if not os.path.isdir(ir_dir):
        # Try relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ir_dir = os.path.join(script_dir, ir_dir)

    npy_files = sorted(glob.glob(os.path.join(ir_dir, "frame_*.npy")))
    if not npy_files:
        print(f"No .npy files found in {ir_dir}")
        sys.exit(1)

    ts       = os.path.basename(ir_dir)[3:]   # strip "ir_"
    rgb_path = os.path.join(os.path.dirname(ir_dir), f"rgb_{ts}.avi")
    if not os.path.exists(rgb_path):
        print(f"RGB video not found: {rgb_path}")
        sys.exit(1)

    # Load sync index if present (maps IR frame i -> RGB frame number)
    sync_path = os.path.join(ir_dir, "sync_index.npy")
    if os.path.exists(sync_path):
        sync_index = np.load(sync_path)
        print(f"Sync index loaded: {len(sync_index)} IR frames -> RGB frame mapping")
    else:
        sync_index = None
        print("No sync_index.npy found — assuming 1:1 IR/RGB frame mapping")

    cap    = cv2.VideoCapture(rgb_path)
    total  = len(npy_files)
    n_rgb  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"IR  frames : {total}  ({ir_dir})")
    print(f"RGB frames : {n_rgb}  ({rgb_path})")
    if sync_index is not None:
        # Dual-rate recording: RGB has ~2x IR frames — that's expected
        if n_rgb < total:
            print(f"WARNING: RGB has fewer frames than IR ({n_rgb} < {total}), clamping")
            total = n_rgb
    elif total != n_rgb:
        print(f"WARNING: frame count mismatch ({total} vs {n_rgb}) — will use min({total},{n_rgb})")
        total = min(total, n_rgb)

    idx      = 0
    playing  = False
    play_fps = args.fps
    delay_ms = max(1, int(1000 / play_fps))

    def read_pair(i):
        ir_bgr = load_ir(npy_files[i])
        rgb_frame_no = int(sync_index[i]) if sync_index is not None else i
        cap.set(cv2.CAP_PROP_POS_FRAMES, rgb_frame_no)
        ok, rgb = cap.read()
        if not ok:
            rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
        return ir_bgr, rgb, rgb_frame_no

    WIN = "IR + RGB Review"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, IR_DISPLAY_SIZE[0] + RGB_DISPLAY_SIZE[0], PANEL_HEIGHT + 36)
    cv2.createTrackbar("Frame", WIN, 0, total - 1, lambda x: None)

    ir_bgr, rgb_bgr, cur_rgb_no = read_pair(idx)

    while True:
        rgb_info = f"RGB#{cur_rgb_no}" if sync_index is not None else ""
        fps_label = f"{'PLAY' if playing else 'PAUSE'}  {play_fps:.0f}fps  {rgb_info}"
        frame = make_side_by_side(ir_bgr, rgb_bgr, idx, total, fps_label)
        cv2.imshow(WIN, frame)

        wait = delay_ms if playing else 30   # never block forever so trackbar is polled
        key  = cv2.waitKey(wait) & 0xFF

        # ── trackbar scrub (user dragged the slider) ───────────────────────
        tb_idx = cv2.getTrackbarPos("Frame", WIN)
        if tb_idx != idx:
            idx = tb_idx
            ir_bgr, rgb_bgr, cur_rgb_no = read_pair(idx)

        if key == ord('q'):
            break
        elif key == ord('p'):
            playing = not playing
        elif key == ord('+') or key == ord('='):
            play_fps = min(play_fps + 5, 60)
            delay_ms = max(1, int(1000 / play_fps))
        elif key == ord('-'):
            play_fps = max(play_fps - 5, 1)
            delay_ms = max(1, int(1000 / play_fps))
        elif key in (ord(' '), 83, 0xFF & ord('d')):   # Space / Right / d
            idx = min(idx + 1, total - 1)
            ir_bgr, rgb_bgr, cur_rgb_no = read_pair(idx)
            cv2.setTrackbarPos("Frame", WIN, idx)
        elif key in (81, 0xFF & ord('a')):             # Left / a
            idx = max(idx - 1, 0)
            ir_bgr, rgb_bgr, cur_rgb_no = read_pair(idx)
            cv2.setTrackbarPos("Frame", WIN, idx)
        elif key == ord('f'):
            idx = min(idx + 10, total - 1)
            ir_bgr, rgb_bgr, cur_rgb_no = read_pair(idx)
            cv2.setTrackbarPos("Frame", WIN, idx)
        elif key == ord('b'):
            idx = max(idx - 10, 0)
            ir_bgr, rgb_bgr, cur_rgb_no = read_pair(idx)
            cv2.setTrackbarPos("Frame", WIN, idx)
        elif key == ord('g'):
            n = input(f"  Go to frame [0-{total-1}]: ").strip()
            if n.isdigit():
                idx = max(0, min(int(n), total - 1))
            ir_bgr, rgb_bgr, cur_rgb_no = read_pair(idx)
            cv2.setTrackbarPos("Frame", WIN, idx)
        elif playing:
            # Auto-advance
            if idx < total - 1:
                idx += 1
                ir_bgr, rgb_bgr, cur_rgb_no = read_pair(idx)
                cv2.setTrackbarPos("Frame", WIN, idx)
            else:
                playing = False  # end of recording

    cap.release()
    cv2.destroyAllWindows(WIN)


if __name__ == "__main__":
    main()
