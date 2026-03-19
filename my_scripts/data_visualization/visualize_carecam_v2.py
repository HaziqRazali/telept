#!/usr/bin/env python3
"""Visualize 2D keypoints from out.csv / out2.csv onto the carecam (iPad) video."""

import cv2
import csv
import re
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = "/home/haziq/datasets/telept/data/ipad/170326_18-04"
VIDEO_NAME = (
    "170326_18-04_306_68BFDC34_rgb_video_RGB_EBCCC581-ABF9-4D14-AC2C-"
    "C9D2FC6E6B39-5878-0000054A0829CB9A.mp4"
)
VIDEO_PATH = os.path.join(DATA_DIR, VIDEO_NAME)
OUT_CSV    = os.path.join(DATA_DIR, "out.csv")
OUT2_CSV   = os.path.join(DATA_DIR, "out2.csv")
SAVE_PATH  = os.path.join(DATA_DIR, VIDEO_NAME.replace(".mp4", "_results.mp4"))

# ── Drawing params ─────────────────────────────────────────────────────────────
CIRCLE_RADIUS = 12
CIRCLE_COLOR  = (0, 255, 0)   # BGR – green
CIRCLE_THICK  = -1            # filled circle
CONF_THRESH   = 0.3           # skip keypoints below this confidence


# ── Helper: parse numpy-style array string ─────────────────────────────────────
def parse_np_array_str(s: str, ncols: int = 3):
    """Turn a numpy repr like '[[x y c]\n [x y c]...]' into an ndarray."""
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if not nums:
        return None
    arr = np.array([float(v) for v in nums])
    if arr.size % ncols != 0:
        return None
    return arr.reshape(-1, ncols)


# ── Load video metadata from out2.csv ─────────────────────────────────────────
with open(OUT2_CSV, "r") as f:
    reader = csv.DictReader(f)
    meta = next(reader)

fps        = float(meta["fps"])
num_frames = int(meta["num_frames"])
vid_w      = int(meta["video_width"])
vid_h      = int(meta["video_height"])
print(f"[out2.csv] {vid_w}x{vid_h}  {fps:.3f} fps  {num_frames} frames")


# ── Load per-frame keypoints from out.csv ─────────────────────────────────────
# frame_num is 1-indexed in the CSV
frame_kps: dict[int, np.ndarray] = {}

with open(OUT_CSV, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fn  = int(row["frame_num"])
        raw = row.get("person1_pose2D_raw", "").strip()
        if not raw:
            continue
        kps = parse_np_array_str(raw, ncols=3)
        if kps is not None:
            frame_kps[fn] = kps

print(f"[out.csv]  Loaded keypoints for {len(frame_kps)} frames (frame_num 1–{max(frame_kps) if frame_kps else 0})")


# ── Open input video ───────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

actual_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps    = cap.get(cv2.CAP_PROP_FPS)
total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"[video]    {actual_w}x{actual_h}  {actual_fps:.3f} fps  {total_vid_frames} frames")
print(f"[keypoints] available for {len(frame_kps)} / {total_vid_frames} video frames")


# ── Setup writer ───────────────────────────────────────────────────────────────
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(SAVE_PATH, fourcc, actual_fps, (actual_w, actual_h))
if not writer.isOpened():
    raise RuntimeError(f"Cannot open VideoWriter for: {SAVE_PATH}")


# ── Process frames ─────────────────────────────────────────────────────────────
frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1          # 1-indexed to match CSV frame_num
    kps = frame_kps.get(frame_idx)

    if kps is not None:
        for kp in kps:
            x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
            if conf < CONF_THRESH:
                continue
            cv2.circle(frame, (int(round(x)), int(round(y))),
                       CIRCLE_RADIUS, CIRCLE_COLOR, CIRCLE_THICK)

    writer.write(frame)

    if frame_idx % 50 == 0:
        print(f"  frame {frame_idx:4d} / {total_vid_frames}")

cap.release()
writer.release()
print(f"\nSaved → {SAVE_PATH}")
