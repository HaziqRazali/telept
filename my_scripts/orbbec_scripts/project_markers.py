"""
Project HandMark1 3D markers onto the video using solvePnP.

Steps:
  1. Load frame 0 of the video and C3D
  2. Brute-force all 6! permutations of 2D<->3D correspondence
  3. Pick the permutation with lowest reprojection error
  4. Project HandMark1 markers onto every frame of the video
  5. Save as a new annotated video

Usage:
    conda activate orbbec
    python project_markers.py
"""

import cv2
import numpy as np
import ezc3d
from itertools import permutations
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
VIDEO_PATH  = "/data/telept/my_scripts/orbbec_scripts/recording_20260318_161152.mp4"
C3D_PATH    = "/data/telept/my_scripts/orbbec_scripts/Take 2014-10-02 05.49.50 AM.c3d"
ANNOT_TXT   = "/data/telept/my_scripts/orbbec_scripts/recording_20260318_161152_circle.txt"
OUT_VIDEO   = os.path.splitext(VIDEO_PATH)[0] + "_projected.mp4"

# ── Camera intrinsics (Femto Bolt 1920x1080) ──────────────────────────────────
K = np.array([
    [1123.86669921875, 0.0,              948.0269165039062],
    [0.0,              1123.028076171875, 539.6485595703125],
    [0.0,              0.0,              1.0              ],
], dtype=np.float64)

dist_coeffs = np.array([
    0.07333821058273315,
   -0.10178927332162857,
   -0.0004722462617792189,
   -0.00022512981377076358,
    0.041689008474349976,
], dtype=np.float64)  # k1, k2, p1, p2, k3  (OpenCV order)

# ── Load 2D annotated points ──────────────────────────────────────────────────
pts2d = []
with open(ANNOT_TXT) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # format: "1: x=779, y=471"
        parts = line.split(":")[-1].split(",")
        x = int(parts[0].split("=")[1])
        y = int(parts[1].split("=")[1])
        pts2d.append((x, y))
pts2d = np.array(pts2d, dtype=np.float64)  # (6, 2)
print(f"Loaded {len(pts2d)} 2D points from annotation file.")

# ── Load 3D HandMark1 points from C3D (frame 0) ───────────────────────────────
c = ezc3d.c3d(C3D_PATH)
labels  = c["parameters"]["POINT"]["LABELS"]["value"]
points  = c["data"]["points"]  # (4, n_markers, n_frames)

hand_indices = [i for i, l in enumerate(labels) if "HandMark1" in l]
hand_labels  = [labels[i] for i in hand_indices]
print(f"HandMark1 markers: {hand_labels}")

# Frame 0, XYZ in metres
pts3d = points[:3, hand_indices, 0].T.astype(np.float64)  # (6, 3)
print("3D points (frame 0):")
for i, (lbl, pt) in enumerate(zip(hand_labels, pts3d)):
    print(f"  [{i}] {lbl}: {pt}")

# ── Brute-force best permutation via solvePnP ─────────────────────────────────
print("\nSearching over 6! = 720 permutations for best 2D<->3D correspondence...")

best_error = np.inf
best_perm  = None
best_rvec  = None
best_tvec  = None

for perm in permutations(range(6)):
    pts3d_perm = pts3d[list(perm)]
    ok, rvec, tvec = cv2.solvePnP(
        pts3d_perm, pts2d, K, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        continue
    proj, _ = cv2.projectPoints(pts3d_perm, rvec, tvec, K, dist_coeffs)
    proj = proj.reshape(-1, 2)
    err = np.mean(np.linalg.norm(proj - pts2d, axis=1))
    if err < best_error:
        best_error = err
        best_perm  = perm
        best_rvec  = rvec.copy()
        best_tvec  = tvec.copy()

print(f"Best permutation: {best_perm}")
print(f"Best reprojection error: {best_error:.3f} px")
print(f"Mapping (3D marker -> 2D click):")
for click_idx, marker_idx in enumerate(best_perm):
    print(f"  click {click_idx+1} ({pts2d[click_idx]}) -> {hand_labels[marker_idx]} ({pts3d[marker_idx]})")

# ── Camera pose in mocap world ────────────────────────────────────────────────
R, _ = cv2.Rodrigues(best_rvec)
# Camera center in world coords
cam_pos_world = -R.T @ best_tvec
print(f"\nCamera position in mocap world: {cam_pos_world.T}")

# ── Project all HandMark1 points onto every video frame ───────────────────────
pts3d_ordered = pts3d[list(best_perm)]  # reordered to match 2D clicks

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps_vid = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps_vid, (W, H))

colors_markers = [
    (0,   255, 0),
    (255, 128, 0),
    (0,   128, 255),
    (255, 0,   255),
    (0,   255, 255),
    (255, 255, 0),
]

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Project 3D HandMark1 points (static — same pose every frame)
    proj, _ = cv2.projectPoints(pts3d, best_rvec, best_tvec, K, dist_coeffs)
    proj = proj.reshape(-1, 2)

    for i, (px, py) in enumerate(proj):
        px, py = int(round(px)), int(round(py))
        if 0 <= px < W and 0 <= py < H:
            col = colors_markers[i % len(colors_markers)]
            cv2.circle(frame, (px, py), 10, col, 2)
            cv2.circle(frame, (px, py), 3, col, -1)
            cv2.putText(frame, hand_labels[i], (px + 12, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    # Draw annotated 2D clicks as white X for comparison
    for i, (ax, ay) in enumerate(pts2d.astype(int)):
        cv2.drawMarker(frame, (ax, ay), (255, 255, 255),
                       cv2.MARKER_CROSS, 14, 2)

    cv2.putText(frame, f"Frame {frame_idx}  reproj_err={best_error:.2f}px",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
print(f"\nSaved annotated video: {OUT_VIDEO}  ({frame_idx} frames)")
