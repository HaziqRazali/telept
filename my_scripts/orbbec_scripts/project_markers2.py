"""
Project Reference_Marker1-6 and Unlabeled_ markers onto the Femto Bolt video.
Assumes camera is STATIC — camera pose is solved once from frame 0.

  Step 1 (once):
    Brute-force SolvePnP(2D clicks, 3D Reference_Marker positions at C3D frame 0)
    → fixed camera pose (R0, t0)

  Step 2 (every video frame):
    C3D frame = video_frame - vid_offset  (vid_offset=45 found empirically)
    Project Reference_Markers and Unlabeled_ markers at that C3D frame
    through the fixed (R0, t0, K).

Usage:
    conda activate c3d_viewer
    python project_markers2.py
    python project_markers2.py --vid_offset 45
"""

import cv2
import numpy as np
import ezc3d
from itertools import permutations
import math
import os
import argparse

# ── Defaults ──────────────────────────────────────────────────────────────────
BASE = "/home/haziq/datasets/telept/data/mocap_recordings"
DEFAULT_C3D   = f"{BASE}/2026_03_20.c3d"
DEFAULT_VIDEO = f"{BASE}/2026_03_20.mp4"
DEFAULT_ANNOT = f"{BASE}/2026_03_20_circle.txt"

# ── Camera intrinsics (Femto Bolt 1920×1080) ──────────────────────────────────
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
], dtype=np.float64)  # k1, k2, p1, p2, k3 (OpenCV order)

COLORS = [
    (0,   255, 0),
    (255, 128, 0),
    (0,   128, 255),
    (255, 0,   255),
    (0,   255, 255),
    (255, 255, 0),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_c3d_markers(c3d_path, names):
    """
    Return xyz[n_markers, n_frames, 3] and res[n_markers, n_frames]
    for the markers whose labels are in `names` (preserving order).
    """
    c = ezc3d.c3d(c3d_path)
    all_labels = []
    for k in sorted(kk for kk in c["parameters"]["POINT"] if kk.startswith("LABELS")):
        all_labels.extend(c["parameters"]["POINT"][k]["value"])
    pts = c["data"]["points"]  # (4, total_markers, n_frames)
    all_labels = all_labels[: pts.shape[1]]
    rate = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    units = c["parameters"]["POINT"]["UNITS"]["value"][0]

    indices = [all_labels.index(n) for n in names]
    xyz = pts[:3, indices, :].transpose(1, 2, 0)  # (n_markers, n_frames, 3)
    res = pts[3,  indices, :]                      # (n_markers, n_frames)
    return xyz, res, rate, units


def load_2d_clicks(txt_path):
    """Load annotation file produced by annotate_circle.py."""
    pts = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")[-1].split(",")
            x = int(parts[0].split("=")[1])
            y = int(parts[1].split("=")[1])
            pts.append((x, y))
    return np.array(pts, dtype=np.float64)  # (N, 2)


def project_points(pts3d, R, t):
    """Project (N,3) world points → (N,2) pixel coords using R, t, K, dist."""
    pts3d = np.asarray(pts3d, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(R)
    proj, _ = cv2.projectPoints(pts3d, rvec, t, K, dist_coeffs)
    return proj.reshape(-1, 2)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3d",   default=DEFAULT_C3D)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--annot", default=DEFAULT_ANNOT)
    parser.add_argument("--out",   default=None,
                        help="Output video path (default: input_projected.mp4)")
    parser.add_argument("--vid_offset", type=int, default=45,
                        help="video_frame = c3d_frame + vid_offset (default: 45)")
    args = parser.parse_args()

    offset = args.vid_offset

    out_video = args.out or os.path.splitext(args.video)[0] + "_projected.mp4"

    # ── Load data ──────────────────────────────────────────────────────────────
    ref_names = [f"Reference_Marker{i}" for i in range(1, 7)]

    print("Loading C3D...")
    ref_xyz, ref_res, rate, units = load_c3d_markers(args.c3d, ref_names)
    n_frames_c3d = ref_xyz.shape[1]
    print(f"  Reference_Markers: {ref_xyz.shape}  ({units})")
    print(f"  C3D frames: {n_frames_c3d}  @ {rate} fps")

    # Load ALL markers to find Unlabeled_ ones
    c_raw = ezc3d.c3d(args.c3d)
    all_labels_raw = []
    for k in sorted(kk for kk in c_raw["parameters"]["POINT"] if kk.startswith("LABELS")):
        all_labels_raw.extend(c_raw["parameters"]["POINT"][k]["value"])
    all_pts_raw = c_raw["data"]["points"]  # (4, M, F)
    all_labels_raw = all_labels_raw[:all_pts_raw.shape[1]]
    unlabeled_idx = [i for i, l in enumerate(all_labels_raw) if l.startswith("Unlabeled_")]
    unlabeled_names = [all_labels_raw[i] for i in unlabeled_idx]
    # all_pts_raw kept in memory for per-frame access
    unl_xyz_all = all_pts_raw[:3, unlabeled_idx, :].transpose(1, 2, 0).astype(np.float64)  # (N_unl, F, 3)
    unl_res_all = all_pts_raw[3,  unlabeled_idx, :]                                          # (N_unl, F)
    print(f"  Unlabeled_ markers: {len(unlabeled_names)}")

    pts2d = load_2d_clicks(args.annot)
    print(f"  2D clicks loaded: {len(pts2d)}")

    # ── Frame 0: 3D Reference_Markers ─────────────────────────────────────────
    pts3d_f0 = ref_xyz[:, 0, :]   # (6, 3)
    valid_f0  = ref_res[:, 0] >= 0
    if not valid_f0.all():
        print(f"  WARNING: {(~valid_f0).sum()} Reference_Markers invalid at frame 0")

    print("\n3D Reference_Markers at frame 0:")
    for i, (name, pt) in enumerate(zip(ref_names, pts3d_f0)):
        print(f"  [{i}] {name}: {pt}  {'OK' if valid_f0[i] else 'INVALID'}")

    # ── Brute-force SolvePnP ──────────────────────────────────────────────────
    n = len(pts2d)
    assert n == 6, f"Expected 6 clicks, got {n}"
    print(f"\nSearching {n}! = {math.factorial(n)} permutations for best 2D↔3D correspondence...")

    best_err   = np.inf
    best_perm  = None
    best_rvec  = None
    best_tvec  = None

    for perm in permutations(range(n)):
        pts3d_perm = pts3d_f0[list(perm)]
        ok, rvec, tvec = cv2.solvePnP(
            pts3d_perm, pts2d, K, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue
        proj = project_points(pts3d_perm, cv2.Rodrigues(rvec)[0], tvec)
        err = float(np.mean(np.linalg.norm(proj - pts2d, axis=1)))
        if err < best_err:
            best_err   = err
            best_perm  = perm
            best_rvec  = rvec.copy()
            best_tvec  = tvec.copy()

    R0, _ = cv2.Rodrigues(best_rvec)
    t0    = best_tvec.reshape(3)

    print(f"Best permutation : {best_perm}")
    print(f"Reprojection err : {best_err:.3f} px")
    print(f"Camera pos (world): {(-R0.T @ t0).ravel()}")
    print("\nCorrespondence (click → marker):")
    for click_i, marker_i in enumerate(best_perm):
        print(f"  click {click_i+1} {tuple(pts2d[click_i].astype(int))} "
              f"→ {ref_names[marker_i]} {tuple(pts3d_f0[marker_i].round(1))}")

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    total_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_vid   = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps_vid, (W, H))

    n_total = total_vid
    print(f"\nProcessing {n_total} video frames  (C3D offset={offset}, so C3D frames {-offset}..{n_total-1-offset})...")

    frame_idx = 0
    while frame_idx < n_total:
        ret, frame = cap.read()
        if not ret:
            break

        c3d_frame = frame_idx - offset
        in_range = 0 <= c3d_frame < n_frames_c3d

        if in_range:
            # -- Reference_Markers at this C3D frame
            pts3d_fN = ref_xyz[:, c3d_frame, :]       # (6, 3)
            valid_fN = ref_res[:, c3d_frame] >= 0     # (6,)

            proj = project_points(pts3d_fN, R0, t0)  # (6, 2)

            for i, (px, py) in enumerate(proj):
                if not valid_fN[i] or not np.isfinite(px) or not np.isfinite(py):
                    continue
                px, py = int(round(px)), int(round(py))
                if 0 <= px < W and 0 <= py < H:
                    col = COLORS[i % len(COLORS)]
                    cv2.circle(frame, (px, py), 10, col, 2)
                    cv2.circle(frame, (px, py), 3,  col, -1)
                    cv2.putText(frame, ref_names[i], (px + 12, py - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            # -- Unlabeled_ markers at this C3D frame
            unl_xyz_fN = unl_xyz_all[:, c3d_frame, :]              # (N_unl, 3)
            valid_unl  = (unl_res_all[:, c3d_frame] >= 0) & np.all(np.isfinite(unl_xyz_fN), axis=1)
            if valid_unl.any():
                proj_unl = project_points(unl_xyz_fN[valid_unl], R0, t0)
                for (px, py) in proj_unl:
                    if not np.isfinite(px) or not np.isfinite(py):
                        continue
                    px, py = int(round(px)), int(round(py))
                    if 0 <= px < W and 0 <= py < H:
                        cv2.circle(frame, (px, py), 5, (255, 0, 0), -1)

        # -- Draw original 2D clicks on the video frame that corresponds to C3D frame 0
        if frame_idx == offset:
            for ax, ay in pts2d.astype(int):
                cv2.drawMarker(frame, (ax, ay), (255, 255, 255),
                               cv2.MARKER_CROSS, 16, 2)

        status = f"C3D {c3d_frame}" if in_range else "C3D: out of range"
        cv2.putText(frame,
                    f"Video {frame_idx}  {status}  reproj_err={best_err:.2f}px",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 50 == 0:
            print(f"  {frame_idx}/{n_total}")

    cap.release()
    writer.release()
    print(f"\nSaved: {out_video}  ({frame_idx} frames)")


if __name__ == "__main__":
    main()
