"""Stage E: mocap->camera calibration.

For every (video frame, mocap sample) pair where the chessboard is detected
AND all 6 Board markers are tracked:
  1. solvePnP -> board pose in the camera frame (intrinsics from Stage A)
  2. transform the Stage-B board-frame marker positions into the camera frame
  3. pair them with the mocap marker positions at the same instant
Then a single rigid transform mocap->camera is fit with Umeyama (SVD).
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from config import (
    INTRINSICS_FILE,
    MARKERS_FILE,
    MARKER_NAMES,
    TRANSFORM_FILE,
    TRIM_FILE,
)
from data_loader import load_c3d
from intrinsics import board_object_points, detect_board
from sync import mocap_index_for_video_time


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_intrinsics() -> dict:
    return json.loads(INTRINSICS_FILE.read_text())


def load_markers_mm() -> np.ndarray:
    """Board-frame marker positions (6,3) ordered as MARKER_NAMES."""
    data = json.loads(MARKERS_FILE.read_text())
    by_name = {m["name"]: m for m in data["markers"]}
    return np.array([[by_name[n]["x_mm"], by_name[n]["y_mm"], by_name[n]["z_mm"]]
                     for n in MARKER_NAMES], float)


def solve_board_pose(
    corners: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """solvePnP for a planar board with the twin-ambiguity fixed.

    For a planar target there are two poses that project identically (one in
    front, one behind the camera).  We keep the one whose board normal faces
    the camera: if ``R[2,2] < 0``, flip the third column (R @ diag(1,1,-1)).
    Returns (R (3,3), t (3,)) mapping board frame -> camera frame.
    """
    obj = board_object_points()
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, dist)
    R, _ = cv2.Rodrigues(rvec)
    if R[2, 2] < 0:  # board normal pointing away from camera (planar twin)
        R = R @ np.diag([1.0, 1.0, -1.0])
    return R, tvec.reshape(3)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Best rigid transform (R, t) minimizing ||R@src + t - dst|| (src,dst: 3xN)."""
    n = src.shape[1]
    mu_s = src.mean(axis=1, keepdims=True)
    mu_d = dst.mean(axis=1, keepdims=True)
    cov = (dst - mu_d) @ (src - mu_s).T / n
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    t = (mu_d - R @ mu_s).reshape(3)
    return R, t


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------
def build_good_pairs(
    mocap: dict,
    video_times: np.ndarray,
    offset: float,
    trim: dict,
    intrinsics: dict,
    verbose: bool = True,
) -> list[dict]:
    """Collect (video_idx, corners, mocap_idx) for usable frames."""
    K = np.array(intrinsics["camera_matrix"])
    dist = np.array(intrinsics["dist_coeffs"])

    # marker index per name in the c3d
    label_idx = {name: i for i, name in enumerate(mocap["labels"])}
    marker_idx = [label_idx[n] for n in MARKER_NAMES]

    v0 = trim["video_first"]
    v1 = trim["video_last"]
    pairs = []
    n_board = 0
    for vi in range(v0, v1 + 1):
        img = cv2.imread(str(__import__("data_loader").FRAME_DIR / f"{vi:06d}.jpg"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = detect_board(gray)
        if not ok:
            continue
        n_board += 1
        mi = mocap_index_for_video_time(video_times[vi], offset, mocap["fps"])
        if mi < 0 or mi >= mocap["n_frames"]:
            continue
        if not all(mocap["presence"][mi, j] for j in marker_idx):
            continue
        pairs.append({"video_idx": vi, "corners": corners, "mocap_idx": mi})

    if verbose:
        print(f"board detected in {n_board} frames; usable pairs (board + all "
              f"6 markers): {len(pairs)}")
    return pairs


def compute_transform(
    mocap: dict,
    video_times: np.ndarray,
    offset: float,
    trim: dict,
    intrinsics: dict,
    markers_mm: np.ndarray,
    verbose: bool = True,
    outlier_mm: float = 30.0,
) -> dict:
    """Fit the mocap->camera rigid transform with iterative outlier rejection.

    Outlier frames (corner-order flips, bad detections) are removed in a
    robust loop so they don't poison the fit; they are reported.
    """
    pairs = build_good_pairs(mocap, video_times, offset, trim, intrinsics, verbose)
    if len(pairs) < 3:
        raise RuntimeError(f"Only {len(pairs)} usable frames - need >= 3")

    K = np.array(intrinsics["camera_matrix"])
    dist = np.array(intrinsics["dist_coeffs"])
    label_idx = {name: i for i, name in enumerate(mocap["labels"])}
    marker_idx = [label_idx[n] for n in MARKER_NAMES]

    # per-frame marker correspondences (mocap -> camera)
    frames = []  # (video_idx, P_mocap (3,6), P_cam (3,6))
    for p in pairs:
        R, t = solve_board_pose(p["corners"], K, dist)
        P_cam = (R @ markers_mm.T + t[:, None])          # (3, 6)
        P_mocap = mocap["xyz"][marker_idx, :, p["mocap_idx"]].T  # (3, 6)
        frames.append((p["video_idx"], P_mocap, P_cam))

    # --- iterative robust fit ------------------------------------------
    keep = list(range(len(frames)))
    R, t = None, None
    for _ in range(10):
        A = np.concatenate([frames[i][1] for i in keep], axis=1)
        B = np.concatenate([frames[i][2] for i in keep], axis=1)
        R, t = umeyama(A, B)
        # per-frame mean residual (mm)
        resids = []
        for i in keep:
            pm, pc = frames[i][1], frames[i][2]
            resids.append(np.linalg.norm(R @ pm + t[:, None] - pc, axis=0).mean())
        resids = np.array(resids)
        worst = int(np.argmax(resids))
        if len(keep) > 3 and resids[worst] > outlier_mm:
            rejected = keep.pop(worst)
            if verbose:
                print(f"  rejected frame {frames[rejected][0]} "
                      f"(mean res {resids[worst]:.1f} mm)")
            continue
        break

    # final residuals on kept frames
    A = np.concatenate([frames[i][1] for i in keep], axis=1)
    B = np.concatenate([frames[i][2] for i in keep], axis=1)
    resid = np.linalg.norm(R @ A + t[:, None] - B, axis=0)  # per marker, mm
    per_frame_err = []
    for i in keep:
        vi, pm, pc = frames[i]
        e = np.linalg.norm(R @ pm + t[:, None] - pc, axis=0)
        per_frame_err.append({"video_idx": vi, "mean_mm": float(e.mean()),
                              "max_mm": float(e.max())})
    rejected_frames = [frames[i][0] for i in range(len(frames)) if i not in keep]

    result = {
        "rotation": R.tolist(),
        "translation_mm": t.tolist(),
        "convention": "P_camera = R @ P_mocap + t  (mm)",
        "camera_matrix": intrinsics["camera_matrix"],
        "dist_coeffs": intrinsics["dist_coeffs"],
        "n_frames_used": len(keep),
        "n_frames_total": len(frames),
        "rejected_frames": rejected_frames,
        "n_correspondences": int(A.shape[1]),
        "mean_residual_mm": float(resid.mean()),
        "median_residual_mm": float(np.median(resid)),
        "p95_residual_mm": float(np.percentile(resid, 95)),
        "max_residual_mm": float(resid.max()),
        "per_frame_error_mm": per_frame_err,
        "sync_offset_s": float(offset),
        "trim": trim,
    }
    TRANSFORM_FILE.write_text(json.dumps(result, indent=2))

    if verbose:
        print(f"used {len(keep)}/{len(frames)} frames, "
              f"{A.shape[1]} marker correspondences")
        print(f"mean residual: {resid.mean():.2f} mm   "
              f"p95: {np.percentile(resid, 95):.2f} mm   "
              f"max: {resid.max():.2f} mm")
        print("R =\n", np.round(R, 5))
        print("t (mm) =", np.round(t, 2))
    return result


def run_calibration(verbose: bool = True) -> dict:
    """Full Stage E from saved files (intrinsics, markers, trim, sync)."""
    intrinsics = load_intrinsics()
    markers_mm = load_markers_mm()
    trim = json.loads(TRIM_FILE.read_text())
    sync = json.loads(__import__("config").SYNC_FILE.read_text())
    offset = sync["offset_s"]
    mocap = load_c3d()
    video_times = np.load(__import__("config").FRAME_TIMESTAMPS_FILE)
    # re-apply trim to get frame ranges (start/end in video seconds)
    from trim import apply_trim
    trim_full = apply_trim(video_times, mocap, offset, trim["start_s"], trim["end_s"])
    return compute_transform(mocap, video_times, offset, trim_full, intrinsics,
                             markers_mm, verbose)
