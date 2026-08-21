"""Chessboard detection + camera intrinsic self-calibration from the video."""

from __future__ import annotations

import json

import cv2
import numpy as np

from config import (
    BOARD_INNER_CORNERS,
    FRAME_DIR,
    INTRINSICS_FILE,
    SQUARE_SIZE_MM,
    VIDEO_PATH,
)
from data_loader import pre_extract_frames


def board_object_points() -> np.ndarray:
    """3D object points for the chessboard, matching OpenCV corner ordering.

    OpenCV returns corners row-by-row (top to bottom), left to right.  We use
    the same ordering: (col * size, row * size, 0).
    """
    cols, rows = BOARD_INNER_CORNERS
    obj = np.zeros((cols * rows, 3), np.float32)
    k = 0
    for r in range(rows):
        for c in range(cols):
            obj[k] = (c * SQUARE_SIZE_MM, r * SQUARE_SIZE_MM, 0.0)
            k += 1
    return obj


def detect_board(gray: np.ndarray):
    """Find chessboard corners; returns (ok, corners) with corners None if absent."""
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FILTER_QUADS
    )
    ok, corners = cv2.findChessboardCorners(gray, BOARD_INNER_CORNERS, None, flags)
    return ok, corners


def detect_boards_all_frames(video_path=VIDEO_PATH, gray_frames=None) -> list:
    """Detect the board in every cached frame; return list of (frame_idx, corners)."""
    frames, _ = pre_extract_frames(video_path)
    results = []
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = detect_board(gray)
        if ok:
            results.append((i, corners))
    return results


def calibrate_intrinsics(video_path=VIDEO_PATH, verbose: bool = True) -> dict:
    """Self-calibrate camera intrinsics from all chessboard detections.

    Returns a result dict and writes output/intrinsics.json.
    """
    frames, _ = pre_extract_frames(video_path)
    img0 = cv2.imread(str(frames[0]))
    h, w = img0.shape[:2]

    obj_pts = board_object_points()
    obj_points, img_points = [], []
    good_frames = []

    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = detect_board(gray)
        if not ok:
            continue
        obj_points.append(obj_pts)
        img_points.append(corners.reshape(-1, 2))
        good_frames.append(i)

    if len(good_frames) < 5:
        raise RuntimeError(f"Only {len(good_frames)} good chessboard frames - need >= 5")

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, (w, h), None, None
    )

    # per-frame reprojection error
    per_frame_err = []
    for o, i, rv, tv in zip(obj_points, img_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        err = np.mean(np.linalg.norm(proj[:, 0, :] - i, axis=1))
        per_frame_err.append(float(err))

    result = {
        "video": str(video_path),
        "resolution": [w, h],
        "board_inner_corners": list(BOARD_INNER_CORNERS),
        "square_size_mm": SQUARE_SIZE_MM,
        "n_detected_frames": len(good_frames),
        "n_total_frames": len(frames),
        "rms": float(rms),
        "mean_reproj_px": float(np.mean(per_frame_err)),
        "max_reproj_px": float(np.max(per_frame_err)),
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.reshape(-1).tolist(),
        "per_frame_reproj_px": per_frame_err,
        "good_frame_indices": good_frames,
    }
    INTRINSICS_FILE.write_text(json.dumps(result, indent=2))

    if verbose:
        print(f"calibrated from {len(good_frames)}/{len(frames)} frames")
        print(f"resolution: {w}x{h}")
        print(f"RMS (cv2): {rms:.4f} px")
        print(f"mean per-frame reproj: {np.mean(per_frame_err):.3f} px  "
              f"max: {np.max(per_frame_err):.3f} px")
        print(f"camera matrix:\n{K}")
        print(f"distortion: {dist.reshape(-1)}")
    return result


if __name__ == "__main__":
    calibrate_intrinsics()
