"""Synthetic end-to-end validation of the calibration math (no GUI needed).

Verifies:
  1. umeyama()       recovers a known rigid transform exactly
  2. solve_board_pose() recovers the board pose from projected corners
  3. full pipeline    mocap->camera transform recovered from synthetic
                      board frames + markers (solvePnP + Umeyama chained)

Run:  python3 test_calibration.py
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from calibrate import solve_board_pose, umeyama
from config import INTRINSICS_FILE
from intrinsics import board_object_points


def _load_K():
    intr = json.loads(INTRINSICS_FILE.read_text())
    return np.array(intr["camera_matrix"]), np.array(intr["dist_coeffs"])


def _rotation(rng):
    """Random rotation as a 3-vector (Rodrigues form)."""
    return rng.standard_normal(3)


def test_umeyama():
    rng = np.random.default_rng(0)
    R_true, _ = cv2.Rodrigues(rng.standard_normal(3))
    t_true = rng.standard_normal(3) * 500.0
    pts = rng.standard_normal((3, 60)) * 300.0
    dst = R_true @ pts + t_true[:, None]
    R, t = umeyama(pts, dst)
    err = max(np.abs(R - R_true).max(), np.abs(t - t_true).max())
    assert err < 1e-6, f"umeyama err {err}"
    print(f"  umeyama OK (max err {err:.2e})")


def test_solve_board_pose():
    K, dist = _load_K()
    obj = board_object_points()
    rng = np.random.default_rng(1)
    # board facing camera (so R[2,2] > 0 after Rodrigues)
    for _ in range(5):
        rvec = _rotation(rng)
        R, _ = cv2.Rodrigues(rvec)
        if R[2, 2] < 0:  # flip so board faces camera
            rvec, _ = cv2.Rodrigues(R @ np.diag([1, 1, -1]))
        tvec = rng.standard_normal(3) * 300 + np.array([0.0, 0.0, 1500.0])
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        R_rec, t_rec = solve_board_pose(proj.reshape(-1, 1, 2), K, dist)
        proj2, _ = cv2.projectPoints(obj, cv2.Rodrigues(R_rec)[0], t_rec, K, dist)
        err = np.abs(proj2 - proj).max()
        assert err < 1e-2, f"reproj err {err}"
    print("  solve_board_pose OK (reprojects exactly)")


def test_full_pipeline():
    K, dist = _load_K()
    obj = board_object_points()
    rng = np.random.default_rng(2)

    # plausible board-frame marker layout (on the 40 mm grid, outside pattern)
    s = 40.0
    markers_board = np.array([
        [0.0, -80.0, 0.0], [120.0, -80.0, 0.0], [240.0, -80.0, 0.0],   # top
        [-80.0, 80.0, 0.0], [320.0, 80.0, 0.0],                         # sides
        [120.0, 240.0, 0.0],                                             # bottom
    ], float)

    # true mocap->camera transform
    R_true, _ = cv2.Rodrigues(rng.standard_normal(3))
    t_true = rng.standard_normal(3) * 800.0
    R_true_T = R_true.T

    P_m, P_c = [], []
    for _ in range(12):
        # random board pose in CAMERA space (board facing the camera)
        Rc, _ = cv2.Rodrigues(rng.standard_normal(3))
        if Rc[2, 2] < 0:  # keep the board facing the camera (physical)
            Rc = Rc @ np.diag([1, 1, -1])
        tc = rng.standard_normal(3) * 300 + np.array([0.0, 0.0, 1200.0])

        # markers & corners in camera frame
        M_c = Rc @ markers_board.T + tc[:, None]
        corners_cam = Rc @ obj.T + tc[:, None]

        # ... and the corresponding mocap-frame positions (inverse of T_true)
        M_m = R_true_T @ (M_c - t_true[:, None])
        corners_mocap = R_true_T @ (corners_cam - t_true[:, None])

        # image corners from the camera-frame pose
        proj, _ = cv2.projectPoints(corners_cam.T, np.zeros(3), np.zeros(3), K, dist)
        # recover board pose in camera from image corners
        Rc_rec, tc_rec = solve_board_pose(proj.reshape(-1, 1, 2), K, dist)
        M_c_rec = Rc_rec @ markers_board.T + tc_rec[:, None]
        P_m.append(M_m)
        P_c.append(M_c_rec)

    A = np.concatenate(P_m, axis=1)
    B = np.concatenate(P_c, axis=1)
    R, t = umeyama(A, B)

    assert np.abs(R - R_true).max() < 1e-3, f"R err {np.abs(R-R_true).max()}"
    assert np.abs(t - t_true).max() < 1e-1, f"t err {np.abs(t-t_true).max()}"
    resid = np.linalg.norm(R @ A + t[:, None] - B, axis=0)
    print(f"  full pipeline OK (R err {np.abs(R-R_true).max():.2e}, "
          f"t err {np.abs(t-t_true).max():.2e} mm, residual {resid.max():.2e} mm)")


def test_outlier_rejection():
    """A corner-order-flipped frame must be rejected, not poison the fit."""
    K, dist = _load_K()
    obj = board_object_points()
    rng = np.random.default_rng(42)
    R_true, _ = cv2.Rodrigues(rng.standard_normal(3))
    t_true = rng.standard_normal(3) * 800.0
    R_true_T = R_true.T
    s = 40.0
    markers_board = np.array([
        [0, -80, 0], [120, -80, 0], [240, -80, 0],
        [-80, 80, 0], [320, 80, 0], [120, 240, 0],
    ], float)

    frames = []
    for fr in range(12):
        Rc, _ = cv2.Rodrigues(rng.standard_normal(3))
        if Rc[2, 2] < 0:
            Rc = Rc @ np.diag([1, 1, -1])
        tc = rng.standard_normal(3) * 300 + np.array([0, 0, 1200.])
        M_c = Rc @ markers_board.T + tc[:, None]
        M_m = R_true_T @ (M_c - t_true[:, None])
        corners_cam = Rc @ obj.T + tc[:, None]
        proj, _ = cv2.projectPoints(corners_cam.T, np.zeros(3), np.zeros(3), K, dist)
        corners = proj.reshape(-1, 1, 2)
        if fr == 6:  # poison: 180-deg corner-order flip
            corners = corners[::-1].copy()
        Rc_rec, tc_rec = solve_board_pose(corners, K, dist)
        frames.append((fr, M_m, Rc_rec @ markers_board.T + tc_rec[:, None]))

    # iterative outlier rejection (mirrors compute_transform)
    keep = list(range(len(frames)))
    for _ in range(10):
        A = np.concatenate([frames[i][1] for i in keep], axis=1)
        B = np.concatenate([frames[i][2] for i in keep], axis=1)
        R, t = umeyama(A, B)
        resids = np.array([np.linalg.norm(R @ frames[i][1] + t[:, None]
                                          - frames[i][2], axis=0).mean()
                           for i in keep])
        worst = int(np.argmax(resids))
        if len(keep) > 3 and resids[worst] > 30.0:
            keep.pop(worst)
            continue
        break

    rejected = [frames[i][0] for i in range(len(frames)) if i not in keep]
    assert 6 in rejected, f"poisoned frame not rejected: {rejected}"
    A = np.concatenate([frames[i][1] for i in keep], axis=1)
    B = np.concatenate([frames[i][2] for i in keep], axis=1)
    R, t = umeyama(A, B)
    assert np.abs(R - R_true).max() < 1e-6
    print(f"  outlier rejection OK (rejected {rejected})")


if __name__ == "__main__":
    print("umeyama..."); test_umeyama()
    print("solve_board_pose..."); test_solve_board_pose()
    print("full pipeline..."); test_full_pipeline()
    print("outlier rejection..."); test_outlier_rejection()
    print("ALL CALIBRATION TESTS PASSED")
