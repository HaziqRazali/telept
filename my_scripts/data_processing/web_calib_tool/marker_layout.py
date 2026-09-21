"""Stage B: reflective-marker layout on the chessboard schematic.

The user clicks, on a drawn schematic of the chessboard, where the 6
reflective markers sit.  The board-frame coordinates (mm) of those clicks are
saved, then auto-verified against the C3D ``Board1..Board6`` trajectories via
pairwise-distance matching (the board is rigid, so both distance matrices must
agree up to a few mm).

Coordinate frame
----------------
Board frame, origin = top-left *inner corner* of the pattern (matching
``intrinsics.board_object_points``).  +X to the right along a corner row, +Y
down along a corner column, +Z out of the board plane.  Units: mm.
The physical 9x6 pattern spans x in [-40, 320], y in [-40, 200]; the canvas
extends 2 squares (80 mm) beyond on every side.
"""

from __future__ import annotations

import json
import itertools

import cv2
import numpy as np

from config import (
    BOARD_INNER_CORNERS,
    BOARD_MARGIN_SQUARES,
    C3D_PATH,
    MARKER_NAMES,
    MARKERS_FILE,
    NUM_MARKERS,
    SQUARE_SIZE_MM,
)
from data_loader import load_c3d

PX_PER_MM = 2.0  # canvas resolution (deterministic pixel<->mm mapping)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------
def board_extent() -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) of the canvas in board-frame mm."""
    cols, rows = BOARD_INNER_CORNERS
    s = SQUARE_SIZE_MM
    m = BOARD_MARGIN_SQUARES * s
    pat_x_lo, pat_x_hi = -s, cols * s          # pattern spans 9 squares wide
    pat_y_lo, pat_y_hi = -s, rows * s          # ... and 6 squares tall
    return pat_x_lo - m, pat_x_hi + m, pat_y_lo - m, pat_y_hi + m


def pattern_extent() -> tuple[float, float, float, float]:
    """(x_min, x_max, y_min, y_max) of the checkerboard pattern only (mm)."""
    s = SQUARE_SIZE_MM
    cols, rows = BOARD_INNER_CORNERS
    return -s, cols * s, -s, rows * s


def canvas_size_px() -> tuple[int, int]:
    x_lo, x_hi, y_lo, y_hi = board_extent()
    return int(round((x_hi - x_lo) * PX_PER_MM)), int(round((y_hi - y_lo) * PX_PER_MM))


def mm_to_px(x: float, y: float) -> tuple[int, int]:
    x_lo, _, y_lo, _ = board_extent()
    return int(round((x - x_lo) * PX_PER_MM)), int(round((y - y_lo) * PX_PER_MM))


def px_to_mm(px: float, py: float) -> tuple[float, float]:
    x_lo, _, y_lo, _ = board_extent()
    return px / PX_PER_MM + x_lo, py / PX_PER_MM + y_lo


def snap_to_grid(x: float, y: float) -> tuple[float, float]:
    """Round to the nearest multiple of SQUARE_SIZE_MM."""
    s = SQUARE_SIZE_MM
    return round(x / s) * s, round(y / s) * s


# ----------------------------------------------------------------------
# Canvas rendering
# ----------------------------------------------------------------------
def render_canvas(placed: list[tuple[str, float, float]] | None = None) -> np.ndarray:
    """Render the board schematic + margin dot grid as a BGR image.

    placed: optional list of (name, x_mm, y_mm) already-placed markers.
    """
    cols, rows = BOARD_INNER_CORNERS
    s = SQUARE_SIZE_MM
    W, H = canvas_size_px()
    img = np.full((H, W, 3), 235, np.uint8)  # light grey bg

    # --- checkerboard (9 cols x 6 rows) -------------------------------
    for c in range(cols + 1):
        for r in range(rows + 1):
            x0, y0 = mm_to_px(-s + c * s, -s + r * s)
            x1, y1 = mm_to_px(-s + (c + 1) * s, -s + (r + 1) * s)
            color = (0, 0, 0) if (c + r) % 2 == 0 else (255, 255, 255)
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), color, -1)

    # --- margin dot grid (outside the pattern only) --------------------
    x_lo, x_hi, y_lo, y_hi = board_extent()
    px_lo, px_hi, py_lo, py_hi = pattern_extent()
    dot_color = (200, 120, 20)
    for x in np.arange(x_lo, x_hi + 1e-6, s):
        for y in np.arange(y_lo, y_hi + 1e-6, s):
            if px_lo <= x <= px_hi and py_lo <= y <= py_hi:
                continue  # skip dots on the pattern
            cx, cy = mm_to_px(x, y)
            cv2.circle(img, (cx, cy), 7, dot_color, -1)
            cv2.circle(img, (cx, cy), 7, (255, 255, 255), 1)

    # --- origin marker + hint -----------------------------------------
    ox, oy = mm_to_px(0, 0)
    cv2.drawMarker(img, (ox, oy), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    cv2.putText(img, "origin (0,0) = top-left inner corner", (ox + 12, oy - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"grid spacing {s:.0f} mm  |  canvas = pattern + "
                     f"{BOARD_MARGIN_SQUARES} squares each side",
                (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1,
                cv2.LINE_AA)

    # --- already-placed markers ----------------------------------------
    if placed:
        for name, x, y in placed:
            cx, cy = mm_to_px(x, y)
            cv2.circle(img, (cx, cy), 10, (0, 160, 0), -1)
            cv2.circle(img, (cx, cy), 10, (0, 0, 0), 1)
            cv2.putText(img, name, (cx + 12, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 0), 2, cv2.LINE_AA)
    return img


# ----------------------------------------------------------------------
# Verification vs C3D (rigid distance matching)
# ----------------------------------------------------------------------
def c3d_marker_pairwise_distances(c3d: dict, names: list[str] | None = None) -> np.ndarray:
    """(N,N) pairwise distances (mm) between markers, robust to board motion.

    Distances are invariant under rigid motion, so for each marker pair we
    take the MEDIAN of the per-frame distance over all frames where BOTH
    markers are visible.  This stays correct even if the board is moved
    during the capture (a mean-POSITION approach would be smeared and wrong).
    Returns np.nan for pairs that are never co-visible.
    """
    names = list(names) if names is not None else list(MARKER_NAMES)
    n = len(names)
    label_idx = {name: i for i, name in enumerate(c3d["labels"])}
    xyz, pres = c3d["xyz"], c3d["presence"]
    D = np.zeros((n, n))
    for a in range(n):
        ia = label_idx.get(names[a])
        if ia is None:
            D[a, :] = D[:, a] = np.nan
            continue
        for b in range(a + 1, n):
            ib = label_idx.get(names[b])
            if ib is None:
                D[a, b] = D[b, a] = np.nan
                continue
            both = pres[ia] & pres[ib]
            if both.sum() == 0:
                D[a, b] = D[b, a] = np.nan
                continue
            # NB: xyz[ia] is (3, F); mask the LAST axis to keep (3, n).
            # (xyz[ia, :, both] would reorder axes and give (n, 3) -- wrong.)
            pa = xyz[ia][:, both]
            pb = xyz[ib][:, both]
            d = np.linalg.norm(pa - pb, axis=0)
            D[a, b] = D[b, a] = float(np.median(d))
    return D


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """(N,3) -> (N,N) pairwise Euclidean distance matrix."""
    return np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))


def verify_markers(markers_mm: np.ndarray, c3d: dict) -> dict:
    """Match clicked markers (board frame, Nx3) to C3D labels.

    Compares pairwise-distance matrices; the C3D side is built per-frame with
    a robust median (see c3d_marker_pairwise_distances), so the board may
    move during the capture.  Brute-forces the best permutation (N<=6 so 720
    perms max) minimizing the mean absolute difference between the two
    matrices.
    """
    n = markers_mm.shape[0]
    d_click = pairwise_distances(markers_mm)
    d_c3d = c3d_marker_pairwise_distances(c3d, MARKER_NAMES[:n])

    # markers never visible (all their pairs are NaN)
    bad = [MARKER_NAMES[i] for i in range(n) if np.isnan(d_c3d[i]).all()]
    if bad:
        return {"ok": False, "error": f"missing markers in C3D: {bad}"}

    best_perm, best_cost = None, np.inf
    for perm in itertools.permutations(range(n)):
        diff = np.abs(d_click - d_c3d[np.ix_(perm, perm)])
        cost = np.nanmean(diff[np.triu_indices(n, 1)])
        if cost < best_cost:
            best_cost, best_perm = cost, perm

    assigned = [MARKER_NAMES[i] for i in best_perm]
    err = np.abs(d_click - d_c3d[np.ix_(best_perm, best_perm)])
    iu = np.triu_indices(n, 1)
    return {
        "ok": True,
        "assigned": assigned,
        "mean_dist_err_mm": float(np.nanmean(err[iu])),
        "max_dist_err_mm": float(np.nanmax(err[iu])),
        "n_markers": n,
    }


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def save_markers(markers_mm: np.ndarray, verify: dict | None = None) -> None:
    """Save clicked marker positions to markers.json (board frame, mm)."""
    n = markers_mm.shape[0]
    labels = verify["assigned"] if (verify and verify["ok"]) else \
        [f"{i + 1}" for i in range(n)]
    data = {
        "board_inner_corners": list(BOARD_INNER_CORNERS),
        "square_size_mm": SQUARE_SIZE_MM,
        "margin_squares": BOARD_MARGIN_SQUARES,
        "coordinate_note": "board frame; origin = top-left inner corner; "
                           "+X right along corners, +Y down along corners, mm",
        "markers": [
            {"name": labels[i],
             "x_mm": float(markers_mm[i, 0]),
             "y_mm": float(markers_mm[i, 1]),
             "z_mm": 0.0}
            for i in range(n)
        ],
        "verification": verify,
    }
    MARKERS_FILE.write_text(json.dumps(data, indent=2))


def load_markers() -> dict | None:
    if not MARKERS_FILE.exists():
        return None
    return json.loads(MARKERS_FILE.read_text())


# ----------------------------------------------------------------------
# CLI self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    img = render_canvas()
    cv2.imwrite(str(__import__("config").CACHE_DIR / "marker_canvas.png"), img)
    print(f"canvas saved: {__import__('config').CACHE_DIR / 'marker_canvas.png'} "
          f"({img.shape[1]}x{img.shape[0]} px)")

    # demo: place 6 markers on the 40mm grid around the pattern and verify
    s = SQUARE_SIZE_MM
    demo = np.array([
        [1 * s, -2 * s, 0],    # top edge
        [4 * s, -2 * s, 0],
        [7 * s, -2 * s, 0],
        [-2 * s, 2 * s, 0],    # left edge
        [9 * s, 2 * s, 0],     # right edge  (x=360 > pattern hi 320)
        [4 * s, 6 * s, 0],     # bottom edge (y=240 > pattern lo... )
    ], float)
    c3d = load_c3d(C3D_PATH)
    v = verify_markers(demo, c3d)
    print("demo verify:", json.dumps(v, indent=2))
