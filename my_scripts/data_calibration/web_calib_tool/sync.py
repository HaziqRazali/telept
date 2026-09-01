"""Stage C: LED-based temporal synchronization.

Two binary "blink" traces are produced:
  * video  : mean grayscale intensity inside a user-drawn ROI box, thresholded
  * mocap  : is any mocap marker inside a user-drawn 3D box, per frame

Then a single time offset is found by cross-correlating the two traces.

Offset convention (documented here, used everywhere):
    video_time = mocap_time + offset
So for a video frame at time t_v the matching mocap time is ``t_v - offset``.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from config import SYNC_FILE, C3D_PATH, VIDEO_PATH
from data_loader import load_c3d, pre_extract_frames


# ----------------------------------------------------------------------
# Video-side trace
# ----------------------------------------------------------------------
def compute_video_trace(
    roi: tuple[int, int, int, int], video_path=VIDEO_PATH
) -> tuple[np.ndarray, np.ndarray]:
    """Mean grayscale intensity inside ROI (x0, y0, x1, y1) per video frame.

    Returns (trace, video_times).
    """
    frames, times = pre_extract_frames(video_path)
    x0, y0, x1, y1 = roi
    trace = np.empty(len(frames), np.float32)
    for i, fp in enumerate(frames):
        img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
        trace[i] = img[y0:y1, x0:x1].mean()
    return trace, times


def threshold_trace(trace: np.ndarray, thr: float) -> np.ndarray:
    """Binary 1/0 trace from a threshold."""
    return (trace > thr).astype(np.float64)


# ----------------------------------------------------------------------
# Mocap-side trace
# ----------------------------------------------------------------------
def compute_mocap_trace(
    mocap: dict, box_lo: np.ndarray, box_hi: np.ndarray
) -> np.ndarray:
    """Binary trace: 1 where ANY tracked marker is inside the 3D box."""
    xyz = mocap["xyz"]
    pres = mocap["presence"]
    inside = (xyz >= box_lo[None, :, None]) & (xyz <= box_hi[None, :, None])
    inside = inside.all(axis=1) & pres          # (N, F)
    return inside.any(axis=0).astype(np.float64)


# ----------------------------------------------------------------------
# Cross-correlation
# ----------------------------------------------------------------------
def cross_correlate(
    video_bin: np.ndarray,
    video_times: np.ndarray,
    mocap_bin: np.ndarray,
    fps: float,
    max_offset: float = 60.0,
) -> tuple[float, float]:
    """Find offset (s) aligning the two binary traces.

    Returns (offset, agreement) where agreement in [0,1] is the fraction of
    overlapping samples that agree at the best offset.
    """
    m_t = np.arange(len(mocap_bin)) / fps
    # video signal sampled onto the (dense) mocap time grid
    v_on_m = np.interp(m_t, video_times, video_bin)
    max_lag = int(max_offset * fps)

    best_score, best_lag = -1.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a = v_on_m[lag:]
            b = mocap_bin[: len(a)]
        else:
            a = v_on_m[:lag]
            b = mocap_bin[-lag:]
        if len(a) < 60:
            continue
        score = np.mean((a > 0.5) == (b > 0.5))
        if score > best_score:
            best_score, best_lag = score, lag

    offset = best_lag / fps  # video_time = mocap_time + offset
    return offset, best_score


# ----------------------------------------------------------------------
# Auto-find the blinking source (mocap side)
# ----------------------------------------------------------------------
def auto_find_led(
    mocap: dict, max_move_mm: float = 30.0, min_frames: int = 30
) -> dict | None:
    """Locate a stationary marker whose presence toggles (the blinking LED).

    Ranks candidates by (presence-transitions, visible-frame-count), i.e. it
    prefers a marker that clearly blinks, but falls back to any stationary
    marker (e.g. one that appears once at the start).

    Returns dict {name, position_mm (3,), transitions, visible_frames} or None.
    """
    cands = []
    for i, name in enumerate(mocap["labels"]):
        p = mocap["presence"][i]
        n_vis = int(p.sum())
        if n_vis < min_frames:
            continue
        pos = mocap["xyz"][i, :, p]              # (n_vis, 3) - bool mask moves axis
        if pos.std(axis=0).max() > max_move_mm:
            continue                              # not stationary
        transitions = int(np.sum(np.diff(p.astype(np.int8)) != 0))
        cands.append({
            "name": name,
            "position_mm": pos.mean(axis=0),
            "transitions": transitions,
            "visible_frames": n_vis,
        })
    if not cands:
        return None
    cands.sort(key=lambda c: (c["transitions"], c["visible_frames"]), reverse=True)
    return cands[0]


# ----------------------------------------------------------------------
# Time<->frame mapping
# ----------------------------------------------------------------------
def video_index_at_time(t: float, video_times: np.ndarray) -> int:
    return int(np.argmin(np.abs(video_times - t)))


def mocap_index_for_video_time(
    t_video: float, offset: float, fps: float
) -> int:
    """Mocap frame index matching a video timestamp (video_time = mocap_time + offset)."""
    return int(round((t_video - offset) * fps))


def video_time_for_mocap_index(idx: int, fps: float, offset: float) -> float:
    return idx / fps + offset


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def save_sync(offset: float, agreement: float, roi, box_lo, box_hi, thr_video) -> None:
    SYNC_FILE.write_text(json.dumps({
        "offset_s": float(offset),
        "agreement": float(agreement),
        "convention": "video_time = mocap_time + offset_s",
        "video_roi": list(roi) if roi is not None else None,
        "video_threshold": float(thr_video),
        "mocap_box_lo_mm": list(box_lo) if box_lo is not None else None,
        "mocap_box_hi_mm": list(box_hi) if box_hi is not None else None,
    }, indent=2))


def load_sync() -> dict | None:
    if not SYNC_FILE.exists():
        return None
    return json.loads(SYNC_FILE.read_text())
