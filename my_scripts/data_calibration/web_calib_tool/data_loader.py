"""Loading / caching for the iPad video and the C3D mocap data.

Design notes
------------
* The iPad video is VARIABLE frame rate (VFR).  Frame index != time, so we
  pre-extract every frame to a JPEG cache and record its true timestamp
  (CAP_PROP_POS_MSEC).  All downstream sync/calibration math works in
  SECONDS, not frame indices.
* The C3D is uniform (100 Hz) so ``time = frame_index / fps``.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import ezc3d
import numpy as np

from config import (
    C3D_PATH,
    FRAME_DIR,
    FRAME_META_FILE,
    FRAME_TIMESTAMPS_FILE,
    VIDEO_PATH,
)


# ----------------------------------------------------------------------
# Video
# ----------------------------------------------------------------------
def pre_extract_frames(
    video_path: Path = VIDEO_PATH, force: bool = False
) -> tuple[list[Path], np.ndarray]:
    """Extract every video frame to a JPEG cache, returning (paths, times).

    Times are in seconds (one per frame).  Skips extraction if the cache is
    already complete for this video.
    """
    video_path = Path(video_path)
    meta_path = FRAME_META_FILE
    times_path = FRAME_TIMESTAMPS_FILE

    # --- reuse cache if present ----------------------------------------
    if not force and meta_path.exists() and times_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("video") == str(video_path):
            frames = sorted(FRAME_DIR.glob("*.jpg"))
            times = np.load(times_path)
            if len(frames) == meta["count"] == len(times):
                return frames, times

    # --- extract --------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: list[Path] = []
    times: list[float] = []
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ok2, img = cap.retrieve()
        if not ok2:
            continue
        out = FRAME_DIR / f"{idx:06d}.jpg"
        cv2.imwrite(str(out), img)
        frames.append(out)
        times.append(t)
        idx += 1
    cap.release()

    times = np.asarray(times, dtype=np.float64)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta = {
        "video": str(video_path),
        "count": len(frames),
        "resolution": [w, h],
        "avg_fps": float(len(frames) / (times[-1] - times[0])) if len(frames) > 1 else None,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    np.save(times_path, times)
    return frames, times


def load_frame(frame_path: Path) -> np.ndarray:
    """Load a cached frame as a BGR numpy array."""
    img = cv2.imread(str(frame_path))
    if img is None:
        raise RuntimeError(f"Failed to read cached frame: {frame_path}")
    return img


def load_frame_times() -> np.ndarray:
    """Return per-frame timestamps (seconds).  Raises if not extracted yet."""
    if not FRAME_TIMESTAMPS_FILE.exists():
        raise RuntimeError("Frames not extracted yet - run pre_extract_frames()")
    return np.load(FRAME_TIMESTAMPS_FILE)


# ----------------------------------------------------------------------
# C3D
# ----------------------------------------------------------------------
def load_c3d(path: Path = C3D_PATH) -> dict:
    """Parse a C3D file into a friendly dict.

    Returns
    -------
    dict with:
        labels     : list[str]            marker names (len N)
        xyz        : ndarray (N, 3, F)    positions, mm
        presence   : ndarray (N, F) bool  True where the marker is tracked
        fps        : float
        n_frames   : int
    """
    path = Path(path)
    c = ezc3d.c3d(str(path))

    labels = list(c["parameters"]["POINT"]["LABELS"]["value"])
    pts = c["data"]["points"]  # (4, N, F) x,y,z,residual
    # ezc3d returns (3, N, F); reorder to (N, 3, F) for sane indexing
    xyz = pts[:3].astype(np.float64).transpose(1, 0, 2)
    n, _, f = xyz.shape
    fps = float(c["parameters"]["POINT"]["RATE"]["value"][0])

    # presence: finite, non-zero coordinates (QTM writes 0 / NaN for gaps)
    fin = np.all(np.isfinite(xyz), axis=1)
    nz = np.all(xyz != 0.0, axis=1)
    presence = fin & nz

    return {
        "path": str(path),
        "labels": labels,
        "xyz": xyz,
        "presence": presence,
        "fps": fps,
        "n_frames": f,
        "n_markers": n,
    }


def mocap_time_of_frame(frame_idx: int, fps: float) -> float:
    """Mocap frame index -> time in seconds."""
    return frame_idx / fps
