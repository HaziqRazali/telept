"""Stage D: shared start/end trim on the synchronized time axis.

Trim is specified in VIDEO time (seconds); the mocap range is derived via the
sync offset, so both streams cover the same physical duration.
"""

from __future__ import annotations

import json

import numpy as np

from config import TRIM_FILE
from sync import mocap_index_for_video_time


def apply_trim(
    video_times: np.ndarray,
    mocap: dict,
    offset: float,
    start_s: float,
    end_s: float,
) -> dict:
    """Compute video + mocap frame ranges for a trim window (video seconds)."""
    t0, t1 = min(start_s, end_s), max(start_s, end_s)
    v_idx = np.where((video_times >= t0) & (video_times <= t1))[0]

    m0 = mocap_index_for_video_time(t0, offset, mocap["fps"])
    m1 = mocap_index_for_video_time(t1, offset, mocap["fps"])
    m0 = max(0, min(m0, mocap["n_frames"] - 1))
    m1 = max(0, min(m1, mocap["n_frames"] - 1))
    m_idx = np.arange(min(m0, m1), max(m0, m1) + 1)

    return {
        "start_s": float(t0),
        "end_s": float(t1),
        "duration_s": float(t1 - t0),
        "video_frames": int(len(v_idx)),
        "video_first": int(v_idx[0]) if len(v_idx) else None,
        "video_last": int(v_idx[-1]) if len(v_idx) else None,
        "mocap_frames": int(len(m_idx)),
        "mocap_first": int(m_idx[0]) if len(m_idx) else None,
        "mocap_last": int(m_idx[-1]) if len(m_idx) else None,
        "valid": bool(
            t1 > t0
            and len(v_idx) > 1
            and len(m_idx) > 1
            and m0 >= 0
            and m1 < mocap["n_frames"]
        ),
    }


def save_trim(start_s: float, end_s: float) -> None:
    TRIM_FILE.write_text(json.dumps(
        {"start_s": float(start_s), "end_s": float(end_s)}, indent=2))


def load_trim() -> dict | None:
    if not TRIM_FILE.exists():
        return None
    return json.loads(TRIM_FILE.read_text())
