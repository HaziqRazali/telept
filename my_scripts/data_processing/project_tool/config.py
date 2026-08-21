"""Configuration for the mocap->RGB projection tool.

Loads a video + C3D recorded in the SAME session as the calibration, so the
calibration parameters saved by pc_calib_tool (output/intrinsics.json and
output/transform.json) apply directly.  This tool never calibrates - it just
projects the mocap markers onto the synchronized video and lets you scrub.
"""

import json
import os
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Input data (defaults; overridable in the app's "0. Data" tab, persisted to
# output/settings.json)
VIDEO_PATH = Path(
    "/home/haziq/datasets/telept/data/NUS/ipad/rgb_1787047433275.mp4"
)
C3D_PATH = Path(
    "/home/haziq/datasets/telept/data/NUS/mocap/haziq_upperlimb_right2 dynamic 03.c3d"
)

# Calibration parameters come from the calibration tool's output (same-day
# session).  Point these wherever your saved calibrations live.
CALIB_ROOT = PROJECT_ROOT.parent / "pc_calib_tool" / "output"
INTRINSICS_FILE = CALIB_ROOT / "intrinsics.json"
TRANSFORM_FILE = CALIB_ROOT / "transform.json"
MARKERS_FILE = CALIB_ROOT / "markers.json"   # (unused by this tool)

# Derived dirs (auto-created on import)
CACHE_DIR = PROJECT_ROOT / "cache"
FRAME_DIR = CACHE_DIR / "frames"
OUTPUT_DIR = PROJECT_ROOT / "output"
for _d in (CACHE_DIR, FRAME_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FRAME_META_FILE = CACHE_DIR / "frame_meta.json"
FRAME_TIMESTAMPS_FILE = CACHE_DIR / "timestamps.npy"

# ----------------------------------------------------------------------
# Chessboard (kept for module compatibility; not used by projection)
# ----------------------------------------------------------------------
SQUARE_SIZE_MM = 40.0
BOARD_INNER_CORNERS = (8, 5)   # (cols, rows) of inner corners -> 9x6 squares
BOARD_MARGIN_SQUARES = 2

# ----------------------------------------------------------------------
# Marker names (used for coloring in the mocap 3D/2D renders)
# ----------------------------------------------------------------------
MARKER_NAMES = ["Board1", "Board2", "Board3", "Board4", "Board5", "Board6"]
NUM_MARKERS = len(MARKER_NAMES)

# ----------------------------------------------------------------------
# This tool's own outputs (sync + trim, per recording)
# ----------------------------------------------------------------------
SYNC_FILE = OUTPUT_DIR / "sync.json"
TRIM_FILE = OUTPUT_DIR / "trim.json"
SETTINGS_FILE = OUTPUT_DIR / "settings.json"


def load_settings() -> dict:
    """Return saved {video_path, c3d_path, mocap_fps_override} (or {})."""
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except (ValueError, OSError):
            pass
    return {}


def save_settings(video_path, c3d_path, mocap_fps_override=None) -> None:
    """Persist the video/c3d paths (and optional fps override) used by the GUI."""
    SETTINGS_FILE.write_text(json.dumps({
        "video_path": str(video_path),
        "c3d_path": str(c3d_path),
        "mocap_fps_override": mocap_fps_override if mocap_fps_override else None,
    }, indent=2))
