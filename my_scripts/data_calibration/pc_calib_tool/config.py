"""Central configuration for the iPad<->Mocap calibration tool."""

import json
import os
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Input data (defaults; you can override them at runtime in the app's
# "0. Data" tab, which persists your choice to output/settings.json)
VIDEO_PATH = Path(
    "/home/haziq/datasets/telept/data/NUS/ipad/old/rgb_1787045535038.mp4"
)
C3D_PATH = Path(
    "/home/haziq/datasets/telept/data/NUS/mocap/old/mocap_rgb_calib_sync_real.c3d"
)

# Derived dirs (auto-created on import)
CACHE_DIR = PROJECT_ROOT / "cache"
FRAME_DIR = CACHE_DIR / "frames"
OUTPUT_DIR = PROJECT_ROOT / "output"
for _d in (CACHE_DIR, FRAME_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FRAME_META_FILE = CACHE_DIR / "frame_meta.json"
FRAME_TIMESTAMPS_FILE = CACHE_DIR / "timestamps.npy"

# ----------------------------------------------------------------------
# Chessboard (calib.io generated: 9x6 squares, 40 mm)
# ----------------------------------------------------------------------
SQUARE_SIZE_MM = 40.0
BOARD_INNER_CORNERS = (8, 5)   # (cols, rows) of inner corners -> 9x6 squares
BOARD_MARGIN_SQUARES = 2       # extra grid dots beyond the pattern, per side

# ----------------------------------------------------------------------
# Reflective markers on the board (mocap labels)
# ----------------------------------------------------------------------
MARKER_NAMES = ["Board1", "Board2", "Board3", "Board4", "Board5", "Board6"]
NUM_MARKERS = len(MARKER_NAMES)

# ----------------------------------------------------------------------
# Output files
# ----------------------------------------------------------------------
INTRINSICS_FILE = OUTPUT_DIR / "intrinsics.json"
MARKERS_FILE = OUTPUT_DIR / "markers.json"
SYNC_FILE = OUTPUT_DIR / "sync.json"
TRIM_FILE = OUTPUT_DIR / "trim.json"
TRANSFORM_FILE = OUTPUT_DIR / "transform.json"

# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
GRADIO_HOST = "0.0.0.0"
# Overridable so you can run on a fresh port if the default is stuck:
#   CALIB_TOOL_PORT=7861 python3 app.py
GRADIO_PORT = int(os.environ.get("CALIB_TOOL_PORT", "7860"))

# ----------------------------------------------------------------------
# Persisted user paths (set in the app's "0. Data" tab)
# ----------------------------------------------------------------------
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
