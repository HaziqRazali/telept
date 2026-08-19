"""Central configuration for the iPad<->Mocap calibration tool."""

from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

# Input data
VIDEO_PATH = Path(
    "/home/haziq/datasets/telept/data/NUS/ipad/rgb_1787045535038.mp4"
)
C3D_PATH = Path(
    "/home/haziq/datasets/telept/data/NUS/mocap/mocap_rgb_calib_sync01.c3d"
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
GRADIO_PORT = 7860
