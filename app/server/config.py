"""Server configuration constants."""

import os

# Networking
HOST = os.getenv("SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("SERVER_PORT", "8000"))

# SAM3DBody paths
# Fast-SAM-3D-Body repo provides the accelerated sam_3d_body package;
# checkpoints are loaded from the original sam-3d-body location.
SAM3D_FAST_ROOT = os.getenv("SAM3D_FAST_ROOT", os.path.expanduser("~/Fast-SAM-3D-Body"))
SAM3D_ROOT = os.getenv("SAM3D_ROOT", os.path.expanduser("~/sam-3d-body"))
SAM3D_CHECKPOINT = os.getenv(
    "SAM3D_CHECKPOINT",
    os.path.join(SAM3D_ROOT, "checkpoints", "sam-3d-body-dinov3", "model.ckpt"),
)
SAM3D_MHR_PATH = os.getenv("SAM3D_MHR_PATH", os.path.expanduser("~/MHR/assets/mhr_model.pt"))
# YOLO-Pose detector (auto-downloaded from ultralytics hub on first run)
# Will be compiled to .engine by build_trt_engines.sh for extra ~2x speedup
SAM3D_DETECTOR_NAME = os.getenv("SAM3D_DETECTOR_NAME", "vitdet")
SAM3D_DETECTOR_PATH = os.getenv("SAM3D_DETECTOR_PATH", "")
SAM3D_YOLO_MODEL = os.getenv(
    "SAM3D_YOLO_MODEL",
    os.path.join(SAM3D_FAST_ROOT, "checkpoints", "yolo", "yolo11m-pose.pt"),
)
SAM3D_SEGMENTOR_NAME = os.getenv("SAM3D_SEGMENTOR_NAME", "")
SAM3D_SEGMENTOR_PATH = os.getenv("SAM3D_SEGMENTOR_PATH", "")
# MoGe2 FOV estimator – model-s (35M) auto-downloaded from HuggingFace
SAM3D_FOV_NAME = os.getenv("SAM3D_FOV_NAME", "moge2")
SAM3D_FOV_PATH = os.getenv("SAM3D_FOV_PATH", "")
SAM3D_BBOX_THRESH = float(os.getenv("SAM3D_BBOX_THRESH", "0.5"))

# Processing
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/sam3d_server")

# Whether to use the real SAM3DBody model (False = return rest-pose stub)
USE_SAM3D = os.getenv("USE_SAM3D", "0") == "1"

# Optional shared secret for simple API key auth.
# Set the API_KEY environment variable on the server to enable it.
# The Flutter app must then be configured with the same key in Settings.
# Leave unset (or empty) to disable auth (default: open, local-network only).
API_KEY = os.getenv("API_KEY", "")
