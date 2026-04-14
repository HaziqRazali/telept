"""Server configuration constants."""

import os

# Networking
HOST = os.getenv("SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("SERVER_PORT", "8000"))

# SAM3DBody paths (adjust to your environment)
SAM3D_ROOT = os.getenv("SAM3D_ROOT", os.path.expanduser("~/sam-3d-body"))
SAM3D_CHECKPOINT = os.getenv(
    "SAM3D_CHECKPOINT",
    os.path.join(SAM3D_ROOT, "checkpoints", "sam-3d-body-dinov3", "model.safetensors"),
)
SAM3D_MHR_PATH = os.getenv("SAM3D_MHR_PATH", "")
SAM3D_DETECTOR_NAME = os.getenv("SAM3D_DETECTOR_NAME", "rtmdet")
SAM3D_DETECTOR_PATH = os.getenv("SAM3D_DETECTOR_PATH", "")
SAM3D_SEGMENTOR_NAME = os.getenv("SAM3D_SEGMENTOR_NAME", "")
SAM3D_SEGMENTOR_PATH = os.getenv("SAM3D_SEGMENTOR_PATH", "")
SAM3D_FOV_NAME = os.getenv("SAM3D_FOV_NAME", "")
SAM3D_FOV_PATH = os.getenv("SAM3D_FOV_PATH", "")
SAM3D_BBOX_THRESH = float(os.getenv("SAM3D_BBOX_THRESH", "0.5"))

# Processing
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "500"))
TEMP_DIR = os.getenv("TEMP_DIR", "/tmp/sam3d_server")

# Whether to use the real SAM3DBody model (False = return rest-pose stub)
USE_SAM3D = os.getenv("USE_SAM3D", "0") == "1"
