#!/usr/bin/env bash
# =============================================================================
# TelePT GPU Server – Install Script
# =============================================================================
# Run this once on your GPU machine to set up the Python environment and all
# dependencies needed to serve SAM3DBody mesh predictions.
#
# Usage:
#   chmod +x install_gpu_server.sh
#   bash install_gpu_server.sh
#
# Requirements:
#   - conda (Miniconda or Anaconda)
#   - NVIDIA GPU + CUDA 11.8 or 12.x drivers
#   - The sam-3d-body repo already cloned at ~/sam-3d-body (or set SAM3D_DIR)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration – override via env vars before running
# ---------------------------------------------------------------------------
ENV_NAME="${CONDA_ENV:-telept_server}"
PYTHON_VER="${PYTHON_VERSION:-3.10}"
SAM3D_DIR="${SAM3D_DIR:-$HOME/sam-3d-body}"
CUDA_VERSION="${CUDA_VERSION:-118}"   # 118 = CUDA 11.8 | 121 = CUDA 12.1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================"
echo "  TelePT GPU Server – Environment Setup"
echo "========================================================"
echo "  Conda env  : $ENV_NAME"
echo "  Python     : $PYTHON_VER"
echo "  SAM3D dir  : $SAM3D_DIR"
echo "  CUDA index : cu${CUDA_VERSION}"
echo "========================================================"

# ---------------------------------------------------------------------------
# 1. Verify prerequisites
# ---------------------------------------------------------------------------
if ! command -v conda &> /dev/null; then
    echo "[ERROR] conda not found. Install Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if [ ! -d "$SAM3D_DIR" ]; then
    echo "[ERROR] SAM3DBody repo not found at $SAM3D_DIR"
    echo "  Clone it with:"
    echo "    git clone https://github.com/microsoft/sam-3d-body.git ~/sam-3d-body"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Create / activate conda environment
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Creating conda environment '$ENV_NAME' (Python $PYTHON_VER)..."
conda create -y -n "$ENV_NAME" python="$PYTHON_VER" || true

# Source conda so we can activate inside the script
# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "[1/6] Active Python: $(python --version) at $(which python)"

# ---------------------------------------------------------------------------
# 3. Install PyTorch (GPU)
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Installing PyTorch (cu${CUDA_VERSION})..."
pip install --upgrade pip wheel

pip install \
    torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"

python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ---------------------------------------------------------------------------
# 4. Install SAM3DBody dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Installing SAM3DBody dependencies..."

cd "$SAM3D_DIR"

# Install from the repo's own requirements if present
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi
if [ -f "requirements_extra.txt" ]; then
    pip install -r requirements_extra.txt
fi

# Ensure the package itself is importable
pip install -e . 2>/dev/null || pip install . 2>/dev/null || true

cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# 5. Install MHR / body model dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Installing MHR + body model dependencies..."

pip install smplx pyrender pyopengl
# chumpy's setup.py does "import pip" which breaks in isolated build envs;
# --no-build-isolation bypasses that.
pip install --no-build-isolation chumpy

# ---------------------------------------------------------------------------
# 6. Install server-specific dependencies
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Installing FastAPI server dependencies..."

pip install \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "python-multipart>=0.0.9" \
    "numpy>=1.24.0" \
    "trimesh>=4.0.0" \
    "opencv-python-headless>=4.8.0"

# ---------------------------------------------------------------------------
# 7. Verify installation
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Verifying installation..."

python - <<'PYCHECK'
import sys
errors = []

try:
    import torch
    print(f"  [OK] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except ImportError as e:
    errors.append(f"  [FAIL] torch: {e}")

try:
    import cv2
    print(f"  [OK] opencv {cv2.__version__}")
except ImportError as e:
    errors.append(f"  [FAIL] cv2: {e}")

try:
    import fastapi
    print(f"  [OK] fastapi {fastapi.__version__}")
except ImportError as e:
    errors.append(f"  [FAIL] fastapi: {e}")

try:
    import trimesh
    print(f"  [OK] trimesh {trimesh.__version__}")
except ImportError as e:
    errors.append(f"  [FAIL] trimesh: {e}")

try:
    import numpy as np
    print(f"  [OK] numpy {np.__version__}")
except ImportError as e:
    errors.append(f"  [FAIL] numpy: {e}")

try:
    import sys, os
    sam3d_dir = os.path.expanduser("~/sam-3d-body")
    sys.path.insert(0, sam3d_dir)
    from sam_3d_body import load_sam_3d_body
    print(f"  [OK] sam_3d_body importable from {sam3d_dir}")
except ImportError as e:
    errors.append(f"  [FAIL] sam_3d_body: {e}")

if errors:
    print("\nSome packages failed to import:")
    for err in errors:
        print(err)
    sys.exit(1)
else:
    print("\nAll packages verified OK.")
PYCHECK

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "========================================================"
echo "  Installation complete!"
echo "========================================================"
echo ""
echo "To enable real SAM3DBody processing:"
echo "  1. Edit server/mesh_gen.py  – uncomment the SAM3DBody block"
echo "  2. Edit server/main.py      – uncomment generate_sam3d_meshes import"
echo ""
echo "To start the server:"
echo "  conda activate $ENV_NAME"
echo "  cd $(realpath "$SCRIPT_DIR")"
echo "  USE_SAM3D=1 SAM3D_CHECKPOINT=<path_to_checkpoint> python main.py"
echo ""
echo "Server will listen on http://0.0.0.0:8000"
echo "Update the app's config.dart with your GPU server IP."
