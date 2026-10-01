#!/usr/bin/env bash
set -euo pipefail

# Run a default-pose renderer in a usable local Conda environment.  The
# wrappers are intentionally self-bootstrapping so they work on hosts where
# the original mhr_new environment does not exist.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APPEARANCE="${1:-}"

if [[ -z "$APPEARANCE" ]]; then
    echo "Usage: _run_default_mhr.sh blue|reskinned" >&2
    exit 2
fi

if [[ "$APPEARANCE" != "blue" && "$APPEARANCE" != "reskinned" ]]; then
    echo "Unknown appearance: $APPEARANCE (expected blue or reskinned)" >&2
    exit 2
fi

declare -a CANDIDATES=()
if [[ -n "${MHR_PYTHON:-}" ]]; then
    CANDIDATES+=("$MHR_PYTHON")
fi

# Prefer the host's PyTorch environment, then the other project environments.
# The active base environment is included last; it is accepted only if it has
# PyTorch, so a bare base environment cannot accidentally be selected.
CONDA_BASE=""
if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi
if [[ -n "$CONDA_BASE" ]]; then
    for env_name in "${MHR_CONDA_ENV:-pytorch_env_cu128}" sam_3d_body mhr_new sapiens2; do
        CANDIDATES+=("$CONDA_BASE/envs/$env_name/bin/python")
    done
    CANDIDATES+=("$CONDA_BASE/bin/python")
fi
if command -v python >/dev/null 2>&1; then
    CANDIDATES+=("$(command -v python)")
fi

PYTHON=""
for candidate in "${CANDIDATES[@]}"; do
    # Resolve a command name supplied through MHR_PYTHON as well as a path.
    if [[ "$candidate" != */* ]]; then
        candidate="$(command -v "$candidate" 2>/dev/null || true)"
    fi
    [[ -x "$candidate" ]] || continue
    if "$candidate" -c 'import torch' >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done

if [[ -z "$PYTHON" ]]; then
    cat >&2 <<'EOF'
Could not find a Python environment containing PyTorch.
Available choices can be inspected with: conda env list
Set MHR_PYTHON=/path/to/python to select one explicitly.
EOF
    exit 1
fi

# Activate the selected environment inside this process.  Calling the
# environment's interpreter directly would also work, but activation keeps
# any native library paths supplied by Conda available to OpenGL/PyRender.
if [[ -n "$CONDA_BASE" && "$PYTHON" == "$CONDA_BASE"/*/bin/python ]]; then
    ENV_PREFIX="${PYTHON%/bin/python}"
    # shellcheck disable=SC1091
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$ENV_PREFIX"
    PYTHON="$(command -v python)"
fi

MISSING="$($PYTHON - <<'PY'
import importlib.util

required = {
    "numpy": "numpy",
    "cv2": "opencv-python-headless",
    "pyrender": "pyrender",
    "trimesh": "trimesh",
}
missing = [package for module, package in required.items()
           if importlib.util.find_spec(module) is None]
print(" ".join(missing))
PY
)"

if [[ -n "$MISSING" ]]; then
    echo "Installing missing renderer packages in: $CONDA_PREFIX" >&2
    "$PYTHON" -m pip install --disable-pip-version-check $MISSING
fi

exec "$PYTHON" "$SCRIPT_DIR/render_default_mhr.py" --appearance "$APPEARANCE"
