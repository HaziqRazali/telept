#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${MHR_PYTHON:-/home/haziq/anaconda3/envs/mhr_new/bin/python}"

exec "$PYTHON" "$SCRIPT_DIR/render_default_mhr.py" --appearance blue
