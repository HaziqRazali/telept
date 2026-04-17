#!/bin/bash
# Convenience wrapper to run Flutter commands inside the tele_pt conda environment.
# Usage:  ./run_flutter.sh [flutter arguments]
# Example: ./run_flutter.sh run
#          ./run_flutter.sh build apk --release
#          ./run_flutter.sh devices

# =============================================================================
# Configuration – edit these to match your machine
# =============================================================================
CONDA_ENV="${CONDA_ENV:-tele_pt}"
FLUTTER_DIR="${FLUTTER_DIR:-$HOME/flutter}"
ANDROID_SDK_DIR="${ANDROID_SDK_DIR:-$HOME/android-sdk}"
# Conda env prefix (auto-detected; override if needed)
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-$HOME/anaconda3/envs/$CONDA_ENV}"
# =============================================================================

export ANDROID_SDK_ROOT="$ANDROID_SDK_DIR"
export ANDROID_HOME="$ANDROID_SDK_DIR"
export JAVA_HOME="$CONDA_ENV_PREFIX"
export PATH="$FLUTTER_DIR/bin:$ANDROID_SDK_DIR/platform-tools:$ANDROID_SDK_DIR/cmdline-tools/latest/bin:$PATH"

cd "$(dirname "$0")/mobile"
flutter "$@"
