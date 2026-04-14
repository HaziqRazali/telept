#!/bin/bash
# Convenience wrapper to run Flutter commands inside the tele_pt conda environment.
# Usage:  ./run_flutter.sh [flutter arguments]
# Example: ./run_flutter.sh run
#          ./run_flutter.sh build apk --release
#          ./run_flutter.sh devices

export ANDROID_SDK_ROOT="/home/haziq/android-sdk"
export ANDROID_HOME="$ANDROID_SDK_ROOT"
export JAVA_HOME="/home/haziq/anaconda3/envs/tele_pt"
export PATH="/home/haziq/flutter/bin:$ANDROID_SDK_ROOT/platform-tools:$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$PATH"

cd "$(dirname "$0")/mobile"
flutter "$@"
