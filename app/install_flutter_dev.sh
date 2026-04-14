#!/usr/bin/env bash
# =============================================================================
# TelePT Flutter Dev Environment – Install Script
# =============================================================================
# Sets up everything needed to build and run the Flutter app from this machine
# (useful when working directly on the GPU workstation via USB-connected phone).
#
# What this installs:
#   - conda env "tele_pt" (Python 3.11, Java 17)
#   - Flutter SDK 3.41.6 → ~/flutter
#   - Android SDK (platform-tools, build-tools 35, android-35, NDK 28, CMake)
#   - Conda activation hooks (auto-sets JAVA_HOME, ANDROID_SDK_ROOT, PATH)
#   - udev rules for Samsung (and common Android) devices
#
# Usage:
#   chmod +x install_flutter_dev.sh
#   bash install_flutter_dev.sh
#
# Override defaults via env vars before running:
#   CONDA_ENV=my_env FLUTTER_DIR=~/flutter ANDROID_SDK_DIR=~/android-sdk bash install_flutter_dev.sh
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONDA_ENV="${CONDA_ENV:-tele_pt}"
FLUTTER_DIR="${FLUTTER_DIR:-$HOME/flutter}"
ANDROID_SDK_DIR="${ANDROID_SDK_DIR:-$HOME/android-sdk}"
FLUTTER_VERSION="${FLUTTER_VERSION:-3.41.6}"
FLUTTER_CHANNEL="${FLUTTER_CHANNEL:-stable}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOBILE_DIR="$SCRIPT_DIR/mobile"

echo "========================================================"
echo "  TelePT Flutter Dev Environment Setup"
echo "========================================================"
echo "  Conda env   : $CONDA_ENV"
echo "  Flutter     : $FLUTTER_DIR  (v$FLUTTER_VERSION)"
echo "  Android SDK : $ANDROID_SDK_DIR"
echo "========================================================"

# ---------------------------------------------------------------------------
# 0. Prerequisites check
# ---------------------------------------------------------------------------
echo ""
echo "[0/7] Checking prerequisites..."

if ! command -v conda &>/dev/null; then
    echo "[ERROR] conda not found. Install Miniconda first:"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi

for tool in curl unzip git; do
    if ! command -v "$tool" &>/dev/null; then
        echo "[ERROR] '$tool' not found. Install it with: sudo apt-get install $tool"
        exit 1
    fi
done

echo "  Prerequisites OK."

# ---------------------------------------------------------------------------
# 1. Create conda env with Python 3.11 + OpenJDK 17
# ---------------------------------------------------------------------------
echo ""
echo "[1/7] Creating conda env '$CONDA_ENV' (Python 3.11 + OpenJDK 17)..."

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
    echo "  Env '$CONDA_ENV' already exists – skipping creation."
else
    conda create -y -n "$CONDA_ENV" python=3.11 openjdk=17 -c conda-forge
fi

conda activate "$CONDA_ENV"
echo "  Active Python : $(python --version)"
echo "  Active Java   : $(java -version 2>&1 | head -1)"

JAVA_HOME_PATH="$(conda env list | grep "^$CONDA_ENV " | awk '{print $NF}')"
if [ -z "$JAVA_HOME_PATH" ]; then
    JAVA_HOME_PATH="$HOME/anaconda3/envs/$CONDA_ENV"
fi
echo "  JAVA_HOME will be: $JAVA_HOME_PATH"

# ---------------------------------------------------------------------------
# 2. Add conda activation env-var hooks
# ---------------------------------------------------------------------------
echo ""
echo "[2/7] Writing conda activation hooks (env vars auto-set on activate)..."

ACT_DIR="$CONDA_PREFIX/etc/conda/activate.d"
DEACT_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
mkdir -p "$ACT_DIR" "$DEACT_DIR"

cat > "$ACT_DIR/telept_flutter.sh" <<ACTIVATE
#!/bin/bash
export JAVA_HOME="$JAVA_HOME_PATH"
export ANDROID_SDK_ROOT="$ANDROID_SDK_DIR"
export ANDROID_HOME="$ANDROID_SDK_DIR"
export PATH="$FLUTTER_DIR/bin:$ANDROID_SDK_DIR/platform-tools:$ANDROID_SDK_DIR/cmdline-tools/latest/bin:\$PATH"
ACTIVATE

cat > "$DEACT_DIR/telept_flutter.sh" <<DEACTIVATE
#!/bin/bash
unset JAVA_HOME
unset ANDROID_SDK_ROOT
unset ANDROID_HOME
DEACTIVATE

chmod +x "$ACT_DIR/telept_flutter.sh" "$DEACT_DIR/telept_flutter.sh"
echo "  Hooks written."

# Re-source env vars for remainder of this script
export JAVA_HOME="$JAVA_HOME_PATH"
export ANDROID_SDK_ROOT="$ANDROID_SDK_DIR"
export ANDROID_HOME="$ANDROID_SDK_DIR"
export PATH="$FLUTTER_DIR/bin:$ANDROID_SDK_DIR/platform-tools:$ANDROID_SDK_DIR/cmdline-tools/latest/bin:$PATH"

# ---------------------------------------------------------------------------
# 3. Install Flutter SDK
# ---------------------------------------------------------------------------
echo ""
echo "[3/7] Installing Flutter $FLUTTER_VERSION → $FLUTTER_DIR ..."

if [ -f "$FLUTTER_DIR/bin/flutter" ]; then
    INSTALLED_VER="$("$FLUTTER_DIR/bin/flutter" --version 2>/dev/null | head -1 | awk '{print $2}')"
    echo "  Flutter already installed: v$INSTALLED_VER – skipping download."
else
    FLUTTER_ARCHIVE="flutter_linux_${FLUTTER_VERSION}-${FLUTTER_CHANNEL}.tar.xz"
    FLUTTER_URL="https://storage.googleapis.com/flutter_infra_release/releases/${FLUTTER_CHANNEL}/linux/${FLUTTER_ARCHIVE}"

    echo "  Downloading $FLUTTER_ARCHIVE ..."
    TMP_TAR=$(mktemp /tmp/flutter_XXXX.tar.xz)
    curl -L --progress-bar "$FLUTTER_URL" -o "$TMP_TAR"

    echo "  Extracting to $HOME ..."
    mkdir -p "$HOME"
    tar -xf "$TMP_TAR" -C "$HOME"
    # Flutter archive extracts to ~/flutter by default; rename if needed
    if [ "$(realpath "$HOME/flutter")" != "$(realpath "$FLUTTER_DIR")" ]; then
        mv "$HOME/flutter" "$FLUTTER_DIR"
    fi
    rm -f "$TMP_TAR"
    echo "  Flutter extracted."
fi

echo "  Flutter: $("$FLUTTER_DIR/bin/flutter" --version 2>/dev/null | head -1)"

# ---------------------------------------------------------------------------
# 4. Install Android SDK
# ---------------------------------------------------------------------------
echo ""
echo "[4/7] Installing Android SDK → $ANDROID_SDK_DIR ..."

mkdir -p "$ANDROID_SDK_DIR/cmdline-tools"

# --- cmdline-tools (sdkmanager) ---
if [ ! -f "$ANDROID_SDK_DIR/cmdline-tools/latest/bin/sdkmanager" ]; then
    CMDTOOLS_URL="https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip"
    echo "  Downloading cmdline-tools..."
    TMP_ZIP=$(mktemp /tmp/cmdtools_XXXX.zip)
    curl -L --progress-bar "$CMDTOOLS_URL" -o "$TMP_ZIP"
    TMP_UNZIP=$(mktemp -d)
    unzip -q "$TMP_ZIP" -d "$TMP_UNZIP"
    mkdir -p "$ANDROID_SDK_DIR/cmdline-tools/latest"
    cp -r "$TMP_UNZIP/cmdline-tools/"* "$ANDROID_SDK_DIR/cmdline-tools/latest/"
    rm -rf "$TMP_ZIP" "$TMP_UNZIP"
    echo "  cmdline-tools installed."
else
    echo "  cmdline-tools already installed – skipping."
fi

SDKMANAGER="$ANDROID_SDK_DIR/cmdline-tools/latest/bin/sdkmanager"

# Accept all licenses non-interactively
yes | "$SDKMANAGER" --licenses > /dev/null 2>&1 || true

echo "  Installing SDK components..."
"$SDKMANAGER" \
    "platform-tools" \
    "build-tools;35.0.0" \
    "build-tools;28.0.3" \
    "platforms;android-35" \
    "ndk;28.2.13433566" \
    "cmake;3.22.1"

echo "  Android SDK components installed."

# ---------------------------------------------------------------------------
# 5. Write local.properties for the Flutter project
# ---------------------------------------------------------------------------
echo ""
echo "[5/7] Writing mobile/android/local.properties..."

cat > "$MOBILE_DIR/android/local.properties" <<PROPS
sdk.dir=$ANDROID_SDK_DIR
flutter.sdk=$FLUTTER_DIR
PROPS

echo "  local.properties updated."

# ---------------------------------------------------------------------------
# 6. Set up udev rules for Android devices (requires sudo)
# ---------------------------------------------------------------------------
echo ""
echo "[6/7] Setting up udev rules for Android USB debugging..."

UDEV_RULE_FILE="/etc/udev/rules.d/51-android.rules"

if [ -f "$UDEV_RULE_FILE" ]; then
    echo "  udev rules already exist at $UDEV_RULE_FILE – skipping."
else
    if ! sudo -n true 2>/dev/null; then
        echo "  [SKIP] No passwordless sudo. Run manually:"
        echo "    sudo bash $SCRIPT_DIR/setup_udev.sh"
    else
        sudo bash "$SCRIPT_DIR/setup_udev.sh"
        echo "  udev rules installed."
    fi
fi

# ---------------------------------------------------------------------------
# 7. Run flutter pub get
# ---------------------------------------------------------------------------
echo ""
echo "[7/7] Running flutter pub get in $MOBILE_DIR ..."

cd "$MOBILE_DIR"
"$FLUTTER_DIR/bin/flutter" pub get

echo ""
echo "========================================================"
echo "  Flutter dev environment setup complete!"
echo "========================================================"
echo ""
echo "Quick-start:"
echo ""
echo "  # Activate the env (sets JAVA_HOME, ANDROID_SDK_ROOT, etc.)"
echo "  conda activate $CONDA_ENV"
echo ""
echo "  # Check everything looks good"
echo "  flutter doctor"
echo ""
echo "  # Build + deploy to connected phone"
echo "  $SCRIPT_DIR/run_flutter.sh run"
echo ""
echo "  # Or run from mobile/ directly"
echo "  cd $MOBILE_DIR && flutter run"
echo ""
echo "If your phone is connected via USB, also run once:"
echo "  adb devices"
echo "(Should list your device. If empty, check USB Debugging is enabled.)"
