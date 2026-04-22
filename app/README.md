# TelePT – 3D Body Capture App

A cross-platform mobile app (Flutter) and local GPU server (Python/FastAPI) for recording video, running 3D body reconstruction via [SAM3DBody](https://github.com/microsoft/sam-3d-body), and viewing the resulting mesh sequence on your phone/tablet.

```
app/
├── install_flutter_dev.sh   # One-shot Flutter dev environment install
├── run_flutter.sh           # Wrapper: builds/runs the app with correct env vars
├── setup_udev.sh            # One-time USB udev rules for Android devices (run w/ sudo)
│
├── mobile/                  # Flutter app (Android + iOS/iPadOS)
│   ├── lib/
│   │   ├── main.dart                  # App entry point
│   │   ├── config.dart                # Server URL & constants  ← edit this
│   │   ├── models/
│   │   │   ├── mesh_frame.dart        # Parsed mesh data model
│   │   │   └── mesh_meta.dart         # Server response metadata
│   │   ├── screens/
│   │   │   ├── home_screen.dart       # Record / pick video UI
│   │   │   └── viewer_screen.dart     # Split-view: video left, 3D mesh right
│   │   ├── services/
│   │   │   ├── api_service.dart       # Upload video, parse ZIP response
│   │   │   └── obj_parser.dart        # Wavefront OBJ parser
│   │   └── widgets/
│   │       └── mesh_viewer.dart       # 3D mesh renderer (Canvas + vector_math)
│   ├── android/
│   ├── ios/
│   └── pubspec.yaml
│
└── server/                  # Python GPU server
    ├── install_gpu_server.sh  # One-shot GPU server install
    ├── main.py                # FastAPI app with POST /process
    ├── config.py              # Server configuration
    ├── mesh_gen.py            # Mesh generation (stub + SAM3DBody)
    └── requirements.txt
```

---

## Quick-Start (existing setup on this machine)

> The workstation already has Flutter 3.41.6, Android SDK, and the `tele_pt` conda env installed.
> For a fresh machine see the full installation sections below.

```bash
# 1. Activate the dev environment (sets JAVA_HOME, ANDROID_SDK_ROOT, etc.)
conda activate tele_pt

# 2. Start the app on the connected phone (USB)
/home/haziq/telept/app/run_flutter.sh run

# 3a. In a separate terminal, start the stub server (no GPU needed)
conda activate telept_server
cd /home/haziq/telept/app/server
uvicorn main:app --host 0.0.0.0 --port 8000

# 3b. OR start with real SAM3DBody (GPU required)
conda activate telept_server
cd /home/haziq/telept/app/server
USE_SAM3D=1 \
SAM3D_CHECKPOINT=/home/haziq/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt \
SAM3D_MHR_PATH=/home/haziq/MHR/assets/mhr_model.pt \
SAM3D_DETECTOR_NAME=yolo_pose \
uvicorn main:app --host 0.0.0.0 --port 8000
```

# kill 
lsof -ti:8000 | xargs kill -9

---

## Setting up on a fresh machine

These steps cover everything: running the GPU server **and** building/deploying the Flutter app to a connected phone.

### Step 1 — Clone the repo

```bash
git clone https://github.com/HaziqRazali/telept.git
cd telept/app
```

### Step 2 — Install the Flutter + Android dev environment

Creates the `tele_pt` conda env (Python 3.11 + Java 17), downloads Flutter 3.41.6, Android SDK, and sets up all env vars automatically on `conda activate`.

```bash
bash install_flutter_dev.sh
```

### Step 3 — Install the GPU server environment

Creates the `telept_server` conda env with PyTorch (CUDA), SAM3DBody, FastAPI, OpenCV, etc.

```bash
bash server/install_gpu_server.sh
```

> For CUDA 12.x: `CUDA_VERSION=121 bash server/install_gpu_server.sh`

### Step 4 — Set up USB udev rules (one-time, requires sudo)

So Linux recognises the Android phone over USB.

```bash
sudo bash setup_udev.sh
```

### Step 5 — Enable USB Debugging on the phone

**Settings → About phone → tap Build number 7×** → back → **Developer Options → USB Debugging → ON**

Plug in the phone, then verify:
```bash
conda activate tele_pt
adb devices   # should list your phone's serial number
# If empty: accept the "Allow USB debugging?" dialog on the phone screen
```

### Step 6 — Deploy the app to the phone

```bash
conda activate tele_pt
./run_flutter.sh run
# Builds, installs, and launches the app on the connected phone
```

### Step 7 — Start the server

Open a second terminal:

```bash
# Stub mode (no GPU needed)
conda activate telept_server
cd /home/haziq/telept/app/server
uvicorn main:app --host 0.0.0.0 --port 8000

# Real SAM3DBody mode (GPU required)
conda activate telept_server
cd /home/haziq/telept/app/server
USE_SAM3D=1 \
SAM3D_CHECKPOINT=/home/haziq/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt \
SAM3D_MHR_PATH=/home/haziq/MHR/assets/mhr_model.pt \
uvicorn main:app --host 0.0.0.0 --port 8000
# Listening on http://0.0.0.0:8000
```

Find the IP to enter in the app:
```bash
hostname -I | awk '{print $1}'
```

### Step 8 — Configure the server URL in the app

On the phone, tap the **⚙ gear icon** (top-right of home screen) → enter `http://<server-ip>:8000` → **Save**.
This is persisted — you only need to do it once.

---

> **Steps 2, 3, 4 are one-time only.** After that, daily use is just Steps 6 and 7.

---

## Server IP — do you need to know it?

**Short answer: yes, once.** Both devices must be on the same Wi-Fi. You do *not* need to recompile the app — the server URL is configurable at runtime:

1. Open the app → tap the **⚙ settings icon** (top-right of home screen).
2. Enter your server's IP, e.g. `http://192.168.1.68:8000`.
3. Tap **Save**. The setting is persisted — you only need to do this once per network.

To find the server IP:
```bash
hostname -I | awk '{print $1}'
```

The app will also work on any GPU server as long as:
- It is on the **same local Wi-Fi subnet** as the phone.
- Port **8000** is not blocked by a firewall (`sudo ufw allow 8000` if needed).
- You enter the correct IP in the app settings.

---

## 1.  Flutter Dev Environment Setup

> **This also creates the `tele_pt` conda environment** — the same env used to run `run_flutter.sh`.
> If you have already run this script on the current machine, skip to [Configure the server URL](#configure-the-server-url).

### Automated install (recommended)

Run once on any Linux machine where you want to develop the Flutter app:

```bash
chmod +x /data/telept/app/install_flutter_dev.sh
bash /data/telept/app/install_flutter_dev.sh
```

This script:
- Creates a `tele_pt` conda env with **Python 3.11** + **OpenJDK 17**
- Downloads **Flutter 3.41.6** (stable) → `~/flutter`
- Downloads and installs the **Android SDK** → `~/android-sdk`
  - platform-tools (adb), build-tools 35.0.0, platforms android-35 & android-36, NDK 28.2, CMake 3.22.1
  - Installs `clang` and `ninja-build` (needed for Flutter Linux toolchain) if missing
- Writes conda activation hooks so `JAVA_HOME`, `ANDROID_SDK_ROOT`, and `PATH` are set automatically
- Writes `mobile/android/local.properties` pointing at the SDK and Flutter installs
- Runs `flutter pub get` to fetch all Dart dependencies
- Attempts to install udev rules for Android USB debugging (requires sudo)

Override any path via env vars before running:
```bash
CONDA_ENV=my_env \
FLUTTER_DIR=~/flutter \
ANDROID_SDK_DIR=~/android-sdk \
FLUTTER_VERSION=3.41.6 \
bash install_flutter_dev.sh
```

### Manual steps (reference)

<details>
<summary>Expand for manual install commands</summary>

```bash
# 1. Conda env
conda create -y -n tele_pt python=3.11 openjdk=17 -c conda-forge
conda activate tele_pt

# 2. Flutter
curl -LO https://storage.googleapis.com/flutter_infra_release/releases/stable/linux/flutter_linux_3.41.6-stable.tar.xz
tar -xf flutter_linux_3.41.6-stable.tar.xz -C ~
export PATH="$HOME/flutter/bin:$PATH"

# 3. Android cmdline-tools
mkdir -p ~/android-sdk/cmdline-tools
curl -LO https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip
unzip commandlinetools-linux-11076708_latest.zip -d /tmp/cmdtools
mv /tmp/cmdtools/cmdline-tools ~/android-sdk/cmdline-tools/latest

# 4. Android SDK components
export ANDROID_SDK_ROOT=~/android-sdk
export PATH="$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$ANDROID_SDK_ROOT/platform-tools:$PATH"
yes | sdkmanager --licenses
sdkmanager "platform-tools" "build-tools;35.0.0" "build-tools;28.0.3" "platforms;android-35" "platforms;android-36" "ndk;28.2.13676358" "cmake;3.22.1"

# linux toolchain deps (for flutter doctor)
sudo apt-get install -y clang ninja-build

# 5. local.properties
cat > ~/telept/app/mobile/android/local.properties <<EOF
sdk.dir=$HOME/android-sdk
flutter.sdk=$HOME/flutter
EOF

# 6. udev rules for Android USB debugging (run once with sudo)
sudo bash /data/telept/app/setup_udev.sh

# 7. Dependencies
cd /data/telept/app/mobile && flutter pub get
```
</details>

### Verify the install

```bash
conda activate tele_pt
flutter doctor
```

Expected output:
```
[✓] Flutter (Channel stable, 3.41.6)
[✓] Android toolchain - develop for Android devices
[✓] Linux toolchain - develop for Linux desktop
[✓] Connected device (1 available)        ← phone via USB
```

### Configure the server URL

The server URL is set **inside the app** — no need to edit files or recompile.
Tap the **⚙ gear icon** on the home screen, enter the GPU server IP, and tap **Save**.
The value is persisted across app restarts.

If you want to change the compile-time default, edit `mobile/lib/config.dart`:

```dart
static const String defaultServerUrl = 'http://192.168.1.68:8000';
```

Find your server's IP:
```bash
hostname -I | awk '{print $1}'
```

### Run the app on a connected phone

```bash
# Using the convenience wrapper (recommended):
/data/telept/app/run_flutter.sh run

# Or manually from mobile/:
conda activate tele_pt
cd /data/telept/app/mobile
flutter run
```

The wrapper (`run_flutter.sh`) sets all required env vars (`JAVA_HOME`, `ANDROID_SDK_ROOT`, `ANDROID_HOME`, `PATH`) so it works from any shell, even without the conda env active.

### Other Flutter commands

```bash
# List connected devices
/data/telept/app/run_flutter.sh devices

# Hot reload (while flutter run is already running → press r in that terminal)
# Hot restart (press R)

# Build release APK
/data/telept/app/run_flutter.sh build apk --release
# → mobile/build/app/outputs/flutter-apk/app-release.apk

# Analyze for issues
/data/telept/app/run_flutter.sh analyze
```

### Android USB debugging

Enable on the phone: **Settings → About phone → tap Build number 7×** → **Developer Options → USB Debugging → ON**

Then on the workstation:
```bash
adb devices   # should list the device serial
# If empty, accept the "Allow USB debugging?" prompt on the phone
```

If `adb` is not found, make sure the `tele_pt` conda env is active (it adds `platform-tools` to `PATH`).

---

## 2.  GPU Server Setup & Run

### Automated install (recommended)

Run once on the GPU machine:

```bash
chmod +x /data/telept/app/server/install_gpu_server.sh
bash /data/telept/app/server/install_gpu_server.sh
```

This creates a `telept_server` conda env and installs:
- PyTorch (CUDA 11.8 by default — change `CUDA_VERSION=121` for CUDA 12.1)
- All SAM3DBody + MHR dependencies
- FastAPI, Uvicorn, OpenCV, trimesh

Override via env vars:
```bash
CONDA_ENV=telept_server \
SAM3D_DIR=~/sam-3d-body \
CUDA_VERSION=121 \
bash server/install_gpu_server.sh
```

### Run (stub mode — no GPU required)

The server returns a rest-pose icosphere for every frame by default. This lets you test the full pipeline without a GPU:

```bash
conda activate telept_server
cd /home/haziq/telept/app/server
uvicorn main:app --host 0.0.0.0 --port 8000
```

Verify:
```bash
curl http://localhost:8000/health
# {"status":"ok","use_sam3d":false}
```

### Run (real SAM3DBody mode)

When SAM3DBody is installed and you have a GPU:

1. In `server/mesh_gen.py` — **uncomment** the entire `SAM3DBody REAL PROCESSING` section at the bottom.
2. In `server/main.py` — **uncomment** the `from mesh_gen import generate_sam3d_meshes` import line.
3. In `server/main.py` — replace the `raise HTTPException(status_code=501 ...)` with `zip_path, n_frames, fps = generate_sam3d_meshes(video_path, work_dir)`.
4. Run:

```bash
conda activate telept_server
cd /home/haziq/telept/app/server
USE_SAM3D=1 \
SAM3D_CHECKPOINT=/home/haziq/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt \
SAM3D_MHR_PATH=/home/haziq/MHR/assets/mhr_model.pt \
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Server configuration

All settings via environment variables (`config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `8000` | HTTP port |
| `USE_SAM3D` | `0` | Set to `1` to enable real SAM3DBody |
| `MAX_UPLOAD_SIZE_MB` | `500` | Max upload size |
| `TEMP_DIR` | `/tmp/sam3d_server` | Temp directory for processing |
| `SAM3D_ROOT` | `~/sam-3d-body` | SAM3DBody repo root |
| `SAM3D_CHECKPOINT` | auto | Model checkpoint path |
| `SAM3D_MHR_PATH` | `~/MHR` | MHR model directory |

---

## 3.  App Usage

1. Start the server on your workstation (see above).
2. Connect phone to the **same Wi-Fi** as the workstation.
3. Open the app → tap **Record Video** (or **Pick from Gallery**).
4. The app uploads the video → server processes it → returns a ZIP of `.obj` meshes.
5. The viewer opens:
   - **Left panel**: original video playback
   - **Right panel**: 3D mesh viewer (orbit with finger-drag, zoom with pinch)
   - **Bottom bar**: play/pause + scrub slider

---

## API Reference

### `GET /health`

```json
{"status": "ok", "use_sam3d": false}
```

### `POST /process`

Upload a video (multipart form, field `video`).

**Response:** `application/zip` containing:
- `frame_0000.obj`, `frame_0001.obj`, ... (one OBJ per frame)
- `meta.json`: `{"frame_count": 120, "fps": 30.0}`

**Error codes:** `400` bad type · `413` too large · `500` processing error · `501` SAM3D not enabled

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `flutter doctor` shows Android SDK missing | Check `local.properties` has correct `sdk.dir` |
| `adb devices` shows nothing | Enable USB Debugging on phone; accept the dialog; run `setup_udev.sh` |
| App can't reach server | Same Wi-Fi; check `config.dart` IP; check firewall on port 8000 |
| Android cleartext HTTP blocked | `android:usesCleartextTraffic="true"` is already set in `AndroidManifest.xml` |
| iOS ATS blocks HTTP | `NSAllowsLocalNetworking` is already set in `Info.plist` |
| Server `ModuleNotFoundError: cv2` | `pip install opencv-python-headless` |
| Server OOM on GPU | SAM3DBody needs ~8 GB VRAM; reduce video resolution or length |
| `GestureDetector` conflict crash | Fixed — unified `onScaleStart/Update/End` in `mesh_viewer.dart` |
