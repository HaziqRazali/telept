# iPad ↔ Mocap Calibration Tool — PC (native desktop) version

Calibrates an iPad camera to an already-calibrated motion-capture system
(Vicon/QTM-style) using a 40 mm 9×6 chessboard (8×5 inner corners) with 6
reflective markers (`Board1`–`Board6`) placed on a 40 mm grid *outside* the
pattern, synchronized by a flashing light bulb.

Output: the rigid transform **mocap → camera** (`P_camera = R @ P_mocap + t`,
mm) plus the camera intrinsics.

This is the **native desktop** sibling of `../web_calib_tool/`.  It has the
exact same 5-stage workflow and writes the same `output/*.json` files, but it
runs as a local **PySide6 (Qt6) window instead of a Gradio web app** — there
is no browser and no server round-trip, so **video scrubbing is instant**
(frames are preloaded into RAM and blitted straight to the widget).

## Why a PC version?

The Gradio web app was laggy when scrubbing through the video (every scrub
goes server → browser → image encoding → network).  This version renders
everything in-process on the machine with the monitor:

* scrub slider updates the view in the same event loop — no network, no delay
* click *directly on the video* to draw the LED ROI box (2 clicks)
* **middle-mouse scrollwheel** over the video zooms in/out **around the cursor**
  (1.2× per notch; the *Video zoom* spin does the same, recentred)
* every image pane scales to **fit** its widget — the full RGB frame is always
  visible, never clipped
* click the video scrub slider, then use **arrow keys** for frame-exact
  scrubbing (`PageUp`/`PageDown` = ±10 frames)

## Requirements

Python 3.10+ and a dedicated virtualenv.  The stack is:

* **PySide6** — the maintained Qt6 binding (Qt 5 / PyQt5 are EOL); renders
  in-process for 0-delay scrubbing
* **opencv-python-headless** — OpenCV without its bundled Qt.  Do **not** use
  the regular `opencv-python` here: it sets `QT_QPA_PLATFORM_PLUGIN_PATH` to
  its own `cv2/qt/plugins`, which makes PySide6 abort with
  `Could not load the Qt platform plugin "xcb"`.
* `numpy`, `scipy`, `matplotlib`, `ezc3d`

Set it up once (this already exists as `.venv/` on this machine):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
.venv/bin/python app.py          # or:  source .venv/bin/activate && python app.py
```

A window opens (must be on a machine with a display).  Set the iPad video +
C3D paths in **tab 0**, click *Load data*.  If the saved/default paths exist
on this machine they are auto-loaded on start.

## Workflow (5 tabs, in order)

0. **Data** — Set the paths to your iPad video and C3D mocap file (pre-filled
   from `config.py` / `output/settings.json`), click *Load data*.  If the
   C3D's stored rate is wrong (e.g. mocap comes out SHORTER than the video
   even though it was started first), set **Mocap FPS override** here and
   reload.
1. **Marker layout** — Click the 6 orange dots on the board schematic where
   the reflective markers sit (grid = 40 mm).  Click order is arbitrary; the
   tool auto-matches your clicks to the C3D `Board1..Board6` labels by rigid
   distance matching and reports the error.  **Save markers** writes one
   `output/markers.json` with the layout AND the board geometry; **Load
   markers** restores both from that same file (the board config is applied
   at runtime).  Save when verified.
2. **Sync** — Video: click 2 points on the video to box the flashing LED →
   *Compute traces* (intensity trace + threshold → binary).  Mocap: set the
   3D box by **clicking 2 corners on the 2D top view** (X–Y rectangle; the Z
   center and ±half come from the box spinboxes), or use *Auto-find LED* →
   presence trace.  Both binary traces are plotted on a **shared axis
   starting at 0**.  Then align by eye:
   * the **video scrub slider** moves the RGB view and has a **blue dashed
     bar** on the plot; the **mocap scrub slider** moves the C3D view and
     has a **purple dashed bar** — move them independently until each bar
     sits on the corresponding blink.
   * click **Set offset from bars** → `offset = video_bar − mocap_bar` and
     the purple bar lands on the blue bar (the sync is locked).
   * the **sync scrub** is a **center-zero jog slider** (0 in the middle):
     drag right moves **both** scrub sliders forward, drag left moves both
     backward, by the same time delta (it recentres on release). It stops
     when either slider reaches the end frame of its recording.
   * nudge **Offset fine-tune** (or *Invert/Reset offset*) to perfect it,
     then *Save sync*.  The mocap axes are locked to the whole recording.
3. **Trim** — One shared start/end (video seconds) applied to both streams.
4. **Calibrate** — Two steps:
   1. *Calibrate intrinsics (chessboard)* — self-calibrates the camera matrix
      from the chessboard video → saves `output/intrinsics.json`.
   2. *Run calibration (mocap → camera)* — solvePnP (board pose per frame)
      + Umeyama fit with iterative outlier rejection → saves
      `output/transform.json` and shows residual stats. Auto-runs step 1
      first if `intrinsics.json` is missing.

## Keyboard shortcuts

* Click the **video scrub slider**, then `←`/`→` = ±1 frame, `PageUp`/`PageDown`
  = ±10 frames.  (`Home`/`End` = first/last frame.)

## Output files (`output/`)

Same format as the web version:

| File | Contents |
|---|---|
| `intrinsics.json` | camera matrix + distortion (self-calibrated at video res) |
| `markers.json` | board-frame marker positions (mm) + C3D label assignment |
| `sync.json` | time offset (`video_time = mocap_time + offset_s`) + ROI/box |
| `trim.json` | shared trim window (video seconds) |
| `transform.json` | `R`, `t` (mocap→camera, mm) + residual stats |
| `settings.json` | last-used video/C3D paths (pre-filled in the Data tab) |

## Design notes (same as web version)

- **Time-based sync, not frame indices** — the iPad video is variable frame
  rate (~27.6 fps avg), so every frame is pre-extracted with its real
  timestamp; the C3D is uniform (100 Hz).  `video_time = mocap_time + offset_s`.
- **Intrinsics self-calibrated** from the chessboard video itself (exact
  720×1280 resolution).
- **Planar-target handling** — solvePnP's twin ambiguity is resolved by
  keeping the board normal facing the camera; a robust outlier loop rejects
  corner-order-flipped / bad frames.
- For best auto-sync results, use a **distinctive blink pattern** in the LED
  (a few short pulses) rather than a long steady ON.

## Tests

```bash
python3 test_calibration.py   # synthetic validation of the calibration math
```

## Files

- `config.py` — paths (defaults + `settings.json` persistence), board params, marker names
- `data_loader.py` — frame pre-extraction (JPEG cache + timestamps), C3D loading
- `intrinsics.py` — chessboard detection + `calibrateCamera`
- `marker_layout.py` — Stage-1 canvas + C3D distance verification
- `sync.py` — traces, threshold, cross-correlation, auto-find LED
- `mocap_view.py` — 3D/2D mocap rendering
- `trim.py` — shared trim logic
- `calibrate.py` — solvePnP + Umeyama + outlier rejection
- `app.py` — PyQt5 desktop UI (5 tabs)
