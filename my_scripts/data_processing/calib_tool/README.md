# iPad ↔ Mocap Calibration Tool

Calibrates an iPad camera to an already-calibrated motion-capture system
(Vicon/QTM-style) using a 40 mm 9×6 chessboard (8×5 inner corners) with 6
reflective markers (`Board1`–`Board6`) placed on a 40 mm grid *outside* the
pattern, synchronized by a flashing light bulb.

Output: the rigid transform **mocap → camera** (`P_camera = R @ P_mocap + t`,
mm) plus the camera intrinsics.

## Requirements

Python 3.10+, `gradio`, `opencv-python`, `ezc3d`, `numpy`, `scipy`,
`matplotlib`. (All already present in the working environment.)

## Run

```bash
python3 app.py
```

Then open `http://localhost:7860` in a browser (local, or via SSH tunnel
`ssh -L 7860:localhost:7860 user@server`, or Tailscale). No monitor needed on
the server.

## Workflow (4 tabs, in order)

1. **Marker layout** — Click the 6 orange dots on the board schematic where
   the reflective markers sit (grid = 40 mm). Click order is arbitrary; the
   tool auto-matches your clicks to the C3D `Board1..Board6` labels by rigid
   distance matching and reports the error. Save when verified.
2. **Sync** — Video: click 2 points to box the flashing LED → intensity trace +
   threshold → binary. Mocap: set the 3D box (or *Auto-find LED*) → presence
   trace → binary. *Compute traces* → *Auto-sync* (cross-correlation proposes
   the offset) → fine-tune with the slider → *Save sync*. A synchronized scrub
   moves both viewers together.
3. **Trim** — One shared start/end (video seconds) applied to both streams.
4. **Calibrate** — Runs solvePnP (board pose per frame, intrinsics from the
   same video) + Umeyama fit with iterative outlier rejection → saves
   `output/transform.json` and shows residual stats.

## Output files (`output/`)

| File | Contents |
|---|---|
| `intrinsics.json` | camera matrix + distortion (self-calibrated at video res) |
| `markers.json` | board-frame marker positions (mm) + C3D label assignment |
| `sync.json` | time offset (`video_time = mocap_time + offset_s`) + ROI/box |
| `trim.json` | shared trim window (video seconds) |
| `transform.json` | `R`, `t` (mocap→camera, mm) + residual stats |

## Key design decisions

- **Time-based sync, not frame indices** — the iPad video is variable frame
  rate (~27.6 fps avg), so every frame is pre-extracted with its real
  timestamp; the C3D is uniform (100 Hz). `video_time = mocap_time + offset_s`.
- **Intrinsics self-calibrated** from the chessboard video itself (exact
  720×1280 resolution) — no need to remember a previous calibration.
- **Planar-target handling** — solvePnP's twin ambiguity is resolved by keeping
  the board normal facing the camera; a robust outlier loop rejects
  corner-order-flipped / bad frames.
- For best auto-sync results, use a **distinctive blink pattern** in the LED
  (a few short pulses) rather than a long steady ON — the latter makes the
  offset underdetermined.

## Tests

```bash
python3 test_calibration.py   # synthetic validation of the calibration math
```

## Files

- `config.py` — paths, board params, marker names
- `data_loader.py` — frame pre-extraction (JPEG cache + timestamps), C3D loading
- `intrinsics.py` — chessboard detection + `calibrateCamera`
- `marker_layout.py` — Stage-1 canvas + C3D distance verification
- `sync.py` — traces, threshold, cross-correlation, auto-find LED
- `mocap_view.py` — 3D/2D mocap rendering for the GUI
- `trim.py` — shared trim logic
- `calibrate.py` — solvePnP + Umeyama + outlier rejection
- `app.py` — Gradio UI (4 tabs)
