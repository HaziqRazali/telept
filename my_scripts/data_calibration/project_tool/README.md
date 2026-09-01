# Mocap → RGB Projection Tool (PC, PySide6)

Projects **all** mocap markers from a `.c3d` onto the synchronized iPad RGB
video, using calibration parameters saved by the **calibration tool**
(`pc_calib_tool`).  Recorded in the same day/session, so the intrinsics +
mocap→camera transform from the calibration apply directly.

It never calibrates — it just loads the saved transform and lets you scrub
through the take with the overlay.

## Run

```bash
.venv/bin/python app.py      # needs a display (no browser, no Gradio)
```

`.venv` is a symlink to `../pc_calib_tool/.venv` (same dependencies: PySide6,
opencv-headless, numpy, scipy, matplotlib, ezc3d).

## Workflow (tabs)

1. **0. Data** — set the iPad video + C3D paths, click *Load data*.  Paths
   default to the same-session take
   (`rgb_1787047433275.mp4` + `haziq_upperlimb_right2 dynamic 03.c3d`).
2. **1. Sync** — LED blink alignment, exactly like the calibration tool:
   draw the video ROI, the mocap 3D box, set the threshold, *Compute traces*,
   align the pulses, fine-tune the offset, then **Save sync**.
3. **2. Trim** — shared start/end (video seconds), **Save trim**.
4. **3. Project** —
   * *Load calibration* — reads `intrinsics.json` + `transform.json` from
     `../pc_calib_tool/output/` (change the paths in `config.py` if your
     calibrations live elsewhere).
   * the main pane shows the video frame with every tracked mocap marker
     projected (colored circles + labels; the auto/LED `*` markers are
     yellow).  Scrub the slider (or the mocap scrub / 3D view) to step
     through the take.  *Show all markers* draws even untracked ones;
     *Limit scrub to trim* restricts the slider to the trimmed window.

The projection recomputes on the fly: `mocap_frame = (video_time − offset) ×
fps`, then `P_cam = R·P_mocap + t`, then `projectPoints` with the intrinsics.

## Files

| File | Where | Notes |
|---|---|---|
| `output/sync.json` | `project_tool/output/` | offset, ROI, box, threshold |
| `output/trim.json` | `project_tool/output/` | trim start/end |
| `output/settings.json` | `project_tool/output/` | last video/C3D paths + fps override |
| `intrinsics.json` | `../pc_calib_tool/output/` | camera matrix + distortion (from calibration) |
| `transform.json` | `../pc_calib_tool/output/` | `P_cam = R·P_mocap + t`, mm (from calibration) |

Resume: the app auto-loads `settings.json` (paths) on start; use **Load sync**
and **Load trim** to restore a previous session's state.
