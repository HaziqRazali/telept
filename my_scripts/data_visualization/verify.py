"""
verify.py

Verify the PnP calibration by projecting C3D markers onto both the IR and RGB
streams simultaneously, displayed / saved side-by-side.

Left panel  — IR  (640×576)  : C3D markers projected with rvec/tvec from pnp_results.npz
Right panel — RGB (1920×1080 → scaled to same height) : same markers chained through
              the Orbbec SDK IR→RGB extrinsic.

Sync
----
  IR frame i  →  RGB frame  sync_index[i]  →  C3D frame  via offset.txt

Usage
-----
  conda activate orbbec
  python verify.py [take_dir] [--save]

  take_dir defaults to /data/telept/data/mocap/20141017/take1
  --save   write the full side-by-side video immediately on startup
           (output: <take_dir>/verify_sidebyside.mp4)

Controls (OpenCV window)
  ← / →   or  A / D   step −1 / +1 IR frame
  Space        play / pause
  S            toggle recording of side-by-side video
  Q / Esc      quit
"""

import os
import sys
import glob
import argparse

import cv2
import numpy as np
import ezc3d

# ── args ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("take_dir", nargs="?",
                    default="/data/telept/data/mocap/20141017/take1")
parser.add_argument("--save", action="store_true",
                    help="auto-save full side-by-side video and exit")
args = parser.parse_args()
TAKE_DIR = args.take_dir

# ── Orbbec Femto Bolt: IR → RGB extrinsics (from Orbbec SDK, t in mm) ─────────
R_IR_TO_RGB = np.array([
    [ 0.9938264489173889, -0.0014704565983265638,  0.0015902734594419599],
    [ 0.0012849620543420315,  0.9938266277313232,  0.11093678325414658 ],
    [-0.0017435838235542178, -0.11093448102474213,  0.993826150894165  ],
], dtype=np.float64)
T_IR_TO_RGB_MM = np.array(
    [-32.8342170715332, -1.315084457397461, 1.3067550659179688],
    dtype=np.float64)  # mm

# ── Orbbec Femto Bolt: RGB intrinsics (1920×1080) ─────────────────────────────
K_RGB = np.array([
    [1123.86669921875, 0.0,              948.0269165039062],
    [0.0,             1123.028076171875, 539.6485595703125],
    [0.0,             0.0,              1.0              ],
], dtype=np.float64)
DIST_RGB = np.array([
     0.07333821058273315,
    -0.10178927332162857,
    -0.0004722462617792189,
    -0.00022512981377076358,
     0.041689008474349976,
], dtype=np.float64)  # k1 k2 p1 p2 k3

# ── marker colours (BGR) cycled per marker index ──────────────────────────────
_COLORS = [
    (0, 255, 0), (0, 200, 255), (255, 128, 0), (255, 0, 255),
    (0, 255, 200), (255, 255, 0), (128, 0, 255), (0, 128, 255),
    (255, 80, 80), (80, 255, 80), (80, 80, 255),
]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Locate files
# ─────────────────────────────────────────────────────────────────────────────
pnp_path = os.path.join(TAKE_DIR, "pnp_results.npz")
offset_path = os.path.join(TAKE_DIR, "offset.txt")

# auto-detect C3D
c3d_candidates = glob.glob(os.path.join(TAKE_DIR, "*.c3d"))
if not c3d_candidates:
    sys.exit(f"ERROR: no .c3d file found in {TAKE_DIR}")
c3d_path = c3d_candidates[0]
print(f"C3D:  {c3d_path}")

# auto-detect RGB video
rgb_candidates = glob.glob(os.path.join(TAKE_DIR, "rgb_*.avi")) + \
                 glob.glob(os.path.join(TAKE_DIR, "rgb_*.mp4"))
if not rgb_candidates:
    sys.exit(f"ERROR: no rgb_*.avi / rgb_*.mp4 found in {TAKE_DIR}")
rgb_path = rgb_candidates[0]
print(f"RGB:  {rgb_path}")

# auto-detect IR npy directory
ir_dir_candidates = sorted(glob.glob(os.path.join(TAKE_DIR, "ir_*")))
ir_dir = next((d for d in ir_dir_candidates if os.path.isdir(d)), None)
if ir_dir is None:
    sys.exit(f"ERROR: no ir_* directory found in {TAKE_DIR}")
npy_files = sorted(glob.glob(os.path.join(ir_dir, "frame_*.npy")))
if not npy_files:
    sys.exit(f"ERROR: no frame_*.npy found in {ir_dir}")
print(f"IR:   {ir_dir}  ({len(npy_files)} frames)")

# auto-detect sync_index
sync_index = None
si_path = os.path.join(ir_dir, "sync_index.npy")
if os.path.exists(si_path):
    sync_index = np.load(si_path)
    print(f"sync_index: {si_path}  ({len(sync_index)} entries)")

if not os.path.exists(pnp_path):
    sys.exit(f"ERROR: pnp_results.npz not found in {TAKE_DIR}\n"
             "Run calibrate_pnp.py first.")
if not os.path.exists(offset_path):
    sys.exit(f"ERROR: offset.txt not found in {TAKE_DIR}\n"
             "Run sync_and_calibrate.py first.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load data
# ─────────────────────────────────────────────────────────────────────────────

# PnP results
pnp = np.load(pnp_path)
rvec_ir  = pnp["rvec_median"].reshape(3, 1)   # C3D → IR
tvec_ir  = pnp["tvec_median"].reshape(3, 1)
K_IR     = pnp["K"]
DIST_IR  = pnp["dist"]
print(f"rvec_ir: {rvec_ir.flatten()}")
print(f"tvec_ir: {tvec_ir.flatten()}")

# Rotation matrix from PnP: world (C3D) → IR camera
R_c3d_to_ir, _ = cv2.Rodrigues(rvec_ir)

# ── chain: C3D → IR → RGB ─────────────────────────────────────────────────────
t_ir_to_rgb_m = T_IR_TO_RGB_MM / 1000.0
R_c3d_to_rgb  = R_IR_TO_RGB @ R_c3d_to_ir
t_c3d_to_rgb  = (R_IR_TO_RGB @ tvec_ir).flatten() + t_ir_to_rgb_m
rvec_rgb, _   = cv2.Rodrigues(R_c3d_to_rgb)
tvec_rgb      = t_c3d_to_rgb.reshape(3, 1)
print(f"\nChained C3D→RGB  rvec: {rvec_rgb.flatten()}  tvec: {tvec_rgb.flatten()}")

# Parse offset.txt
offset = {}
with open(offset_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        offset[k.strip()] = v.strip()

c3d_offset  = int(offset["optitrack_frame"])
orb_offset  = int(offset["orbbec_frame"])
c3d_fps     = float(offset["optitrack_fps"])
orb_fps     = float(offset["orbbec_fps"])
print(f"\nSync: C3D frame {c3d_offset} ↔ Orbbec frame {orb_offset}")
print(f"      C3D fps={c3d_fps}  Orbbec fps={orb_fps}")


def ir_frame_to_c3d_frame(ir_idx):
    """IR frame index → C3D frame index via sync_index + offset.txt."""
    orb_f = int(sync_index[ir_idx]) if sync_index is not None else ir_idx
    dt = (orb_f - orb_offset) / orb_fps
    return int(np.clip(c3d_offset + round(dt * c3d_fps), 0, c3d_n - 1))


def rgb_frame_to_c3d_frame(rgb_f):
    """Map RGB video frame index → C3D frame index."""
    if sync_index is not None:
        ir_f  = int(np.clip(np.searchsorted(sync_index, rgb_f), 0, len(sync_index)-1))
        orb_f = int(sync_index[ir_f])
    else:
        orb_f = rgb_f
    dt = (orb_f - orb_offset) / orb_fps
    return int(np.clip(c3d_offset + round(dt * c3d_fps), 0, c3d_n - 1))


# Load C3D
print(f"\nLoading {c3d_path} …")
c = ezc3d.c3d(c3d_path)
c3d_xyz = c["data"]["points"][:3]   # (3, M, N_frames)
c3d_n   = c3d_xyz.shape[2]
c3d_labels = c["parameters"]["POINT"]["LABELS"]["value"]
print(f"  {c3d_n} frames  {c3d_xyz.shape[1]} markers")

# ── output canvas geometry ────────────────────────────────────────────────────
# IR native: 640×576.  Scale RGB to match IR height.
IR_W, IR_H = 640, 576
cap = cv2.VideoCapture(rgb_path)
if not cap.isOpened():
    sys.exit(f"ERROR: cannot open {rgb_path}")
n_rgb   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
rgb_fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
rw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
rh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
RGB_W   = int(round(rw * IR_H / rh))  # ~1024 for 1920×1080 → 576 height
RGB_H   = IR_H
CANVAS_W = IR_W + 2 + RGB_W
CANVAS_H = IR_H
print(f"\nRGB video: {n_rgb} frames  {rw}×{rh}  {rgb_fps_val:.1f}fps")
print(f"Canvas: {CANVAS_W}×{CANVAS_H}  (IR {IR_W}×{IR_H}  |  RGB scaled {RGB_W}×{RGB_H})")

n_ir = len(npy_files)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Projection helpers
# ─────────────────────────────────────────────────────────────────────────────

def project_onto(pts3d, rvec, tvec, K, dist, w, h):
    """Project (N,3) points. Return list of (px, py, point_idx) within (w,h)."""
    if not len(pts3d):
        return []
    proj, _ = cv2.projectPoints(pts3d.astype(np.float64), rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    return [(float(proj[i,0]), float(proj[i,1]), i)
            for i in range(len(pts3d))
            if 0 <= proj[i,0] < w and 0 <= proj[i,1] < h]


def get_visible_markers(c3d_f):
    """Return (pts3d, marker_indices) for visible markers at c3d_f."""
    pts = c3d_xyz[:, :, c3d_f].T          # (M, 3)
    vis = np.where(np.all(np.isfinite(pts), axis=1))[0]
    return pts[vis].astype(np.float64), vis


def draw_markers(img, projected, marker_indices, radius=6):
    """Draw coloured dots at projected positions."""
    for (px, py, i) in projected:
        mi  = int(marker_indices[i])
        col = _COLORS[mi % len(_COLORS)]
        cx, cy = int(px), int(py)
        cv2.circle(img, (cx, cy), radius,     col, -1)
        cv2.circle(img, (cx, cy), radius + 2, col,  1)


def load_ir_bgr(ir_idx):
    """Load IR .npy frame, convert uint16 → 8-bit BGR."""
    raw = np.load(npy_files[ir_idx])          # uint16
    gray = (raw >> 8).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def render_pair(ir_idx):
    """Render one side-by-side BGR frame (CANVAS_W × CANVAS_H)."""
    c3d_f  = ir_frame_to_c3d_frame(ir_idx)
    rgb_f  = int(sync_index[ir_idx]) if sync_index is not None else ir_idx
    pts3d, vis = get_visible_markers(c3d_f)

    # ── IR panel ──
    ir_bgr = load_ir_bgr(ir_idx)
    proj_ir = project_onto(pts3d, rvec_ir, tvec_ir, K_IR, DIST_IR, IR_W, IR_H)
    draw_markers(ir_bgr, proj_ir, vis)
    label = f"IR {ir_idx}  C3D {c3d_f}  vis={len(proj_ir)}"
    cv2.putText(ir_bgr, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),   2, cv2.LINE_AA)
    cv2.putText(ir_bgr, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # ── RGB panel ──
    cap.set(cv2.CAP_PROP_POS_FRAMES, rgb_f)
    ok, rgb_bgr = cap.read()
    if not ok:
        rgb_bgr = np.zeros((rh, rw, 3), dtype=np.uint8)
    proj_rgb = project_onto(pts3d, rvec_rgb, tvec_rgb, K_RGB, DIST_RGB, rw, rh)
    draw_markers(rgb_bgr, proj_rgb, vis)
    label2 = f"RGB {rgb_f}  vis={len(proj_rgb)}"
    cv2.putText(rgb_bgr, label2, (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0),   3, cv2.LINE_AA)
    cv2.putText(rgb_bgr, label2, (6, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1, cv2.LINE_AA)
    rgb_small = cv2.resize(rgb_bgr, (RGB_W, RGB_H))

    # ── separator ──
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:, :IR_W]         = ir_bgr
    canvas[:, IR_W:IR_W+2]   = (80, 80, 80)   # 2-px grey divider
    canvas[:, IR_W+2:]       = rgb_small
    return canvas

# ─────────────────────────────────────────────────────────────────────────────
# 5. Interactive playback  (primary index = IR frame)
# ─────────────────────────────────────────────────────────────────────────────
WIN = "verify — IR | RGB   [A/D=step  Space=play  S=record  Q=quit]"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, min(CANVAS_W, 1440), min(CANVAS_H, 480))

ir_idx  = 0
playing = False
writer  = None
saving  = False
delay   = max(1, int(1000 / (rgb_fps_val or 30.0)))

if args.save:
    # Non-interactive: render all frames and save
    out_path = os.path.join(TAKE_DIR, "verify_sidebyside.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, rgb_fps_val, (CANVAS_W, CANVAS_H))
    print(f"\nSaving {n_ir} frames to {out_path} …")
    for i in range(n_ir):
        frame = render_pair(i)
        writer.write(frame)
        if i % 100 == 0:
            print(f"  {i}/{n_ir}", flush=True)
    writer.release()
    cap.release()
    print("Done.")
    sys.exit(0)

print("\nWindow opened.  A/D or ←/→ = step,  Space = play/pause,  S = record,  Q = quit")

while True:
    canvas = render_pair(ir_idx)

    if saving and writer is not None:
        writer.write(canvas)

    cv2.imshow(WIN, canvas)

    wait = delay if playing else 0
    key  = cv2.waitKey(wait) & 0xFF

    if key in (ord('q'), 27):
        break
    elif key == ord(' '):
        playing = not playing
    elif key in (ord('d'), 83):          # D / →
        ir_idx = min(n_ir - 1, ir_idx + 1)
        playing = False
    elif key in (ord('a'), 81):          # A / ←
        ir_idx = max(0, ir_idx - 1)
        playing = False
    elif key == ord('s'):
        if not saving:
            out_path = os.path.join(TAKE_DIR, "verify_sidebyside.mp4")
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            writer   = cv2.VideoWriter(out_path, fourcc, rgb_fps_val, (CANVAS_W, CANVAS_H))
            if writer.isOpened():
                saving = True
                print(f"\nRecording → {out_path}  (press S again to stop)")
            else:
                print("ERROR: could not open video writer.")
                writer = None
        else:
            saving = False
            writer.release(); writer = None
            print("Recording stopped.")
    elif playing:
        ir_idx = min(n_ir - 1, ir_idx + 1)
        if ir_idx == n_ir - 1:
            playing = False

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
