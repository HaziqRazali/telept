"""
calibrate_pnp.py

PnP calibration: C3D 3-D marker positions → Orbbec IR camera extrinsics.

Workflow
--------
1. Load offset.txt (auto-detected or via button) to sync C3D ↔ IR frames.
2. Load C3D and IR Dir.
3. Inspect frames:
   - IR threshold slider (0-255): binarises the 8-bit-shifted IR image (ir16>>8).
     Pixels BRIGHTER than the threshold are kept; connected blobs above
     MIN_BLOB_AREA pixels are reported as IR clusters (likely LED markers).
   - Left-drag on IR panel → add an ignore bbox (clusters inside are suppressed).
   - Right-click on an ignore bbox in IR → remove it.
   - "Reset IR Bboxes" clears all IR ignore bboxes.
   - Left-drag on C3D panel → add a C3D ignore bbox (markers projected inside
     that screen-space box are EXCLUDED from PnP, shown in red).
   - Right-click a C3D ignore bbox → remove it.
   - "Reset C3D Bboxes" clears all C3D ignore bboxes.
4. "Run PnP": prompts for expected marker count N.
   - Frames where #(IR clusters) == #(active C3D markers) == N are used.
   - All N! pairings tried; lowest reprojection-error pairing wins.
   - Median rvec/tvec saved to pnp_results.npz.
   - After PnP, IR frames are annotated with projected C3D markers (coloured ●).

Controls
--------
  ← / →   step −1 / +1 IR frame
  q        quit
"""

import os
import glob

from itertools import permutations as _permutations, combinations as _combinations

import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D        # noqa: F401
from mpl_toolkits.mplot3d.proj3d import proj_transform

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

try:
    import ezc3d
    HAS_EZC3D = True
except ImportError:
    HAS_EZC3D = False
    print("WARNING: ezc3d not found — install with: pip install ezc3d")


# ── Orbbec Femto Bolt IR camera intrinsics (640×576) ─────────────────────────
# Source: print_ir_intrinsics.py  (Orbbec SDK readback)
# NOTE: The Orbbec SDK reports k1=17, k2=9.4 for the IR sensor.  These large
#       values look unusual for OpenCV's Brown-Conrady model and may actually
#       be Kannala-Brandt / fisheye coefficients.  If reprojection error stays
#       high, try setting DIST to np.zeros(5) first to isolate the problem.
K = np.array([
    [504.14373779296875, 0.0,               329.8353271484375],
    [0.0,               504.02496337890625, 334.18988037109375],
    [0.0,               0.0,               1.0              ],
], dtype=np.float64)

# Option A: SDK values — only correct if the SDK uses Brown-Conrady (unusual for IR)
DIST = np.array([
    17.021440505981445,
     9.416301727294922,
     7.297786214621738e-05,
     3.846113759209402e-05,
     0.3900681436061859,
], dtype=np.float64)  # k1, k2, p1, p2, k3 (OpenCV order — see NOTE above)

# Option B: no distortion — use this first to diagnose; if reprojection error is
#           acceptable it means K is correct and only the distortion model is wrong
DIST = np.zeros(5, dtype=np.float64)

# ── constants ──────────────────────────────────────────────────────────────────
INIT_IR_THRESH = 30
MIN_BLOB_AREA  = 4
# How many extra IR blobs above expected_n to tolerate per frame.
# 0 = exact match only;  1 = allow one noise blob (tries C(n+1,n) subsets).
BLOB_N_TOLERANCE = 1
CROSS_SIZE     = 6
CROSS_COLOR    = (0, 255, 0)
IR_BBOX_COLOR  = (0, 80, 255)    # BGR orange for IR ignore bboxes
IR_BBOX_MPL    = "#ff5000"
C3D_BBOX_MPL   = "#ff88ff"       # magenta for C3D ignore bboxes

PROJ_COLORS = [             # BGR colors for projected C3D markers on IR image
    (0,   255, 0),
    (0,   200, 255),
    (255, 128, 0),
    (255,   0, 255),
    (0,   255, 200),
    (255, 255,   0),
    (128,   0, 255),
    (0,   128, 255),
]


# ── dialogs ────────────────────────────────────────────────────────────────────

def _ask_file(title, filetypes):
    root = tk.Tk(); root.withdraw()
    p = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy(); return p or None


def _ask_dir(title):
    root = tk.Tk(); root.withdraw()
    p = filedialog.askdirectory(title=title)
    root.destroy(); return p or None


def _ask_int(prompt, default=1, minval=1, maxval=20):
    root = tk.Tk(); root.withdraw()
    v = simpledialog.askinteger("PnP", prompt, initialvalue=default,
                                minvalue=minval, maxvalue=maxval)
    root.destroy(); return v


def _msgbox(title, msg):
    root = tk.Tk(); root.withdraw()
    messagebox.showinfo(title, msg)
    root.destroy()


# ── C3D ────────────────────────────────────────────────────────────────────────

def load_c3d(path):
    c   = ezc3d.c3d(path)
    fps = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    xyz = c["data"]["points"][:3]          # (3, M, N_frames)
    n   = xyz.shape[2]
    t   = np.arange(n) / fps
    all_pts = xyz.reshape(3, -1)
    mask    = np.all(np.isfinite(all_pts), axis=0)
    if mask.any():
        fp  = all_pts[:, mask]
        lo  = np.percentile(fp, 1,  axis=1)
        hi  = np.percentile(fp, 99, axis=1)
        pad = np.maximum((hi - lo) * 0.20, 0.05)
        bounds = np.stack([lo - pad, hi + pad], axis=1)
    else:
        bounds = np.array([[-1,1],[-1,1],[-1,1]], dtype=float)
    return xyz, fps, n, t, bounds


# ── IR ─────────────────────────────────────────────────────────────────────────

def load_ir_raw(path):
    return np.load(path)


def ir_to_display(ir16):
    ir8 = (ir16 >> 8).astype(np.uint8)
    return cv2.cvtColor(ir8, cv2.COLOR_GRAY2BGR)


def detect_clusters(ir16, threshold, ignore_bboxes):
    """Binarise IR frame and return list of (cx, cy, area) after masking bboxes.

    The 16-bit IR image is first right-shifted by 8 bits to produce an 8-bit
    image.  Pixels whose value exceeds `threshold` are treated as bright spots
    (LED markers); connected components larger than MIN_BLOB_AREA pixels are
    reported as clusters.
    """
    ir8 = (ir16 >> 8).astype(np.uint8)
    _, binary = cv2.threshold(ir8, threshold, 255, cv2.THRESH_BINARY)
    for (x1, y1, x2, y2) in ignore_bboxes:
        x1c = max(0, min(x1, x2)); x2c = min(binary.shape[1], max(x1, x2))
        y1c = max(0, min(y1, y2)); y2c = min(binary.shape[0], max(y1, y2))
        binary[y1c:y2c, x1c:x2c] = 0
    n_lbl, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
    clusters = []
    for i in range(1, n_lbl):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_BLOB_AREA:
            clusters.append((float(centroids[i, 0]), float(centroids[i, 1]), int(area)))
    return clusters


def compose_ir_frame(ir16, clusters, ignore_bboxes, proj_pts=None):
    """Return annotated BGR image.

    proj_pts: list of (px, py, marker_idx) — projected C3D markers, drawn
    as coloured circles after Run PnP.
    """
    img = ir_to_display(ir16)
    h, w = img.shape[:2]

    # IR ignore bboxes (orange)
    for (x1, y1, x2, y2) in ignore_bboxes:
        cv2.rectangle(img,
                      (max(0, min(x1, x2)), max(0, min(y1, y2))),
                      (min(w, max(x1, x2)), min(h, max(y1, y2))),
                      IR_BBOX_COLOR, 1)

    # IR cluster crosses (green)
    for (cx, cy, _area) in clusters:
        cxi, cyi = int(round(cx)), int(round(cy))
        cv2.line(img, (cxi-CROSS_SIZE, cyi), (cxi+CROSS_SIZE, cyi), CROSS_COLOR, 1)
        cv2.line(img, (cxi, cyi-CROSS_SIZE), (cxi, cyi+CROSS_SIZE), CROSS_COLOR, 1)
        cv2.circle(img, (cxi, cyi), 2, CROSS_COLOR, -1)

    # post-PnP: projected C3D markers (coloured circles)
    if proj_pts:
        for (px, py, midx) in proj_pts:
            col = PROJ_COLORS[midx % len(PROJ_COLORS)]
            cv2.circle(img, (int(round(px)), int(round(py))), 8, col, 2)
            cv2.circle(img, (int(round(px)), int(round(py))), 2, col, -1)
            cv2.putText(img, str(midx), (int(round(px))+9, int(round(py))-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

    return img


def precompute_clusters(npy_files, threshold, ignore_bboxes, status_cb=None):
    result = []
    n = len(npy_files)
    for i, p in enumerate(npy_files):
        result.append(detect_clusters(load_ir_raw(p), threshold, ignore_bboxes))
        if status_cb and i % 30 == 0:
            status_cb(i, n)
    return result


# ── offset.txt ─────────────────────────────────────────────────────────────────

def parse_offset(path):
    cfg = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


# ── PnP ────────────────────────────────────────────────────────────────────────

def _reproj_err(pts3d, pts2d, rvec, tvec):
    proj, _ = cv2.projectPoints(pts3d.astype(np.float64),
                                rvec, tvec, K, DIST)
    return float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1)))


def best_pnp(pts3d, pts2d_candidates, init_rvec=None, init_tvec=None):
    """Try all N! pairings (and C(M,N) subsets if M > N candidates).
    Returns (rvec, tvec, error, best_permutation).

    When len(pts2d_candidates) > len(pts3d) (extra noise blob), all
    C(M,N) subsets of the 2D candidates are tried, each with all N!
    orderings — the subset+ordering with lowest reprojection error wins.

    If init_rvec/init_tvec are provided (temporal prior from previous frame),
    SOLVEPNP_ITERATIVE is used with useExtrinsicGuess=True so the solver
    converges to the nearest geometric solution rather than a mirror-flip.
    This is the primary defence against symmetric-shape ambiguity.

    Without a prior, method is chosen by point count:
      n >= 6 : SOLVEPNP_ITERATIVE
      n >= 4 : SOLVEPNP_EPNP
      n >= 3 : SOLVEPNP_SQPNP

    After the best permutation is found, one LM refinement step is applied.
    """
    n      = len(pts3d)
    n_cand = len(pts2d_candidates)
    use_init = init_rvec is not None and init_tvec is not None
    if use_init:
        method = cv2.SOLVEPNP_ITERATIVE
    elif n >= 6:
        method = cv2.SOLVEPNP_ITERATIVE
    elif n >= 4:
        method = cv2.SOLVEPNP_EPNP
    else:
        method = cv2.SOLVEPNP_SQPNP

    # All subsets of 2D candidates of size n (handles extra noise blobs)
    if n_cand > n:
        subsets = list(_combinations(range(n_cand), n))
    else:
        subsets = [tuple(range(n))]

    best_e = float("inf")
    best_r = best_t = None
    best_p = list(range(n))

    for subset in subsets:
        sub_pts2d = pts2d_candidates[list(subset)]
        for perm in _permutations(range(n)):
            pts2d = sub_pts2d[list(perm)].astype(np.float64)
            if use_init:
                ok, rvec, tvec = cv2.solvePnP(
                    pts3d.astype(np.float64), pts2d, K, DIST,
                    init_rvec.copy(), init_tvec.copy(),
                    useExtrinsicGuess=True, flags=method)
            else:
                ok, rvec, tvec = cv2.solvePnP(
                    pts3d.astype(np.float64), pts2d, K, DIST,
                    flags=method)
            if not ok:
                continue
            err = _reproj_err(pts3d, pts2d, rvec, tvec)
            if err < best_e:
                best_e = err; best_r = rvec.copy(); best_t = tvec.copy()
                best_p = [subset[p] for p in perm]

    # LM refinement on the winning permutation
    if best_r is not None:
        pts2d_best = pts2d_candidates[best_p].astype(np.float64)
        try:
            cv2.solvePnPRefineLM(
                pts3d.astype(np.float64), pts2d_best, K, DIST,
                best_r, best_t)
            best_e = _reproj_err(pts3d, pts2d_best, best_r, best_t)
        except cv2.error:
            pass   # refinement not available in older OpenCV builds

    return best_r, best_t, best_e, best_p


# ── shared state ───────────────────────────────────────────────────────────────

ST = {
    # IR data
    "ir_dir":        None,
    "npy_files":     None,
    "sync_index":    None,
    "ir_n":          0,
    "all_clusters":  None,
    "ignore_bboxes": [],       # list of (x1,y1,x2,y2) in IR pixel coords

    # C3D data
    "c3d_xyz":    None,        # (3, M, N)
    "c3d_fps":    1.0,
    "c3d_n":      0,
    "c3d_t":      None,
    "c3d_bounds": None,
    # C3D ignore bboxes: screen-space display-px coords (x1d,y1d,x2d,y2d)
    "c3d_ignore_bboxes": [],

    # sync
    "offset_dir":       None,
    "c3d_frame_offset": 0,
    "orb_frame_offset": 0,
    "orb_fps":          30.0,

    # PnP results (populated after Run PnP)
    "pnp_rvec_med": None,
    "pnp_tvec_med": None,

    # draw state
    "ir_drawing":       False,
    "ir_draw_start":    None,
    "c3d_drawing":      False,
    "c3d_draw_start_d": None,

    # guard against recursive slider callbacks
    "_sl_updating": False,

    # PnP constraint: both IR clusters AND active C3D markers must equal this
    "expected_n": 5,

    # frames with per-frame reproj error above this threshold are excluded
    # from the final rvec/tvec median (set to 0 to disable)
    "max_err": 20.0,  # loosened default — tighten after confirming K/DIST are correct

    # offset loaded flag — IR dir loading is blocked until this is True
    "offset_loaded": False,

    # PnP frame range (IR frame indices; None = not set)
    "pnp_start": None,
    "pnp_end":   None,
}

_im_ir  = None
_sc_c3d = None
_c3d_bbox_patches = []   # mpatches.Rectangle overlays on fig for C3D ignore boxes


# ── sync ───────────────────────────────────────────────────────────────────────

def ir_frame_to_c3d_frame(ir_idx):
    if ST["npy_files"] is None or ST["c3d_xyz"] is None:
        return 0
    if ST["sync_index"] is not None:
        orb_f = int(ST["sync_index"][min(ir_idx, len(ST["sync_index"])-1)])
    else:
        orb_f = ir_idx
    dt = (orb_f - ST["orb_frame_offset"]) / (ST["orb_fps"] or 1.0)
    return int(np.clip(ST["c3d_frame_offset"] + round(dt * ST["c3d_fps"]),
                       0, ST["c3d_n"] - 1))


# ── C3D ignore bbox helpers ────────────────────────────────────────────────────

def _get_c3d_active_indices(c3d_f):
    """Return indices of markers that are visible AND not inside any C3D ignore bbox."""
    pts = ST["c3d_xyz"][:, :, c3d_f].T          # (M, 3)
    vis = np.where(np.all(np.isfinite(pts), axis=1))[0]
    if not len(vis) or not len(ST["c3d_ignore_bboxes"]):
        return vis
    proj_mat = ax_c3d.get_proj()
    active   = []
    for m in vis:
        x2, y2, _ = proj_transform(pts[m, 0], pts[m, 1], pts[m, 2], proj_mat)
        xd, yd    = ax_c3d.transData.transform((x2, y2))
        excluded  = any(
            min(x1d, x2d) <= xd <= max(x1d, x2d) and
            min(y1d, y2d) <= yd <= max(y1d, y2d)
            for (x1d, y1d, x2d, y2d) in ST["c3d_ignore_bboxes"]
        )
        if not excluded:
            active.append(int(m))
    return np.array(active, dtype=int)


def _disp_in_c3d_bbox(xd, yd):
    """Return index of first C3D bbox containing display-pixel point, else -1."""
    for i, (x1d, y1d, x2d, y2d) in enumerate(ST["c3d_ignore_bboxes"]):
        if (min(x1d, x2d) <= xd <= max(x1d, x2d) and
                min(y1d, y2d) <= yd <= max(y1d, y2d)):
            return i
    return -1


def _sync_c3d_bbox_patches():
    """Keep displayed bbox patches in sync with ST['c3d_ignore_bboxes']."""
    global _c3d_bbox_patches
    for p in _c3d_bbox_patches:
        try:
            fig.patches.remove(p)
        except ValueError:
            pass
    _c3d_bbox_patches.clear()
    for (x1d, y1d, x2d, y2d) in ST["c3d_ignore_bboxes"]:
        patch = mpatches.Rectangle(
            (min(x1d, x2d), min(y1d, y2d)),
            abs(x2d-x1d), abs(y2d-y1d),
            linewidth=1.2, edgecolor=C3D_BBOX_MPL, facecolor="#ff88ff22",
            transform=fig.dpi_scale_trans, zorder=10)
        fig.patches.append(patch)
        _c3d_bbox_patches.append(patch)


# ── post-PnP projection ────────────────────────────────────────────────────────

def _compute_proj_pts(c3d_f):
    """Return list of (px, py, marker_idx) projected onto the IR image, or []."""
    if ST["pnp_rvec_med"] is None or ST["c3d_xyz"] is None:
        return []
    active = _get_c3d_active_indices(c3d_f)
    if not len(active):
        return []
    pts3d = ST["c3d_xyz"][:, active, c3d_f].T.astype(np.float64)   # (K, 3)
    proj, _ = cv2.projectPoints(
        pts3d,
        ST["pnp_rvec_med"].reshape(3, 1),
        ST["pnp_tvec_med"].reshape(3, 1),
        K, DIST)
    proj = proj.reshape(-1, 2)
    return [(float(proj[i, 0]), float(proj[i, 1]), int(m)) for i, m in enumerate(active)]


# ── figure layout ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 10), facecolor="#111111")
fig.canvas.manager.set_window_title("PnP Calibration")

# ── panels ────────────────────────────────────────────────────────────────────
ax_ir  = fig.add_axes([0.02, 0.21, 0.46, 0.75])
ax_c3d = fig.add_axes([0.54, 0.21, 0.44, 0.75], projection="3d")

ax_ir.set_facecolor("#0d0d0d")
ax_ir.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
for sp in ax_ir.spines.values():
    sp.set_edgecolor("#333333")
_ph_ir = ax_ir.text(0.5, 0.5,
    "Click 'Load IR Dir' to open the ir_XXXXXXXX folder",
    transform=ax_ir.transAxes, ha="center", va="center",
    color="#555555", fontsize=11)

ax_c3d.set_facecolor("#0a0a0a")
for _pane in (ax_c3d.xaxis.pane, ax_c3d.yaxis.pane, ax_c3d.zaxis.pane):
    _pane.fill = False; _pane.set_edgecolor("#333333")
ax_c3d.grid(True, color="#222222", linewidth=0.4)
ax_c3d.tick_params(colors="#555555", labelsize=5)
ax_c3d.set_xlabel("X", color="#666666", fontsize=7, labelpad=1)
ax_c3d.set_ylabel("Y", color="#666666", fontsize=7, labelpad=1)
ax_c3d.set_zlabel("Z", color="#666666", fontsize=7, labelpad=1)
title_c3d = ax_c3d.set_title("optitrack.c3d", color="#666666", fontsize=9, pad=3)

# ── IR threshold slider (full width) ────────────────────────────────────────
# Binarises the 8-bit-shifted IR (ir16>>8). Pixels brighter than threshold = blob.
ax_thr = fig.add_axes([0.02, 0.172, 0.96, 0.022], facecolor="#2a2a2a")
thresh_sl = mwidgets.Slider(
    ax_thr, "IR blob thresh  (bright pixels above this = LED cluster)", 0, 255,
    valinit=INIT_IR_THRESH, valstep=1, color="#226633")
thresh_sl.label.set_color("white"); thresh_sl.label.set_fontsize(7)
thresh_sl.valtext.set_color("white")

# ── single frame slider (full width) ────────────────────────────────────────
# Dragging this slider drives both IR and C3D panels simultaneously.
# C3D frame is always computed from IR frame via the sync formula.
ax_ir_sl = fig.add_axes([0.02, 0.142, 0.96, 0.022], facecolor="#1a1a1a")
ir_sl = mwidgets.Slider(ax_ir_sl, "IR frame", 0, 1,
                         valinit=0, valstep=1, color="#555555")
ir_sl.label.set_color("#888888"); ir_sl.valtext.set_color("#888888")

# vertical line markers for PnP range (updated by Set Start / Set End)
_range_start_line = ax_ir_sl.axvline(x=0, color="#44ff44", linewidth=1.5,
                                      linestyle="--", visible=False, zorder=5)
_range_end_line   = ax_ir_sl.axvline(x=1, color="#ff4444", linewidth=1.5,
                                      linestyle="--", visible=False, zorder=5)

# ── info bar ──────────────────────────────────────────────────────────────────
ax_info = fig.add_axes([0.02, 0.108, 0.96, 0.026])
ax_info.axis("off")
info_text = ax_info.text(0.5, 0.5,
    "Load offset.txt, C3D, and IR dir to begin",
    ha="center", va="center", color="#666666", fontsize=8.5,
    transform=ax_info.transAxes, fontfamily="monospace")

# ── buttons ───────────────────────────────────────────────────────────────────
# Row 1 (y=0.068): load + step
ax_b_offset = fig.add_axes([0.02, 0.068, 0.12, 0.030])
btn_offset  = mwidgets.Button(ax_b_offset, "Load offset.txt",
                               color="#222233", hovercolor="#333355")
btn_offset.label.set_color("#aaaaff"); btn_offset.label.set_fontsize(8)

ax_b_c3d = fig.add_axes([0.16, 0.068, 0.09, 0.030])
btn_c3d  = mwidgets.Button(ax_b_c3d, "Load C3D",
                            color="#1a1a33", hovercolor="#2a2a55")
btn_c3d.label.set_color("#8888ff"); btn_c3d.label.set_fontsize(8)

ax_b_ir = fig.add_axes([0.27, 0.068, 0.10, 0.030])
btn_ir  = mwidgets.Button(ax_b_ir, "Load IR Dir",
                           color="#1a3322", hovercolor="#2a5533")
btn_ir.label.set_color("#aaffaa"); btn_ir.label.set_fontsize(8)

ax_b_minus = fig.add_axes([0.40, 0.068, 0.055, 0.030])
btn_minus  = mwidgets.Button(ax_b_minus, "◀ −1",
                              color="#333333", hovercolor="#555555")
btn_minus.label.set_color("white"); btn_minus.label.set_fontsize(9)

ax_b_plus = fig.add_axes([0.46, 0.068, 0.055, 0.030])
btn_plus  = mwidgets.Button(ax_b_plus, "+1 ▶",
                             color="#333333", hovercolor="#555555")
btn_plus.label.set_color("white"); btn_plus.label.set_fontsize(9)

ax_b_rir = fig.add_axes([0.55, 0.068, 0.12, 0.030])
btn_reset_ir = mwidgets.Button(ax_b_rir, "Reset IR Bboxes",
                                color="#332222", hovercolor="#553333")
btn_reset_ir.label.set_color("#ffaaaa"); btn_reset_ir.label.set_fontsize(8)

ax_b_rc3d = fig.add_axes([0.69, 0.068, 0.13, 0.030])
btn_reset_c3d = mwidgets.Button(ax_b_rc3d, "Reset C3D Bboxes",
                                 color="#332233", hovercolor="#553355")
btn_reset_c3d.label.set_color("#ffaaff"); btn_reset_c3d.label.set_fontsize(8)

ax_b_pnp = fig.add_axes([0.84, 0.068, 0.14, 0.030])
btn_pnp  = mwidgets.Button(ax_b_pnp, "▶  Run PnP",
                            color="#1a3322", hovercolor="#2a5533")
btn_pnp.label.set_color("#aaffaa"); btn_pnp.label.set_fontsize(8)

# Row 2 (y=0.030): save/load boxes + N input + range markers
ax_b_save_boxes = fig.add_axes([0.02, 0.030, 0.13, 0.030])
btn_save_boxes  = mwidgets.Button(ax_b_save_boxes, "💾 Save Boxes",
                                   color="#222200", hovercolor="#443300")
btn_save_boxes.label.set_color("#ffdd55"); btn_save_boxes.label.set_fontsize(8)

ax_b_load_boxes = fig.add_axes([0.17, 0.030, 0.13, 0.030])
btn_load_boxes  = mwidgets.Button(ax_b_load_boxes, "📂 Load Boxes",
                                   color="#1a2a1a", hovercolor="#2a4a2a")
btn_load_boxes.label.set_color("#88ee88"); btn_load_boxes.label.set_fontsize(8)

# N constraint TextBox — set this BEFORE clicking Run PnP
# A frame is valid only when both #IR clusters == N AND #active C3D markers == N
ax_n_box = fig.add_axes([0.36, 0.030, 0.04, 0.030], facecolor="#1a1a2a")
n_textbox = mwidgets.TextBox(
    ax_n_box,
    "N required  ",
    initial=str(ST["expected_n"]),
    color="#1a1a2a", hovercolor="#2a2a44")
n_textbox.label.set_color("#aaaaff"); n_textbox.label.set_fontsize(7.5)
n_textbox.text_disp.set_color("#ffff88"); n_textbox.text_disp.set_fontsize(11)

# Set Start / Set End buttons — mark the IR frame range used for PnP
ax_b_set_start = fig.add_axes([0.43, 0.030, 0.11, 0.030])
btn_set_start  = mwidgets.Button(ax_b_set_start, "[ Set Start ]",
                                   color="#112211", hovercolor="#224422")
btn_set_start.label.set_color("#44ff44"); btn_set_start.label.set_fontsize(8)

ax_b_set_end = fig.add_axes([0.56, 0.030, 0.11, 0.030])
btn_set_end  = mwidgets.Button(ax_b_set_end, "[ Set End ]",
                                 color="#221111", hovercolor="#442222")
btn_set_end.label.set_color("#ff4444"); btn_set_end.label.set_fontsize(8)

ax_b_clear_range = fig.add_axes([0.69, 0.030, 0.09, 0.030])
btn_clear_range  = mwidgets.Button(ax_b_clear_range, "Clear Range",
                                    color="#2a1a1a", hovercolor="#443333")
btn_clear_range.label.set_color("#ffaaaa"); btn_clear_range.label.set_fontsize(8)

# Max reprojection error filter — frames above this are excluded from median
# Set to 0 to keep all frames.
ax_maxerr_box = fig.add_axes([0.81, 0.030, 0.04, 0.030], facecolor="#1a1a1a")
maxerr_textbox = mwidgets.TextBox(
    ax_maxerr_box,
    "max reproj err (px)  ",
    initial=str(ST["max_err"]),
    color="#1a1a1a", hovercolor="#2a2a2a")
maxerr_textbox.label.set_color("#ffaaaa"); maxerr_textbox.label.set_fontsize(7.5)
maxerr_textbox.text_disp.set_color("#ffdd88"); maxerr_textbox.text_disp.set_fontsize(10)

# ── rubber-band patches ───────────────────────────────────────────────────────
_ir_rubber = mpatches.Rectangle((0, 0), 0, 0, linewidth=1.5,
    edgecolor=IR_BBOX_MPL, facecolor="none", linestyle="--", visible=False)
ax_ir.add_patch(_ir_rubber)

_c3d_rubber = mpatches.Rectangle((0, 0), 0, 0, linewidth=1.5,
    edgecolor=C3D_BBOX_MPL, facecolor="none", linestyle="--", visible=False,
    transform=fig.dpi_scale_trans)
fig.patches.append(_c3d_rubber)


# ── redraw helpers ────────────────────────────────────────────────────────────

def _frame_in_range(ir_idx):
    s = ST["pnp_start"]; e = ST["pnp_end"]
    if s is None or e is None:
        return True   # no range set → all frames allowed
    return s <= ir_idx <= e


def _count_valid_frames():
    """Count frames where marker counts match AND frame is in the PnP range.
    Mirrors the logic in _collect_frames: C3D must equal expected_n exactly;
    IR blobs may be up to BLOB_N_TOLERANCE above expected_n.
    """
    if ST["all_clusters"] is None or ST["c3d_xyz"] is None:
        return None
    n = ST["expected_n"]
    count = 0
    for ir_idx in range(ST["ir_n"]):
        if not _frame_in_range(ir_idx):
            continue
        n_blobs = len(ST["all_clusters"][ir_idx])
        if not (n <= n_blobs <= n + BLOB_N_TOLERANCE):
            continue
        c3d_f  = ir_frame_to_c3d_frame(ir_idx)
        active = _get_c3d_active_indices(c3d_f)
        if len(active) == n:
            count += 1
    return count


def _range_str():
    s = ST["pnp_start"]; e = ST["pnp_end"]
    if s is None and e is None:
        return "range: all"
    s_str = str(s) if s is not None else "?"
    e_str = str(e) if e is not None else "?"
    span  = (e - s + 1) if (s is not None and e is not None) else "?"
    return f"range: {s_str}–{e_str} ({span} frames)"


def _update_info(ir_idx, c3d_f, n_clusters, n_c3d_active):
    t_ir = ""
    if ST["npy_files"] is not None:
        orb_f = int(ST["sync_index"][min(ir_idx, len(ST["sync_index"])-1)]) \
                if ST["sync_index"] is not None else ir_idx
        dt   = (orb_f - ST["orb_frame_offset"]) / (ST["orb_fps"] or 1.0)
        t_ir = f"  t={dt:.3f}s"
    pnp_stat = "  PnP: computed ✓" if ST["pnp_rvec_med"] is not None else "  PnP: not run"
    n         = ST["expected_n"]
    # IR blobs: allow [n, n+BLOB_N_TOLERANCE]; C3D must be exactly n
    n_match   = (n <= n_clusters <= n + BLOB_N_TOLERANCE and n_c3d_active == n)
    in_range  = _frame_in_range(ir_idx)
    tol_note  = f"+{BLOB_N_TOLERANCE}" if BLOB_N_TOLERANCE else ""
    if n_match and in_range:
        validity = "  ✓ VALID (count+range)"
        color = "#88ff88"
    elif n_match:
        validity = "  ✗ out of range"
        color = "#ffaa44"
    elif in_range:
        validity = f"  ✗ count mismatch (IR need {n}–{n+BLOB_N_TOLERANCE}, C3D need {n})"
        color = "#ff6666"
    else:
        validity = f"  ✗ out of range + count mismatch"
        color = "#ff4444"
    info_text.set_text(
        f"IR frame {ir_idx}{t_ir}  |  C3D frame {c3d_f}/{ST['c3d_n']}"
        f"  |  IR clusters: {n_clusters}"
        f"  |  C3D active: {n_c3d_active}"
        f"  |  {_range_str()}"
        f"  |{validity}"
        f"  |{pnp_stat}")
    info_text.set_color(color)


def _redraw_c3d(c3d_f):
    global _sc_c3d
    if ST["c3d_xyz"] is None:
        return
    fc  = int(np.clip(c3d_f, 0, ST["c3d_n"]-1))
    pts = ST["c3d_xyz"][:, :, fc].T        # (M, 3)
    M   = pts.shape[0]
    vis = np.all(np.isfinite(pts), axis=1)

    active_set = set(_get_c3d_active_indices(fc).tolist())
    colors = []
    for m in range(M):
        if not vis[m]:
            colors.append("#2a2a2a")
        elif m in active_set:
            colors.append("#ffee00")   # yellow = active
        else:
            colors.append("#993333")   # red = inside C3D ignore bbox

    vis_pts = pts[vis]
    vis_col = [colors[m] for m in range(M) if vis[m]]

    if _sc_c3d is None:
        if len(vis_pts):
            _sc_c3d = ax_c3d.scatter(
                vis_pts[:, 0], vis_pts[:, 1], vis_pts[:, 2],
                c=vis_col, s=30, alpha=0.95, depthshade=False)
        else:
            _sc_c3d = ax_c3d.scatter([], [], [], c="#ffee00", s=30, depthshade=False)
    else:
        if len(vis_pts):
            _sc_c3d._offsets3d = (vis_pts[:, 0], vis_pts[:, 1], vis_pts[:, 2])
            _sc_c3d.set_facecolor(vis_col)
        else:
            _sc_c3d._offsets3d = (np.array([]), np.array([]), np.array([]))

    n_vis    = int(vis.sum())
    n_active = len(active_set)
    n_ign    = n_vis - n_active
    title_c3d.set_text(
        f"C3D  frame {fc}   visible: {n_vis}   "
        f"active (yellow): {n_active}   ignored (red): {n_ign}"
        f"   left-drag=add ignore  right-click=remove")


def _redraw_ir(ir_idx):
    global _im_ir
    if ST["npy_files"] is None or ST["all_clusters"] is None:
        return
    c3d_f    = ir_frame_to_c3d_frame(ir_idx)
    ir16     = load_ir_raw(ST["npy_files"][ir_idx])
    clusters = ST["all_clusters"][ir_idx]
    proj_pts = _compute_proj_pts(c3d_f)
    img      = compose_ir_frame(ir16, clusters, ST["ignore_bboxes"], proj_pts)
    rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if _im_ir is None:
        _im_ir = ax_ir.imshow(rgb, aspect="auto", interpolation="bilinear", origin="upper")
        h, w = ir16.shape[:2]
        ax_ir.set_xlim(-0.5, w-0.5); ax_ir.set_ylim(h-0.5, -0.5)
    else:
        _im_ir.set_data(rgb)
    pnp_note = "  [projected C3D ●]" if proj_pts else ""
    ax_ir.set_title(
        f"Orbbec IR   frame {ir_idx}   clusters: {len(clusters)}{pnp_note}"
        f"   left-drag=add ignore  right-click=remove",
        color="white", fontsize=9, pad=3)


def _full_redraw(ir_idx, c3d_override=None):
    c3d_f        = c3d_override if c3d_override is not None else ir_frame_to_c3d_frame(ir_idx)
    _redraw_ir(ir_idx)
    _redraw_c3d(c3d_f)
    n_clusters   = len(ST["all_clusters"][ir_idx]) if ST["all_clusters"] else 0
    n_c3d_active = len(_get_c3d_active_indices(c3d_f)) if ST["c3d_xyz"] is not None else 0
    _update_info(ir_idx, c3d_f, n_clusters, n_c3d_active)
    fig.canvas.draw_idle()


# ── slider activation ─────────────────────────────────────────────────────────

def _update_range_lines():
    """Refresh the green/red range marker lines on the IR slider."""
    n = ST["ir_n"]
    if n < 2:
        _range_start_line.set_visible(False)
        _range_end_line.set_visible(False)
        return
    if ST["pnp_start"] is not None:
        _range_start_line.set_xdata([ST["pnp_start"], ST["pnp_start"]])
        _range_start_line.set_visible(True)
    else:
        _range_start_line.set_visible(False)
    if ST["pnp_end"] is not None:
        _range_end_line.set_xdata([ST["pnp_end"], ST["pnp_end"]])
        _range_end_line.set_visible(True)
    else:
        _range_end_line.set_visible(False)


def _activate_ir_slider(n, start=0):
    ax_ir_sl.set_facecolor("#2a2a2a")
    ir_sl.valmax = float(n - 1)
    ir_sl.ax.set_xlim(0, float(n - 1))
    ir_sl.poly.set_facecolor("#33aa66")
    ir_sl.label.set_color("white"); ir_sl.valtext.set_color("white")
    # default range = full extent (only set once on first load)
    if ST["pnp_start"] is None:
        ST["pnp_start"] = 0
    if ST["pnp_end"] is None:
        ST["pnp_end"] = n - 1
    _update_range_lines()
    ir_sl.set_val(float(np.clip(start, 0, n - 1)))


def on_ir_change(val):
    if ST["_sl_updating"] or ST["npy_files"] is None or ST["all_clusters"] is None:
        return
    ir_idx = int(ir_sl.val)
    c3d_f  = ir_frame_to_c3d_frame(ir_idx)
    _full_redraw(ir_idx, c3d_f)


ir_sl.on_changed(on_ir_change)


# ── threshold ─────────────────────────────────────────────────────────────────

def on_thresh(val):
    if ST["npy_files"] is None:
        return
    print(f"Recomputing clusters (thr={int(val)}) …")
    btn_ir.label.set_text("Recomputing…")
    fig.canvas.draw_idle(); fig.canvas.flush_events()
    ST["all_clusters"] = precompute_clusters(
        ST["npy_files"], int(val), ST["ignore_bboxes"])
    btn_ir.label.set_text("✓ IR Dir")
    _full_redraw(int(ir_sl.val))


thresh_sl.on_changed(on_thresh)


# ── load callbacks ────────────────────────────────────────────────────────────

def _apply_offset(cfg):
    ST["c3d_frame_offset"] = int(cfg.get("optitrack_frame", 0))
    ST["orb_frame_offset"] = int(cfg.get("orbbec_frame",    0))
    ST["orb_fps"]          = float(cfg.get("orbbec_fps",    30.0))
    # pre-load C3D fps from offset.txt so sync works before C3D file is loaded;
    # on_load_c3d() will override this with the actual fps from the .c3d file.
    if "optitrack_fps" in cfg:
        ST["c3d_fps"] = float(cfg["optitrack_fps"])
    ST["offset_loaded"] = True


def on_load_offset(event):
    path = _ask_file("Open offset.txt", [("Text", "*.txt"), ("All", "*.*")])
    if not path:
        return
    _apply_offset(parse_offset(path))
    ST["offset_dir"] = os.path.dirname(os.path.abspath(path))
    print(f"Offset: C3D frame {ST['c3d_frame_offset']}  "
          f"Orbbec frame {ST['orb_frame_offset']}  fps {ST['orb_fps']:.4f}")
    btn_offset.label.set_text("✓ offset.txt")
    btn_offset.ax.set_facecolor("#223322")
    fig.canvas.draw_idle()


def on_load_c3d(event):
    if not HAS_EZC3D:
        info_text.set_text("ezc3d not installed — pip install ezc3d")
        info_text.set_color("#ff6666"); fig.canvas.draw_idle(); return
    path = _ask_file("Open C3D file", [("C3D", "*.c3d"), ("All", "*.*")])
    if not path:
        return
    print(f"\nLoading {os.path.basename(path)} …")
    btn_c3d.label.set_text("Loading…"); fig.canvas.draw_idle(); fig.canvas.flush_events()
    xyz, fps, n, t, bounds = load_c3d(path)
    ST.update({"c3d_xyz": xyz, "c3d_fps": fps, "c3d_n": n,
               "c3d_t": t, "c3d_bounds": bounds})
    print(f"  C3D: {n} frames @ {fps:.4f} fps  markers: {xyz.shape[1]}")
    ax_c3d.set_xlim(bounds[0]); ax_c3d.set_ylim(bounds[1]); ax_c3d.set_zlim(bounds[2])
    c3d_start = ir_frame_to_c3d_frame(int(ir_sl.val)) if ST["npy_files"] is not None \
                else ST["c3d_frame_offset"]
    _redraw_c3d(c3d_start)
    btn_c3d.label.set_text("✓ C3D"); btn_c3d.ax.set_facecolor("#222238")
    fig.canvas.draw_idle()


def on_load_ir(event):
    path = _ask_dir("Select IR frame directory (ir_XXXXXXXX)")
    if not path:
        return
    npy_files = sorted(glob.glob(os.path.join(path, "frame_*.npy")))
    if not npy_files:
        info_text.set_text(f"No frame_*.npy in {path}")
        info_text.set_color("#ff6666"); fig.canvas.draw_idle(); return

    # ─ offset.txt is REQUIRED ───────────────────────────────────────────
    if not ST["offset_loaded"]:
        # try auto-loading from parent folder
        off_path = os.path.join(os.path.dirname(path), "offset.txt")
        if os.path.exists(off_path):
            _apply_offset(parse_offset(off_path))
            ST["offset_dir"] = os.path.dirname(os.path.abspath(off_path))
            print(f"  offset.txt auto-loaded from {off_path}")
            btn_offset.label.set_text("✓ offset.txt (auto)")
            btn_offset.ax.set_facecolor("#223322")
        else:
            # block: refuse to load without offset
            info_text.set_text(
                "⛔  offset.txt is required before loading IR dir.  "
                "Click 'Load offset.txt' or place offset.txt in the parent folder.")
            info_text.set_color("#ff4444")
            fig.canvas.draw_idle()
            return

    sync_path  = os.path.join(path, "sync_index.npy")
    sync_index = np.load(sync_path) if os.path.exists(sync_path) else None
    print(f"  sync_index: {len(sync_index)} entries" if sync_index is not None
          else "  No sync_index.npy — IR frame == orbbec frame")

    ST.update({"ir_dir": path, "npy_files": npy_files,
               "sync_index": sync_index, "ir_n": len(npy_files)})
    # reset range on new IR dir load
    ST["pnp_start"] = None
    ST["pnp_end"]   = None

    print(f"\nPrecomputing clusters for {len(npy_files)} IR frames …")
    btn_ir.label.set_text("Computing…"); fig.canvas.draw_idle(); fig.canvas.flush_events()

    def _progress(i, n):
        btn_ir.label.set_text(f"Computing {i}/{n}")
        fig.canvas.draw_idle(); fig.canvas.flush_events()

    ST["all_clusters"] = precompute_clusters(
        npy_files, int(thresh_sl.val), ST["ignore_bboxes"], _progress)
    print("  Done.")

    _ph_ir.set_visible(False)
    # IR start frame = orb_frame_offset (IR has the same offset as orbbec.mp4)
    _activate_ir_slider(len(npy_files), start=ST["orb_frame_offset"])
    _full_redraw(int(ir_sl.val))
    btn_ir.label.set_text("✓ IR Dir"); btn_ir.ax.set_facecolor("#1a3322")
    fig.canvas.draw_idle()


# ── IR ignore bboxes ──────────────────────────────────────────────────────────

def _pt_in_ir_bbox(x, y, bbox):
    x1, y1, x2, y2 = bbox
    return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)


def _recompute_ir():
    ST["all_clusters"] = precompute_clusters(
        ST["npy_files"], int(thresh_sl.val), ST["ignore_bboxes"])
    _full_redraw(int(ir_sl.val))


def on_reset_ir_bboxes(event):
    ST["ignore_bboxes"].clear()
    if ST["npy_files"] is not None:
        _recompute_ir()
    else:
        fig.canvas.draw_idle()


# ── C3D ignore bboxes ─────────────────────────────────────────────────────────

def on_reset_c3d_bboxes(event):
    ST["c3d_ignore_bboxes"].clear()
    _sync_c3d_bbox_patches()
    _full_redraw(int(ir_sl.val))


# ── set start / end range ───────────────────────────────────────────────────────

def _refresh_range_ui():
    _update_range_lines()
    count = _count_valid_frames()
    if count is not None:
        btn_pnp.label.set_text(f"▶  Run PnP  ({count} valid frames)")
    # update Set Start / End button labels
    s = ST["pnp_start"]; e = ST["pnp_end"]
    btn_set_start.label.set_text(f"[ Start: {s} ]" if s is not None else "[ Set Start ]")
    btn_set_end.label.set_text(f"[ End: {e} ]" if e is not None else "[ Set End ]")
    fig.canvas.draw_idle()


def on_set_start(event):
    if ST["npy_files"] is None:
        return
    ST["pnp_start"] = int(ir_sl.val)
    # if start > end, push end forward
    if ST["pnp_end"] is not None and ST["pnp_start"] > ST["pnp_end"]:
        ST["pnp_end"] = ST["pnp_start"]
    print(f"PnP start = {ST['pnp_start']}")
    _refresh_range_ui()


def on_set_end(event):
    if ST["npy_files"] is None:
        return
    ST["pnp_end"] = int(ir_sl.val)
    # if end < start, push start back
    if ST["pnp_start"] is not None and ST["pnp_end"] < ST["pnp_start"]:
        ST["pnp_start"] = ST["pnp_end"]
    print(f"PnP end = {ST['pnp_end']}")
    _refresh_range_ui()


def on_clear_range(event):
    ST["pnp_start"] = None
    ST["pnp_end"]   = None
    print("PnP range cleared (all frames)")
    _refresh_range_ui()


# ── save / load boxes ─────────────────────────────────────────────────────────

def on_save_boxes(event):
    save_dir = ST["offset_dir"] or ST["ir_dir"] or "."
    path = os.path.join(save_dir, "boxes.txt")
    lines = ["# calibrate_pnp.py — ignore bounding boxes\n"]
    lines.append("# IR ignore bboxes (image pixel coords: x1,y1,x2,y2)\n")
    for (x1, y1, x2, y2) in ST["ignore_bboxes"]:
        lines.append(f"ir_bbox={int(x1)},{int(y1)},{int(x2)},{int(y2)}\n")
    lines.append("# C3D ignore bboxes (screen display-pixel coords — view-dependent)\n")
    for (x1d, y1d, x2d, y2d) in ST["c3d_ignore_bboxes"]:
        lines.append(f"c3d_bbox={x1d:.1f},{y1d:.1f},{x2d:.1f},{y2d:.1f}\n")
    with open(path, "w") as fh:
        fh.writelines(lines)
    print(f"Saved boxes → {path}  "
          f"(IR: {len(ST['ignore_bboxes'])}  C3D: {len(ST['c3d_ignore_bboxes'])})")
    btn_save_boxes.label.set_text("✓ Saved")
    btn_save_boxes.ax.set_facecolor("#333300")
    fig.canvas.draw_idle()


def on_load_boxes(event):
    path = _ask_file("Open boxes.txt", [("Text", "*.txt"), ("All", "*.*")])
    if not path:
        return
    ir_bboxes  = []
    c3d_bboxes = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            nums = [float(v) for v in val.split(",")]
            if key.strip() == "ir_bbox" and len(nums) == 4:
                ir_bboxes.append(tuple(int(v) for v in nums))
            elif key.strip() == "c3d_bbox" and len(nums) == 4:
                c3d_bboxes.append(tuple(nums))
    ST["ignore_bboxes"][:] = ir_bboxes
    ST["c3d_ignore_bboxes"][:] = c3d_bboxes
    _sync_c3d_bbox_patches()
    if ST["npy_files"] is not None:
        _recompute_ir()
    else:
        fig.canvas.draw_idle()
    print(f"Loaded boxes ← {path}  "
          f"(IR: {len(ir_bboxes)}  C3D: {len(c3d_bboxes)})")
    btn_load_boxes.label.set_text("✓ Loaded")
    btn_load_boxes.ax.set_facecolor("#1a3322")
    fig.canvas.draw_idle()


# ── mouse events ──────────────────────────────────────────────────────────────

def _in_c3d_axes(ex, ey):
    """True if display-pixel (ex, ey) falls inside the C3D axes bounding box."""
    pos      = ax_c3d.get_position()
    fw, fh   = fig.get_size_inches() * fig.dpi
    return (pos.x0*fw <= ex <= pos.x1*fw and
            pos.y0*fh <= ey <= pos.y1*fh)


def _on_press(event):
    if event.button == 1:
        # C3D: left-drag → add ignore bbox
        if ST["c3d_xyz"] is not None and _in_c3d_axes(event.x, event.y):
            ax_c3d.disable_mouse_rotation()
            ST["c3d_drawing"]      = True
            ST["c3d_draw_start_d"] = (event.x, event.y)
            _c3d_rubber.set_xy((event.x, event.y))
            _c3d_rubber.set_width(0); _c3d_rubber.set_height(0)
            _c3d_rubber.set_visible(True)
            fig.canvas.draw_idle()
            return

        # IR: left-drag → add ignore bbox
        if event.inaxes is ax_ir and ST["npy_files"] is not None:
            ST["ir_drawing"]    = True
            ST["ir_draw_start"] = (event.xdata, event.ydata)
            _ir_rubber.set_xy((event.xdata, event.ydata))
            _ir_rubber.set_width(0); _ir_rubber.set_height(0)
            _ir_rubber.set_visible(True)
            fig.canvas.draw_idle()

    elif event.button == 3:
        # C3D right-click → remove bbox under cursor
        if _in_c3d_axes(event.x, event.y) and ST["c3d_xyz"] is not None:
            idx = _disp_in_c3d_bbox(event.x, event.y)
            if idx >= 0:
                ST["c3d_ignore_bboxes"].pop(idx)
                _sync_c3d_bbox_patches()
                _full_redraw(int(ir_sl.val))
            return
        # IR right-click → remove IR ignore bbox
        if event.inaxes is ax_ir and event.xdata is not None:
            removed = [b for b in ST["ignore_bboxes"]
                       if _pt_in_ir_bbox(event.xdata, event.ydata, b)]
            if removed:
                for b in removed:
                    ST["ignore_bboxes"].remove(b)
                _recompute_ir()


def _on_motion(event):
    if ST["c3d_drawing"]:
        sx, sy = ST["c3d_draw_start_d"]
        _c3d_rubber.set_xy((min(sx, event.x), min(sy, event.y)))
        _c3d_rubber.set_width(abs(event.x - sx))
        _c3d_rubber.set_height(abs(event.y - sy))
        fig.canvas.draw_idle()
    if ST["ir_drawing"] and event.inaxes is ax_ir:
        sx, sy = ST["ir_draw_start"]
        ex = event.xdata if event.xdata is not None else sx
        ey = event.ydata if event.ydata is not None else sy
        _ir_rubber.set_xy((min(sx, ex), min(sy, ey)))
        _ir_rubber.set_width(abs(ex - sx)); _ir_rubber.set_height(abs(ey - sy))
        fig.canvas.draw_idle()


def _on_release(event):
    # C3D bbox complete
    if ST["c3d_drawing"] and event.button == 1:
        ST["c3d_drawing"] = False
        _c3d_rubber.set_visible(False)
        sx, sy = ST["c3d_draw_start_d"]
        ax_c3d.mouse_init()
        if abs(event.x - sx) > 5 and abs(event.y - sy) > 5:
            ST["c3d_ignore_bboxes"].append(
                (float(sx), float(sy), float(event.x), float(event.y)))
            _sync_c3d_bbox_patches()
            _full_redraw(int(ir_sl.val))
        else:
            fig.canvas.draw_idle()
        return

    # IR bbox complete
    if ST["ir_drawing"] and event.button == 1:
        ST["ir_drawing"] = False
        _ir_rubber.set_visible(False)
        sx, sy = ST["ir_draw_start"]
        ex = event.xdata if event.xdata is not None else sx
        ey = event.ydata if event.ydata is not None else sy
        if abs(ex - sx) > 2 and abs(ey - sy) > 2:
            ST["ignore_bboxes"].append(
                (int(min(sx, ex)), int(min(sy, ey)),
                 int(max(sx, ex)), int(max(sy, ey))))
            _recompute_ir()
        else:
            fig.canvas.draw_idle()


def _on_scroll(event):
    if event.inaxes is not ax_ir or ST["npy_files"] is None:
        return
    xl = list(ax_ir.get_xlim()); yl = list(ax_ir.get_ylim())
    cx = event.xdata or (xl[0]+xl[1])/2
    cy = event.ydata or (yl[0]+yl[1])/2
    f  = 0.80 if event.button == "up" else 1.25
    ax_ir.set_xlim([cx+(x-cx)*f for x in xl])
    ax_ir.set_ylim([cy+(y-cy)*f for y in yl])
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("button_press_event",   _on_press)
fig.canvas.mpl_connect("motion_notify_event",  _on_motion)
fig.canvas.mpl_connect("button_release_event", _on_release)
fig.canvas.mpl_connect("scroll_event",         _on_scroll)


# ── step buttons ──────────────────────────────────────────────────────────────

def _step(d):
    if ST["npy_files"] is None:
        return
    ir_sl.set_val(float(np.clip(ir_sl.val + d, 0, ST["ir_n"]-1)))


# ── Run PnP ───────────────────────────────────────────────────────────────────

def on_n_submit(text):
    """Called when the user presses Enter in the N textbox."""
    try:
        n = int(text.strip())
        n = max(1, min(n, 20))
    except ValueError:
        n_textbox.set_val(str(ST["expected_n"]))
        return
    ST["expected_n"] = n
    n_textbox.set_val(str(n))
    count = _count_valid_frames()
    if count is not None:
        btn_pnp.label.set_text(f"▶  Run PnP  ({count} valid frames)")
        print(f"N={n}: {count} valid frames (IR clusters == C3D active == {n})")
    fig.canvas.draw_idle()


def on_maxerr_submit(text):
    try:
        v = float(text.strip())
        v = max(0.0, v)
    except ValueError:
        maxerr_textbox.set_val(str(ST["max_err"]))
        return
    ST["max_err"] = v
    maxerr_textbox.set_val(f"{v:.1f}")
    fig.canvas.draw_idle()


n_textbox.on_submit(on_n_submit)
maxerr_textbox.on_submit(on_maxerr_submit)


def on_run_pnp(event):
    if ST["npy_files"] is None or ST["c3d_xyz"] is None:
        _msgbox("PnP", "Load C3D and IR dir first."); return

    expected_n = ST["expected_n"]

    print(f"\nRunning PnP  expected_n={expected_n}  "
          f"IR frames={ST['ir_n']}  C3D frames={ST['c3d_n']}")
    print(f"  FPS: orbbec={ST['orb_fps']:.4f}  C3D={ST['c3d_fps']:.4f}  "
          f"offsets: orb_frame={ST['orb_frame_offset']}  c3d_frame={ST['c3d_frame_offset']}")
    print(f"  sync: orb_frame → dt = (orb_f - {ST['orb_frame_offset']}) / {ST['orb_fps']:.4f}  "
          f"→ c3d_frame = {ST['c3d_frame_offset']} + dt × {ST['c3d_fps']:.4f}")

    def _collect_frames():
        """Yield (ir_idx, pts3d, pts2d) for all frames passing filters 1+2."""
        for ir_idx in range(ST["ir_n"]):
            if not _frame_in_range(ir_idx):
                continue
            c3d_f    = ir_frame_to_c3d_frame(ir_idx)
            clusters = ST["all_clusters"][ir_idx]
            active   = _get_c3d_active_indices(c3d_f)
            pts3d_f  = ST["c3d_xyz"][:, active, c3d_f].T.astype(np.float64)
            # Require exact C3D count; allow up to BLOB_N_TOLERANCE extra IR blobs
            if len(pts3d_f) != expected_n:
                continue
            if not (expected_n <= len(clusters) <= expected_n + BLOB_N_TOLERANCE):
                continue
            pts2d_f = np.array([[cx, cy] for cx, cy, _ in clusters], dtype=np.float64)
            yield ir_idx, pts3d_f, pts2d_f

    # ── Pass 1: blind search (no temporal prior) ──────────────────────────────
    print("  Pass 1: blind permutation search …")
    p1_rvec, p1_tvec, p1_err, p1_ir = [], [], [], []
    n_skip = 0
    for ir_idx, pts3d_f, pts2d_f in _collect_frames():
        rvec, tvec, err, _ = best_pnp(pts3d_f, pts2d_f)
        if rvec is None:
            n_skip += 1; continue
        p1_rvec.append(rvec.flatten()); p1_tvec.append(tvec.flatten())
        p1_err.append(err);             p1_ir.append(ir_idx)
        if len(p1_rvec) % 100 == 0:
            print(f"    {len(p1_rvec)} frames …  mean err={np.mean(p1_err):.3f}px",
                  flush=True)

    if not p1_rvec:
        _msgbox("PnP", f"No valid frames found.\nSkipped: {n_skip}\n"
                "Adjust IR threshold / bboxes / range.")
        return

    p1_rvec = np.array(p1_rvec); p1_tvec = np.array(p1_tvec); p1_err = np.array(p1_err)
    rough_rvec = np.median(p1_rvec, axis=0)
    rough_tvec = np.median(p1_tvec, axis=0)
    print(f"  Pass 1 done: {len(p1_rvec)} frames  "
          f"median err={np.median(p1_err):.3f}px  "
          f"mean err={p1_err.mean():.3f}px")
    print(f"  Rough rvec: {rough_rvec}")
    print(f"  Rough tvec: {rough_tvec}")

    # ── Pass 2: temporal-consistent re-solve ──────────────────────────────────
    # Each frame is re-solved using the previous frame's result as an
    # extrinsic guess, preventing mirror-flip ambiguity on symmetric shapes.
    print("  Pass 2: temporally-consistent re-solve …")
    p2_rvec, p2_tvec, p2_err, p2_ir = [], [], [], []
    prev_r = rough_rvec.reshape(3, 1).copy()
    prev_t = rough_tvec.reshape(3, 1).copy()
    for ir_idx, pts3d_f, pts2d_f in _collect_frames():
        rvec, tvec, err, _ = best_pnp(pts3d_f, pts2d_f,
                                       init_rvec=prev_r, init_tvec=prev_t)
        if rvec is None:
            continue
        p2_rvec.append(rvec.flatten()); p2_tvec.append(tvec.flatten())
        p2_err.append(err);             p2_ir.append(ir_idx)
        prev_r = rvec.copy(); prev_t = tvec.copy()   # chain to next frame
        if len(p2_rvec) % 100 == 0:
            print(f"    {len(p2_rvec)} frames …  mean err={np.mean(p2_err):.3f}px",
                  flush=True)

    p2_rvec = np.array(p2_rvec); p2_tvec = np.array(p2_tvec); p2_err = np.array(p2_err)
    print(f"  Pass 2 done: {len(p2_rvec)} frames  "
          f"median err={np.median(p2_err):.3f}px  "
          f"mean err={p2_err.mean():.3f}px")

    # ── Filter by max reprojection error ─────────────────────────────────────
    max_err_thr = ST["max_err"]
    if max_err_thr > 0:
        keep = p2_err <= max_err_thr
        n_filtered = int((~keep).sum())
        all_rvec = p2_rvec[keep]; all_tvec = p2_tvec[keep]
        all_err  = p2_err[keep];  valid_ir  = [p2_ir[i] for i in range(len(p2_ir)) if keep[i]]
        print(f"  Max-err filter ({max_err_thr:.1f}px): "
              f"kept {keep.sum()}/{len(p2_rvec)}  removed {n_filtered}")
    else:
        all_rvec = p2_rvec; all_tvec = p2_tvec
        all_err  = p2_err;  valid_ir  = p2_ir

    if not len(all_rvec):
        _msgbox("PnP", f"All frames were filtered out by max reproj err={max_err_thr:.1f}px.\n"
                "Lower the threshold or set it to 0 to disable.")
        return

    n_used   = len(all_rvec)

    # ── robust mean rotation via chordal SO(3) averaging ──────────────────────
    # np.median on raw Rodrigues vectors is WRONG: equivalent rotations can
    # have wildly different vec representations (axis flip, 2π wrap), so the
    # component-wise median produces a garbage result.
    # Correct approach: convert to rotation matrices, sum them, project the
    # sum back to SO(3) via SVD (Procrustes / chordal mean).
    R_sum = np.zeros((3, 3), dtype=np.float64)
    for rv in all_rvec:
        R, _ = cv2.Rodrigues(rv.reshape(3, 1))
        R_sum += R
    U, _, Vt = np.linalg.svd(R_sum)
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:          # ensure proper rotation (det=+1)
        U[:, -1] *= -1
        R_mean = U @ Vt
    rvec_med, _ = cv2.Rodrigues(R_mean)
    rvec_med = rvec_med.flatten()

    tvec_med = np.median(all_tvec, axis=0)  # translation: component-wise median is fine

    p50 = np.median(all_err)
    p75 = np.percentile(all_err, 75)
    p90 = np.percentile(all_err, 90)
    p95 = np.percentile(all_err, 95)
    print(f"\nPnP done: {n_used} frames used  ({n_skip} skipped by count/range filter)")
    print(f"  reproj err — mean: {all_err.mean():.3f}px  "
          f"p50: {p50:.3f}  p75: {p75:.3f}  p90: {p90:.3f}  p95: {p95:.3f}px")
    print(f"  rvec (med):  {rvec_med}")
    print(f"  tvec (med):  {tvec_med}")

    # store for live projection overlay
    ST["pnp_rvec_med"] = rvec_med
    ST["pnp_tvec_med"] = tvec_med

    save_dir  = ST["offset_dir"] or ST["ir_dir"] or "."
    save_path = os.path.join(save_dir, "pnp_results.npz")
    np.savez(save_path,
             valid_ir_frames=np.array(valid_ir, dtype=np.int32),
             rvecs=all_rvec, tvecs=all_tvec, reproj_errors=all_err,
             rvec_median=rvec_med, tvec_median=tvec_med,
             K=K, dist=DIST, expected_n=np.int32(expected_n))
    print(f"  Saved → {save_path}")

    # refresh all panels (now with projected markers on every IR frame)
    _full_redraw(int(ir_sl.val))

    btn_pnp.label.set_text(f"✓ PnP done  ({n_used} frames)")
    btn_pnp.ax.set_facecolor("#225522")
    _msgbox("PnP complete",
            f"Pass 1 (blind):       {len(p1_rvec)} frames  "
            f"median err {np.median(p1_err):.3f}px\n"
            f"Pass 2 (temporal):    {len(p2_rvec)} frames  "
            f"median err {np.median(p2_err):.3f}px\n"
            f"After max-err filter: {n_used} frames\n\n"
            f"Reproj err  p50={p50:.3f}  p75={p75:.3f}  p90={p90:.3f}px\n\n"
            f"Saved → {save_path}\n\n"
            "IR panel now shows projected C3D markers as coloured circles.")
    fig.canvas.draw_idle()


# ── wire up ───────────────────────────────────────────────────────────────────

btn_offset.on_clicked(on_load_offset)
btn_c3d.on_clicked(on_load_c3d)
btn_ir.on_clicked(on_load_ir)
btn_minus.on_clicked(lambda e: _step(-1))
btn_plus.on_clicked(lambda e: _step(+1))
btn_reset_ir.on_clicked(on_reset_ir_bboxes)
btn_reset_c3d.on_clicked(on_reset_c3d_bboxes)
btn_pnp.on_clicked(on_run_pnp)
btn_save_boxes.on_clicked(on_save_boxes)
btn_load_boxes.on_clicked(on_load_boxes)
btn_set_start.on_clicked(on_set_start)
btn_set_end.on_clicked(on_set_end)
btn_clear_range.on_clicked(on_clear_range)


def _on_key(event):
    if event.key == "q":
        plt.close(fig)
    elif event.key == "right":
        _step(+1)
    elif event.key == "left":
        _step(-1)


fig.canvas.mpl_connect("key_press_event", _on_key)

plt.show()
