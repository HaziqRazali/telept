"""
Interactive sync viewer – Optitrack C3D / Optitrack AVI / Orbbec MP4.

Layout
------
  Top            : Blink-signal plot  (C3D blue | Optitrack orange | Orbbec green)
                   + two threshold sliders (one per video)
  Info bar       : per-stream ON-time and transition counts
  Bottom-left    : optitrack.avi (scaled) with optitrack_trimmed inset (bottom-right corner)
                   + frame slider  (same fps as C3D → one unified slider)
  Bottom-right   : orbbec.mp4 (scaled) with orbbec_trimmed inset (bottom-right corner)
                   + frame slider
  Bottom controls: [▶ Play]  [⟳ Reset]  [◀ −1]  [+1 ▶]  [[S] Save]

Key bindings: q = quit | ← / → = step −1 / +1

Usage
-----
    python sync_optitrack_orbbec.py
"""

import io as _io
import time
import numpy as np
import cv2
import ezc3d
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── paths ─────────────────────────────────────────────────────────────────────
BASE         = "/home/haziq/datasets/telept/data/mocap/20260325_164704"
C3D_PATH     = f"{BASE}/optitrack.c3d"
OPTI_PATH    = f"{BASE}/optitrack.avi"
OPTI_CR_PATH = f"{BASE}/optitrack_trimmed.avi"
ORBBEC_PATH  = f"{BASE}/orbbec.mp4"
ORBBEC_CR_PATH = f"{BASE}/orbbec_trimmed.mp4"

INIT_THRESH_OPTI   = 75
INIT_THRESH_ORBBEC = 75

FULL_SCALE    = 4      # cache full frames at 1/FULL_SCALE resolution
INSET_SCALE   = 5      # enlarge trimmed frames by this factor for the overlay
INSET_MARGIN  = 6      # px gap from frame edge
BOX_COLOR     = (255, 0, 0)
BOX_THICK     = 1

SAVE_PATH_OPTI   = f"{BASE}/sync_optitrack_recording.mp4"
SAVE_PATH_ORBBEC = f"{BASE}/sync_orbbec_recording.mp4"
RECORD_FPS   = 30

# ─── 1. Load C3D ──────────────────────────────────────────────────────────────
print("Loading C3D …")
c        = ezc3d.c3d(C3D_PATH)
c3d_fps  = float(c["parameters"]["POINT"]["RATE"]["value"][0])
xyz      = c["data"]["points"][:3]                        # [3, M, F]
n_c3d    = xyz.shape[2]
n_vis    = np.all(np.isfinite(xyz), axis=0).sum(axis=0)  # [F]
led_c3d  = (n_vis > 0).astype(np.float32)
t_c3d    = np.arange(n_c3d) / c3d_fps
print(f"  C3D: {n_c3d} frames @ {c3d_fps} fps  ({t_c3d[-1]:.3f} s)")

# compute stable 3-D bounds for the scatter viewer
_all_pts     = xyz.reshape(3, -1)
_finite_mask = np.all(np.isfinite(_all_pts), axis=0)
if _finite_mask.any():
    _fp  = _all_pts[:, _finite_mask]
    _lo  = np.percentile(_fp, 1,  axis=1)
    _hi  = np.percentile(_fp, 99, axis=1)
    _pad = np.maximum((_hi - _lo) * 0.20, 0.05)
    c3d_bounds = np.stack([_lo - _pad, _hi + _pad], axis=1)
else:
    c3d_bounds = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=float)

# ─── 2. Load optitrack_trimmed.avi (cache frames + medians) ───────────────────
print("Loading optitrack_trimmed …")
cap      = cv2.VideoCapture(OPTI_CR_PATH)
opti_cr_fps = cap.get(cv2.CAP_PROP_FPS)
n_opti_cr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
med_opti    = np.empty(n_opti_cr, dtype=np.float32)
opti_cr_frames = []

for i in range(n_opti_cr):
    ret, frame = cap.read()
    if not ret:
        med_opti[i:] = 0.0
        blank = np.zeros_like(opti_cr_frames[-1]) if opti_cr_frames else np.zeros((40, 40), np.uint8)
        opti_cr_frames.extend([blank] * (n_opti_cr - i))
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    med_opti[i] = float(np.median(gray))
    opti_cr_frames.append(gray)
    if i % 60 == 0:
        print(f"  opti_trimmed frame {i}/{n_opti_cr}", end="\r")
cap.release()
t_opti_cr = np.arange(n_opti_cr) / opti_cr_fps
print(f"  optitrack_trimmed: {n_opti_cr} frames @ {opti_cr_fps:.4f} fps  ({t_opti_cr[-1]:.3f} s)    ")

# ─── 3. Load orbbec_trimmed.mp4 (cache frames + medians) ─────────────────────
print("Loading orbbec_trimmed …")
cap      = cv2.VideoCapture(ORBBEC_CR_PATH)
orb_cr_fps  = cap.get(cv2.CAP_PROP_FPS)
n_orb_cr    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
med_orbbec  = np.empty(n_orb_cr, dtype=np.float32)
orb_cr_frames = []

for i in range(n_orb_cr):
    ret, frame = cap.read()
    if not ret:
        med_orbbec[i:] = 0.0
        blank = np.zeros_like(orb_cr_frames[-1]) if orb_cr_frames else np.zeros((40, 40), np.uint8)
        orb_cr_frames.extend([blank] * (n_orb_cr - i))
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    med_orbbec[i] = float(np.median(gray))
    orb_cr_frames.append(gray)
    if i % 60 == 0:
        print(f"  orbbec_trimmed frame {i}/{n_orb_cr}", end="\r")
cap.release()
t_orb_cr = np.arange(n_orb_cr) / orb_cr_fps
print(f"  orbbec_trimmed: {n_orb_cr} frames @ {orb_cr_fps:.4f} fps  ({t_orb_cr[-1]:.3f} s)    ")

# ─── 4. Load optitrack.avi at 1/FULL_SCALE ────────────────────────────────────
print(f"Loading optitrack.avi (1/{FULL_SCALE} scale) …")
cap      = cv2.VideoCapture(OPTI_PATH)
n_opti   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
opti_fps = cap.get(cv2.CAP_PROP_FPS)
ow_orig  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
oh_orig  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ow_disp  = max(1, ow_orig // FULL_SCALE)
oh_disp  = max(1, oh_orig // FULL_SCALE)

opti_full_frames = []
for i in range(n_opti):
    ret, frame = cap.read()
    if not ret:
        blank = np.zeros((oh_disp, ow_disp, 3), np.uint8)
        opti_full_frames.extend([blank] * (n_opti - i))
        break
    small = cv2.resize(frame, (ow_disp, oh_disp), interpolation=cv2.INTER_AREA)
    opti_full_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    if i % 60 == 0:
        print(f"  optitrack full frame {i}/{n_opti}", end="\r")
cap.release()
print(f"  optitrack.avi: {n_opti} frames @ {opti_fps:.4f} fps  (disp {ow_disp}x{oh_disp})    ")

# ─── 5. Load orbbec.mp4 at 1/FULL_SCALE ──────────────────────────────────────
print(f"Loading orbbec.mp4 (1/{FULL_SCALE} scale) …")
cap       = cv2.VideoCapture(ORBBEC_PATH)
n_orb     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orb_fps   = cap.get(cv2.CAP_PROP_FPS)
bw_orig   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
bh_orig   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
bw_disp   = max(1, bw_orig // FULL_SCALE)
bh_disp   = max(1, bh_orig // FULL_SCALE)

orb_full_frames = []
for i in range(n_orb):
    ret, frame = cap.read()
    if not ret:
        blank = np.zeros((bh_disp, bw_disp, 3), np.uint8)
        orb_full_frames.extend([blank] * (n_orb - i))
        break
    small = cv2.resize(frame, (bw_disp, bh_disp), interpolation=cv2.INTER_AREA)
    orb_full_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    if i % 60 == 0:
        print(f"  orbbec full frame {i}/{n_orb}", end="\r")
cap.release()
print(f"  orbbec.mp4: {n_orb} frames @ {orb_fps:.4f} fps  (disp {bw_disp}x{bh_disp})    ")

# unified slider ranges
n_opti_shared = min(n_opti, n_opti_cr, n_c3d)   # optitrack + C3D share fps
n_orb_shared  = min(n_orb,  n_orb_cr)
total_sync_t  = min((n_opti_shared - 1) / opti_fps  if opti_fps  > 0 else 0,
                    (n_orb_shared  - 1) / orb_fps   if orb_fps   > 0 else 0)

# ─── 6. Compose helpers ───────────────────────────────────────────────────────
def _make_cr_dims(cr_frames, disp_w, disp_h):
    """Compute inset pixel coords for a given display canvas size."""
    if not cr_frames:
        return 0, 0, 0, 0, 0, 0
    h_cr, w_cr = cr_frames[0].shape[:2]
    iw = min(w_cr * INSET_SCALE, disp_w - 2 * INSET_MARGIN)
    ih = min(h_cr * INSET_SCALE, disp_h - 2 * INSET_MARGIN)
    x1 = disp_w - iw - INSET_MARGIN
    y1 = disp_h - ih - INSET_MARGIN
    return int(x1), int(y1), int(x1 + iw), int(y1 + ih), int(iw), int(ih)

opti_x1, opti_y1, opti_x2, opti_y2, opti_iw, opti_ih = \
    _make_cr_dims(opti_cr_frames, ow_disp, oh_disp)
orb_x1,  orb_y1,  orb_x2,  orb_y2,  orb_iw,  orb_ih  = \
    _make_cr_dims(orb_cr_frames,  bw_disp, bh_disp)


def compose_opti(f):
    """Optitrack full frame with trimmed inset at bottom-right."""
    base = opti_full_frames[min(f, len(opti_full_frames) - 1)].copy()
    cf   = min(f, len(opti_cr_frames) - 1)
    if opti_iw > 0 and opti_ih > 0:
        big = cv2.resize(opti_cr_frames[cf], (opti_iw, opti_ih),
                         interpolation=cv2.INTER_NEAREST)
        if big.ndim == 2:
            big = np.stack([big] * 3, axis=2)
        base[opti_y1:opti_y2, opti_x1:opti_x2] = big
        cv2.rectangle(base, (opti_x1, opti_y1), (opti_x2, opti_y2), BOX_COLOR, BOX_THICK)
    return base


def compose_orb(f):
    """Orbbec full frame with trimmed inset at bottom-right."""
    base = orb_full_frames[min(f, len(orb_full_frames) - 1)].copy()
    cf   = min(f, len(orb_cr_frames) - 1)
    if orb_iw > 0 and orb_ih > 0:
        big = cv2.resize(orb_cr_frames[cf], (orb_iw, orb_ih),
                         interpolation=cv2.INTER_NEAREST)
        if big.ndim == 2:
            big = np.stack([big] * 3, axis=2)
        base[orb_y1:orb_y2, orb_x1:orb_x2] = big
        cv2.rectangle(base, (orb_x1, orb_y1), (orb_x2, orb_y2), BOX_COLOR, BOX_THICK)
    return base


# ─── 7. Figure / axes layout ──────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 11), facecolor="#111111")
fig.canvas.manager.set_window_title("Optitrack / Orbbec Sync Viewer")

# ── signal plot ──
ax_sig = fig.add_axes([0.05, 0.565, 0.92, 0.38])
ax_sig.set_facecolor("#1a1a1a")
ax_sig.set_xlabel("time (s)", color="#aaaaaa", fontsize=8)
ax_sig.set_ylabel("LED on/off", color="#aaaaaa", fontsize=8)
ax_sig.set_title(
    "Blink signals — C3D (blue) | Optitrack trimmed (orange) | Orbbec trimmed (green)",
    color="white", fontsize=10)
ax_sig.tick_params(colors="#888888", labelsize=7)
for sp in ax_sig.spines.values():
    sp.set_edgecolor("#444444")

line_c3d,  = ax_sig.step(t_c3d,    led_c3d,
                          where="post", color="#4499ff", lw=1.0, label="C3D")
led_opti_init  = (med_opti   > INIT_THRESH_OPTI  ).astype(np.float32)
led_orb_init   = (med_orbbec > INIT_THRESH_ORBBEC).astype(np.float32)
line_opti, = ax_sig.step(t_opti_cr, led_opti_init * 1.06,
                          where="post", color="#ffaa33", lw=1.0, alpha=0.85,
                          label=f"Optitrack (thr={INIT_THRESH_OPTI})")
line_orb,  = ax_sig.step(t_orb_cr,  led_orb_init  * 1.12,
                          where="post", color="#33cc66", lw=1.0, alpha=0.85,
                          label=f"Orbbec (thr={INIT_THRESH_ORBBEC})")
ax_sig.set_ylim(-0.2, 1.45)
ax_sig.legend(fontsize=8, facecolor="#222222", labelcolor="white", loc="upper right")

vline_opti = ax_sig.axvline(0, color="#ffaa33", lw=1.2, ls="--", alpha=0.7)
vline_orb  = ax_sig.axvline(0, color="#33cc66", lw=1.2, ls="-.",  alpha=0.7)
vline_c3d  = ax_sig.axvline(0, color="#4499ff", lw=1.2, ls=":",   alpha=0.7)

# ── threshold sliders ──
ax_thr_opti = fig.add_axes([0.10, 0.508, 0.38, 0.026], facecolor="#2a2a2a")
thresh_opti_sl = mwidgets.Slider(ax_thr_opti, "Opti thr", 0, 255,
                                  valinit=INIT_THRESH_OPTI,  valstep=1, color="#dd8833")
thresh_opti_sl.label.set_color("white")
thresh_opti_sl.valtext.set_color("white")

ax_thr_orb  = fig.add_axes([0.55, 0.508, 0.38, 0.026], facecolor="#2a2a2a")
thresh_orb_sl  = mwidgets.Slider(ax_thr_orb,  "Orbbec thr", 0, 255,
                                  valinit=INIT_THRESH_ORBBEC, valstep=1, color="#226633")
thresh_orb_sl.label.set_color("white")
thresh_orb_sl.valtext.set_color("white")

# ── info text ──
ax_info = fig.add_axes([0.05, 0.470, 0.92, 0.030])
ax_info.axis("off")
info_text = ax_info.text(0.5, 0.5, "", ha="center", va="center",
                          color="#cccccc", fontsize=8,
                          transform=ax_info.transAxes, fontfamily="monospace")

# ── bottom-left: optitrack ──
ax_opti = fig.add_axes([0.02, 0.13, 0.27, 0.31])
ax_opti.axis("off")
ax_opti.set_facecolor("#000000")
im_opti = ax_opti.imshow(compose_opti(0), aspect="auto", interpolation="bilinear")
title_opti = ax_opti.set_title(
    "optitrack.avi  t=0.000s  |  frame 0  (trimmed inset)",
    color="white", fontsize=9, pad=3)

# ── bottom-centre: C3D 3-D scatter (shares optitrack fps) ──
ax_c3d = fig.add_axes([0.32, 0.13, 0.27, 0.31], projection="3d")
ax_c3d.set_facecolor("#0a0a0a")
for _pane in (ax_c3d.xaxis.pane, ax_c3d.yaxis.pane, ax_c3d.zaxis.pane):
    _pane.fill = False
    _pane.set_edgecolor("#333333")
ax_c3d.grid(True, color="#222222", linewidth=0.4)
ax_c3d.tick_params(colors="#555555", labelsize=5)
ax_c3d.set_xlabel("X", color="#666666", fontsize=7, labelpad=1)
ax_c3d.set_ylabel("Y", color="#666666", fontsize=7, labelpad=1)
ax_c3d.set_zlabel("Z", color="#666666", fontsize=7, labelpad=1)
ax_c3d.set_xlim(c3d_bounds[0])
ax_c3d.set_ylim(c3d_bounds[1])
ax_c3d.set_zlim(c3d_bounds[2])

def get_visible_pts(f):
    pts  = xyz[:, :, f].T
    mask = np.all(np.isfinite(pts), axis=1)
    return pts[mask]

_pts0 = get_visible_pts(0)
if len(_pts0):
    sc_c3d = ax_c3d.scatter(_pts0[:, 0], _pts0[:, 1], _pts0[:, 2],
                             c="#ffee00", s=30, alpha=0.95, depthshade=False)
else:
    sc_c3d = ax_c3d.scatter([], [], [], c="#ffee00", s=30, depthshade=False)
title_c3d = ax_c3d.set_title("C3D  t=0.000s  |  frame 0",
                              color="white", fontsize=9, pad=3)

# shared slider spans optitrack + C3D columns
ax_opti_sl = fig.add_axes([0.02, 0.084, 0.57, 0.026], facecolor="#2a2a2a")
opti_slider = mwidgets.Slider(ax_opti_sl, "Opti/C3D frame", 0, n_opti_shared - 1,
                               valinit=0, valstep=1, color="#ffaa33")
opti_slider.label.set_color("white")
opti_slider.valtext.set_color("white")

# ── bottom-right: orbbec ──
ax_orb = fig.add_axes([0.63, 0.13, 0.35, 0.31])
ax_orb.axis("off")
ax_orb.set_facecolor("#000000")
im_orb = ax_orb.imshow(compose_orb(0), aspect="auto", interpolation="bilinear")
title_orb = ax_orb.set_title(
    "orbbec.mp4  t=0.000s  |  frame 0  (trimmed inset)",
    color="white", fontsize=9, pad=3)

ax_orb_sl = fig.add_axes([0.63, 0.084, 0.35, 0.026], facecolor="#2a2a2a")
orb_slider = mwidgets.Slider(ax_orb_sl, "Orbbec frame", 0, n_orb_shared - 1,
                              valinit=0, valstep=1, color="#33cc66")
orb_slider.label.set_color("white")
orb_slider.valtext.set_color("white")

# ── sync (time) slider — drives both streams ──
ax_sync_sl = fig.add_axes([0.05, 0.052, 0.92, 0.026], facecolor="#2a2a2a")
sync_slider = mwidgets.Slider(ax_sync_sl, "Sync (s)", 0.0, total_sync_t,
                               valinit=0.0, color="#aa44aa")
sync_slider.label.set_color("white")
sync_slider.valtext.set_color("white")

# ── control buttons ──
ax_play  = fig.add_axes([0.10, 0.010, 0.12, 0.030])
btn_play = mwidgets.Button(ax_play,  "▶  Play",   color="#225522", hovercolor="#338833")
btn_play.label.set_color("white"); btn_play.label.set_fontsize(9)

ax_reset  = fig.add_axes([0.25, 0.010, 0.09, 0.030])
btn_reset = mwidgets.Button(ax_reset, "⟳ Reset",   color="#222255", hovercolor="#333388")
btn_reset.label.set_color("white"); btn_reset.label.set_fontsize(9)

ax_minus  = fig.add_axes([0.38, 0.010, 0.055, 0.030])
btn_minus = mwidgets.Button(ax_minus, "◀  −1",    color="#333333", hovercolor="#555555")
btn_minus.label.set_color("white"); btn_minus.label.set_fontsize(9)

ax_plus   = fig.add_axes([0.44, 0.010, 0.055, 0.030])
btn_plus  = mwidgets.Button(ax_plus,  "+1  ▶",    color="#333333", hovercolor="#555555")
btn_plus.label.set_color("white");  btn_plus.label.set_fontsize(9)

ax_save   = fig.add_axes([0.55, 0.010, 0.12, 0.030])
btn_save  = mwidgets.Button(ax_save,  "[S]  Save", color="#334433", hovercolor="#446644")
btn_save.label.set_color("white");  btn_save.label.set_fontsize(9)

# ─── 8. State ─────────────────────────────────────────────────────────────────
playing        = False
_last_tick     = [0.0]
_opti_frac     = [0.0]
_orb_frac      = [0.0]
_sync_updating = [False]   # guard against feedback loops
# per-stream start frames when sync slider = 0
_sync_offsets  = [0, 0]    # [opti_frame_at_sync0, orb_frame_at_sync0]


# ─── 9. Helpers ───────────────────────────────────────────────────────────────
def compute_info(thr_opti, thr_orb):
    lo = (med_opti   > thr_opti).astype(np.float32)
    lb = (med_orbbec > thr_orb ).astype(np.float32)
    return (
        f"C3D   ON:{led_c3d.sum()/c3d_fps:.2f}s  trans:{int(np.abs(np.diff(led_c3d)).sum())}   |   "
        f"Opti  ON:{lo.sum()/opti_cr_fps:.2f}s  trans:{int(np.abs(np.diff(lo)).sum())}   |   "
        f"Orbbec ON:{lb.sum()/orb_cr_fps:.2f}s  trans:{int(np.abs(np.diff(lb)).sum())}"
    )


info_text.set_text(compute_info(INIT_THRESH_OPTI, INIT_THRESH_ORBBEC))


def _redraw_c3d(f):
    f_c = min(f, n_c3d - 1)
    t   = f_c / c3d_fps
    pts = get_visible_pts(f_c)
    if len(pts):
        sc_c3d._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
    else:
        sc_c3d._offsets3d = (np.array([]), np.array([]), np.array([]))
    title_c3d.set_text(
        f"C3D  t={t:.3f}s  |  frame {f_c}  |  LED={'ON' if led_c3d[f_c] else 'off'}")


def _redraw_opti(f):
    t   = f / opti_fps
    med = med_opti[min(f, len(med_opti) - 1)]
    im_opti.set_data(compose_opti(f))
    title_opti.set_text(
        f"optitrack.avi  t={t:.3f}s  |  frame {f}  |  median={med:.1f}  "
        f"|  C3D LED={'ON' if led_c3d[min(f, n_c3d-1)] else 'off'}")
    vline_opti.set_xdata([t, t])
    vline_c3d.set_xdata([t, t])
    _redraw_c3d(f)
    print(f"Opti frame {f:5d}  t={t:.3f}s  median={med:.1f}", flush=True)


def _redraw_orb(f):
    t   = f / orb_fps
    med = med_orbbec[min(f, len(med_orbbec) - 1)]
    im_orb.set_data(compose_orb(f))
    title_orb.set_text(
        f"orbbec.mp4  t={t:.3f}s  |  frame {f}  |  median={med:.1f}")
    vline_orb.set_xdata([t, t])
    print(f"Orbbec frame {f:5d}  t={t:.3f}s  median={med:.1f}", flush=True)


# ─── 10. Slider / button callbacks ────────────────────────────────────────────
def update_opti(val):
    """Individual opti slider moved — redraw, store new offset, reset sync to 0."""
    if _sync_updating[0]:
        return
    f = int(opti_slider.val)
    _redraw_opti(f)
    _sync_offsets[0] = f
    _sync_updating[0] = True
    sync_slider.set_val(0.0)
    _sync_updating[0] = False
    fig.canvas.draw_idle()


def update_orb(val):
    """Individual orb slider moved — redraw, store new offset, reset sync to 0."""
    if _sync_updating[0]:
        return
    f = int(orb_slider.val)
    _redraw_orb(f)
    _sync_offsets[1] = f
    _sync_updating[0] = True
    sync_slider.set_val(0.0)
    _sync_updating[0] = False
    fig.canvas.draw_idle()


def update_sync(val):
    """Sync slider scrolls both streams forward/backward from their stored offsets."""
    if _sync_updating[0]:
        return
    _sync_updating[0] = True
    t = float(sync_slider.val)
    f_opti = int(np.clip(_sync_offsets[0] + round(t * opti_fps), 0, n_opti_shared - 1))
    f_orb  = int(np.clip(_sync_offsets[1] + round(t * orb_fps),  0, n_orb_shared  - 1))
    opti_slider.set_val(f_opti)
    orb_slider.set_val(f_orb)
    _opti_frac[0] = float(f_opti)
    _orb_frac[0]  = float(f_orb)
    _redraw_opti(f_opti)
    _redraw_orb(f_orb)
    _sync_updating[0] = False
    fig.canvas.draw_idle()


def on_thresh_opti(val):
    thr  = int(thresh_opti_sl.val)
    led  = (med_opti > thr).astype(np.float32)
    line_opti.set_ydata(led * 1.06)
    line_opti.set_label(f"Optitrack (thr={thr})")
    ax_sig.legend(fontsize=8, facecolor="#222222", labelcolor="white", loc="upper right")
    info_text.set_text(compute_info(thr, int(thresh_orb_sl.val)))
    fig.canvas.draw_idle()


def on_thresh_orb(val):
    thr  = int(thresh_orb_sl.val)
    led  = (med_orbbec > thr).astype(np.float32)
    line_orb.set_ydata(led * 1.12)
    line_orb.set_label(f"Orbbec (thr={thr})")
    ax_sig.legend(fontsize=8, facecolor="#222222", labelcolor="white", loc="upper right")
    info_text.set_text(compute_info(int(thresh_opti_sl.val), thr))
    fig.canvas.draw_idle()


def on_play(event):
    global playing
    playing = not playing
    if playing:
        btn_play.label.set_text("⏸  Stop")
        btn_play.ax.set_facecolor("#552222")
        _last_tick[0] = time.perf_counter()
        _opti_frac[0] = float(opti_slider.val)
        _orb_frac[0]  = float(orb_slider.val)
        # snapshot current positions as sync offsets; reset sync to 0
        _sync_offsets[0] = int(opti_slider.val)
        _sync_offsets[1] = int(orb_slider.val)
        _sync_updating[0] = True
        sync_slider.set_val(0.0)
        _sync_updating[0] = False
    else:
        btn_play.label.set_text("▶  Play")
        btn_play.ax.set_facecolor("#225522")
    fig.canvas.draw_idle()


def on_reset(event):
    global playing
    playing = False
    btn_play.label.set_text("▶  Play")
    btn_play.ax.set_facecolor("#225522")
    _sync_updating[0] = True
    opti_slider.set_val(0)
    orb_slider.set_val(0)
    sync_slider.set_val(0.0)
    _sync_updating[0] = False
    _sync_offsets[0] = 0
    _sync_offsets[1] = 0
    _opti_frac[0] = 0.0
    _orb_frac[0]  = 0.0
    fig.canvas.draw_idle()


def _tick():
    global playing
    if not playing:
        return
    now  = time.perf_counter()
    dt   = now - _last_tick[0]
    _last_tick[0] = now

    _opti_frac[0] += dt * opti_fps
    _orb_frac[0]  += dt * orb_fps

    new_opti = int(_opti_frac[0])
    new_orb  = int(_orb_frac[0])

    stopped = False
    if new_opti >= n_opti_shared - 1:
        new_opti = n_opti_shared - 1
        _opti_frac[0] = float(new_opti)
        stopped = True
    if new_orb >= n_orb_shared - 1:
        new_orb = n_orb_shared - 1
        _orb_frac[0] = float(new_orb)
        stopped = True

    opti_changed = (new_opti != int(opti_slider.val))
    orb_changed  = (new_orb  != int(orb_slider.val))

    if opti_changed:
        opti_slider.eventson = False
        opti_slider.set_val(new_opti)
        opti_slider.eventson = True
        _redraw_opti(new_opti)

    if orb_changed:
        orb_slider.eventson = False
        orb_slider.set_val(new_orb)
        orb_slider.eventson = True
        _redraw_orb(new_orb)

    if opti_changed or orb_changed:
        # sync slider shows elapsed time from play-start offsets
        elapsed = (new_opti - _sync_offsets[0]) / opti_fps if opti_fps > 0 else 0.0
        _sync_updating[0] = True
        sync_slider.eventson = False
        sync_slider.set_val(np.clip(elapsed, 0.0, total_sync_t))
        sync_slider.eventson = True
        _sync_updating[0] = False
        fig.canvas.draw_idle()

    if stopped:
        playing = False
        btn_play.label.set_text("▶  Play")
        btn_play.ax.set_facecolor("#225522")
        fig.canvas.draw_idle()


def on_step(direction):
    """Advance or retreat both streams by 1 frame each."""
    new_opti = int(np.clip(opti_slider.val + direction, 0, n_opti_shared - 1))
    new_orb  = int(np.clip(orb_slider.val  + direction * round(orb_fps / opti_fps)
                           if opti_fps > 0 else orb_slider.val + direction,
                           0, n_orb_shared - 1))
    opti_slider.set_val(new_opti)
    orb_slider.set_val(new_orb)
    _opti_frac[0] = float(new_opti)
    _orb_frac[0]  = float(new_orb)


def on_save(event):
    """Render the full GUI to an MP4 from current position until either stream ends."""
    global playing
    playing = False
    btn_play.label.set_text("▶  Play")
    btn_play.ax.set_facecolor("#225522")

    start_opti = int(opti_slider.val)
    start_orb  = int(orb_slider.val)

    opti_rem_t = (n_opti_shared - 1 - start_opti) / opti_fps if opti_fps > 0 else 0
    orb_rem_t  = (n_orb_shared  - 1 - start_orb ) / orb_fps  if orb_fps  > 0 else 0
    total_t    = min(opti_rem_t, orb_rem_t)
    n_rec      = int(total_t * RECORD_FPS) + 1

    def _grab():
        bio = _io.BytesIO()
        fig.savefig(bio, format="rgba", dpi=fig.dpi)
        bio.seek(0)
        raw = np.frombuffer(bio.read(), dtype=np.uint8)
        fw  = int(round(fig.get_figwidth()  * fig.dpi))
        fh  = int(round(fig.get_figheight() * fig.dpi))
        return raw.reshape(fh, fw, 4), fw, fh

    _redraw_opti(start_opti)
    _redraw_orb(start_orb)
    probe, W, H = _grab()
    print(f"[save] frame size: {W}x{H}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(SAVE_PATH_OPTI, fourcc, float(RECORD_FPS), (W, H))
    if not out.isOpened():
        print(f"ERROR: could not open VideoWriter for {SAVE_PATH_OPTI}")
        return

    print(f"Saving {n_rec} frames @ {RECORD_FPS} fps  →  {SAVE_PATH_OPTI}")
    btn_save.label.set_text(f"Saving 0/{n_rec}")
    btn_save.ax.set_facecolor("#553300")
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    for i in range(n_rec):
        t      = i / RECORD_FPS
        f_opti = min(start_opti + int(round(t * opti_fps)), n_opti_shared - 1)
        f_orb  = min(start_orb  + int(round(t * orb_fps)),  n_orb_shared  - 1)

        opti_slider.eventson = False
        orb_slider.eventson  = False
        opti_slider.set_val(f_opti)
        orb_slider.set_val(f_orb)
        opti_slider.eventson = True
        orb_slider.eventson  = True

        _redraw_opti(f_opti)
        _redraw_orb(f_orb)

        if i % 15 == 0:
            btn_save.label.set_text(f"Saving {i}/{n_rec}")
            print(f"\r  frame {i:4d}/{n_rec}", end="", flush=True)

        frame_rgba, _, _ = _grab()
        out.write(cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR))

        if f_opti >= n_opti_shared - 1 or f_orb >= n_orb_shared - 1:
            break

    out.release()
    print(f"\n  Done  →  {SAVE_PATH_OPTI}")
    btn_save.label.set_text("[S]  Save")
    btn_save.ax.set_facecolor("#334433")
    _opti_frac[0] = float(opti_slider.val)
    _orb_frac[0]  = float(orb_slider.val)
    fig.canvas.draw_idle()


# ─── 11. Wire up ─────────────────────────────────────────────────────────────
opti_slider.on_changed(update_opti)
orb_slider.on_changed(update_orb)
sync_slider.on_changed(update_sync)
thresh_opti_sl.on_changed(on_thresh_opti)
thresh_orb_sl.on_changed(on_thresh_orb)
btn_play.on_clicked(on_play)
btn_reset.on_clicked(on_reset)
btn_minus.on_clicked(lambda e: on_step(-1))
btn_plus.on_clicked(lambda e: on_step(+1))
btn_save.on_clicked(on_save)

timer = fig.canvas.new_timer(interval=33)
timer.add_callback(_tick)
timer.start()


def _on_key(event):
    if event.key == "q":
        plt.close(fig)
    elif event.key == "right":
        on_step(+1)
    elif event.key == "left":
        on_step(-1)


fig.canvas.mpl_connect("key_press_event", _on_key)

plt.show()
timer.stop()
