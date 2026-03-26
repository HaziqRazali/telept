"""
Interactive LED sync viewer.

Layout
------
  Top            : LED blink signal plot (C3D blue | MP4 orange) + threshold slider
  Bottom-left    : LED1.mp4 (full, 1/8 scale) with LED1_cropped overlay + red box
                   One slider controls both MP4 streams.
  Bottom-right   : C3D 3D marker scatter viewer + frame slider
  Bottom controls: [Play/Stop]  [Reset]

Key bindings: q = quit

Usage
-----
    python sync_led.py
"""

import io as _io
import numpy as np
import cv2
import ezc3d
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── paths ─────────────────────────────────────────────────────────────────────
C3D_PATH    = "/home/haziq/datasets/telept/data/mocap_recordings/LED1.c3d"
FULL_PATH   = "/home/haziq/datasets/telept/data/mocap_recordings/LED1.mp4"
CROP_PATH   = "/home/haziq/datasets/telept/data/mocap_recordings/LED1_cropped.mp4"
INIT_THRESH = 75

# crop box in full-frame coords
CROP_X1, CROP_Y1 = 766, 2244
CROP_X2, CROP_Y2 = 806, 2284    # 40x40 px
FULL_SCALE  = 8                 # cache full frames at 1/FULL_SCALE resolution
SAVE_PATH   = "/home/haziq/datasets/telept/data/mocap_recordings/sync_led_recording.mp4"
RECORD_FPS  = 30                # output video fps

# ─── 1. Load C3D ──────────────────────────────────────────────────────────────
print("Loading C3D ...")
c        = ezc3d.c3d(C3D_PATH)
c3d_fps  = float(c["parameters"]["POINT"]["RATE"]["value"][0])
xyz      = c["data"]["points"][:3]                          # [3, M, F]
n_c3d    = xyz.shape[2]
n_vis    = np.all(np.isfinite(xyz), axis=0).sum(axis=0)    # [F]
led_c3d  = (n_vis > 0).astype(np.float32)
t_c3d    = np.arange(n_c3d) / c3d_fps
print(f"  C3D: {n_c3d} frames @ {c3d_fps} fps  ({t_c3d[-1]:.3f}s)")

all_pts     = xyz.reshape(3, -1)
finite_mask = np.all(np.isfinite(all_pts), axis=0)
if finite_mask.any():
    fp  = all_pts[:, finite_mask]
    lo  = np.percentile(fp, 1,  axis=1)
    hi  = np.percentile(fp, 99, axis=1)
    pad = np.maximum((hi - lo) * 0.20, 0.05)
    c3d_bounds = np.stack([lo - pad, hi + pad], axis=1)
else:
    c3d_bounds = np.array([[-1, 1], [-1, 1], [-1, 1]], dtype=float)

# ─── 2. Load crop MP4 (cache frames + medians) ────────────────────────────────
print("Loading cropped MP4 ...")
cap_c   = cv2.VideoCapture(CROP_PATH)
mp4_fps = cap_c.get(cv2.CAP_PROP_FPS)
n_mp4   = int(cap_c.get(cv2.CAP_PROP_FRAME_COUNT))
medians     = np.empty(n_mp4, dtype=np.float32)
crop_frames = []

for i in range(n_mp4):
    ret, frame = cap_c.read()
    if not ret:
        medians[i:] = 0.0
        blank = np.zeros_like(crop_frames[-1]) if crop_frames else np.zeros((40, 40), np.uint8)
        crop_frames.extend([blank] * (n_mp4 - i))
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    medians[i] = float(np.median(gray))
    crop_frames.append(gray)          # store as 2-D grayscale
    if i % 60 == 0:
        print(f"  crop frame {i}/{n_mp4}", end="\r")
cap_c.release()
t_mp4 = np.arange(n_mp4) / mp4_fps
print(f"  Crop MP4: {n_mp4} frames @ {mp4_fps:.4f} fps  ({t_mp4[-1]:.3f}s)    ")

# ─── 3. Load full MP4 at 1/FULL_SCALE ────────────────────────────────────────
print(f"Loading full MP4 (1/{FULL_SCALE} scale) ...")
cap_f    = cv2.VideoCapture(FULL_PATH)
n_full   = int(cap_f.get(cv2.CAP_PROP_FRAME_COUNT))
full_fps = cap_f.get(cv2.CAP_PROP_FPS)
fw_orig  = int(cap_f.get(cv2.CAP_PROP_FRAME_WIDTH))
fh_orig  = int(cap_f.get(cv2.CAP_PROP_FRAME_HEIGHT))
fw_disp  = fw_orig // FULL_SCALE
fh_disp  = fh_orig // FULL_SCALE

# scaled crop box coords
cx1 = CROP_X1 // FULL_SCALE
cy1 = CROP_Y1 // FULL_SCALE
cx2 = CROP_X2 // FULL_SCALE
cy2 = CROP_Y2 // FULL_SCALE

full_frames = []
for i in range(n_full):
    ret, frame = cap_f.read()
    if not ret:
        blank = np.zeros((fh_disp, fw_disp, 3), np.uint8)
        full_frames.extend([blank] * (n_full - i))
        break
    small = cv2.resize(frame, (fw_disp, fh_disp), interpolation=cv2.INTER_AREA)
    full_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    if i % 60 == 0:
        print(f"  full frame {i}/{n_full}", end="\r")
cap_f.release()
print(f"  Full MP4: {n_full} frames @ {full_fps:.4f} fps  (disp {fw_disp}x{fh_disp})    ")

n_shared = min(n_mp4, n_full)   # unified slider range

# ─── helper: compose display frame ─────────────────────────────────────────────
BOX_COLOR  = (255, 0, 0)
BOX_THICK  = max(1, FULL_SCALE // 4)
INSET_SCALE = 3                           # enlarge crop by this factor
INSET_W    = (CROP_X2 - CROP_X1) * INSET_SCALE   # 40 * 3 = 120 px
INSET_H    = (CROP_Y2 - CROP_Y1) * INSET_SCALE   # 40 * 3 = 120 px
INSET_MARGIN = 4                          # px gap from frame edge
inset_x1   = fw_disp - INSET_W - INSET_MARGIN
inset_y1   = fh_disp - INSET_H - INSET_MARGIN
inset_x2   = fw_disp - INSET_MARGIN
inset_y2   = fh_disp - INSET_MARGIN

def compose_frame(f):
    """Full frame (display scale) with 5x grayscale crop inset at bottom-right."""
    base = full_frames[f].copy()
    cf   = min(f, len(crop_frames) - 1)
    # enlarge crop 5x, keep grayscale → convert to RGB for display
    big_gray = cv2.resize(crop_frames[cf], (INSET_W, INSET_H),
                          interpolation=cv2.INTER_NEAREST)
    inset = np.stack([big_gray, big_gray, big_gray], axis=2)
    base[inset_y1:inset_y2, inset_x1:inset_x2] = inset
    # red box around the inset
    cv2.rectangle(base, (inset_x1, inset_y1), (inset_x2, inset_y2), BOX_COLOR, BOX_THICK)
    return base

# ─── 4. Figure / axes layout ──────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10), facecolor="#111111")
fig.canvas.manager.set_window_title("LED Sync Viewer")

# signal plot
ax_sig = fig.add_axes([0.05, 0.545, 0.92, 0.39])
ax_sig.set_facecolor("#1a1a1a")
ax_sig.set_xlabel("time (s)", color="#aaaaaa", fontsize=8)
ax_sig.set_ylabel("LED on/off", color="#aaaaaa", fontsize=8)
ax_sig.set_title("LED blink signals — C3D (blue) | MP4 (orange)", color="white", fontsize=10)
ax_sig.tick_params(colors="#888888", labelsize=7)
for sp in ax_sig.spines.values():
    sp.set_edgecolor("#444444")

line_c3d, = ax_sig.step(t_c3d, led_c3d, where="post",
                         color="#4499ff", lw=1.0, label="C3D")
led_mp4_init = (medians > INIT_THRESH).astype(np.float32)
line_mp4, = ax_sig.step(t_mp4, led_mp4_init * 1.05, where="post",
                         color="#ffaa33", lw=1.0, alpha=0.85,
                         label=f"MP4 (thr={INIT_THRESH})")
ax_sig.set_ylim(-0.15, 1.35)
ax_sig.legend(fontsize=8, facecolor="#222222", labelcolor="white", loc="upper right")

vline_mp4 = ax_sig.axvline(0, color="#ffaa33", lw=1.2, ls="--", alpha=0.7)
vline_c3d = ax_sig.axvline(0, color="#4499ff", lw=1.2, ls="--", alpha=0.7)

# threshold slider
ax_thr = fig.add_axes([0.10, 0.485, 0.82, 0.028], facecolor="#2a2a2a")
thresh_slider = mwidgets.Slider(ax_thr, "Median thr", 0, 255,
                                valinit=INIT_THRESH, valstep=1, color="#dd8833")
thresh_slider.label.set_color("white")
thresh_slider.valtext.set_color("white")

# info text
ax_info = fig.add_axes([0.05, 0.445, 0.92, 0.032])
ax_info.axis("off")
info_text = ax_info.text(0.5, 0.5, "", ha="center", va="center",
                          color="#cccccc", fontsize=8,
                          transform=ax_info.transAxes, fontfamily="monospace")

# bottom-left: full MP4 with crop overlay
ax_vid = fig.add_axes([0.02, 0.12, 0.46, 0.30])
ax_vid.axis("off")
ax_vid.set_facecolor("#000000")
im_full = ax_vid.imshow(compose_frame(0), aspect="auto", interpolation="bilinear")
title_vid = ax_vid.set_title(
    "LED1.mp4  t=0.000s  |  frame 0  (crop overlay + red box)",
    color="white", fontsize=9, pad=3)

ax_mp4_sl = fig.add_axes([0.02, 0.065, 0.46, 0.028], facecolor="#2a2a2a")
mp4_slider = mwidgets.Slider(ax_mp4_sl, "MP4 frame", 0, n_shared - 1,
                              valinit=0, valstep=1, color="#ffaa33")
mp4_slider.label.set_color("white")
mp4_slider.valtext.set_color("white")

# bottom-right: C3D 3D viewer
ax_c3d = fig.add_axes([0.53, 0.10, 0.45, 0.32], projection="3d")
ax_c3d.set_facecolor("#0a0a0a")
for pane in (ax_c3d.xaxis.pane, ax_c3d.yaxis.pane, ax_c3d.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#333333")
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


pts0 = get_visible_pts(0)
if len(pts0):
    sc = ax_c3d.scatter(pts0[:, 0], pts0[:, 1], pts0[:, 2],
                        c="#ffee00", s=40, alpha=0.95, depthshade=False)
else:
    sc = ax_c3d.scatter([], [], [], c="#ffee00", s=40, depthshade=False)
title_c3d = ax_c3d.set_title("C3D  t=0.000s  |  frame 0", color="white", fontsize=9, pad=3)

ax_c3d_sl = fig.add_axes([0.53, 0.055, 0.45, 0.028], facecolor="#2a2a2a")
c3d_slider = mwidgets.Slider(ax_c3d_sl, "C3D frame", 0, n_c3d - 1,
                              valinit=0, valstep=1, color="#4499ff")
c3d_slider.label.set_color("white")
c3d_slider.valtext.set_color("white")

# Play/Stop and Reset buttons
ax_play = fig.add_axes([0.10, 0.010, 0.14, 0.032])
btn_play = mwidgets.Button(ax_play, "▶  Play", color="#225522", hovercolor="#338833")
btn_play.label.set_color("white")
btn_play.label.set_fontsize(9)

ax_reset = fig.add_axes([0.28, 0.010, 0.10, 0.032])
btn_reset = mwidgets.Button(ax_reset, "⟳ Reset", color="#222255", hovercolor="#333388")
btn_reset.label.set_color("white")
btn_reset.label.set_fontsize(9)

ax_minus = fig.add_axes([0.42, 0.010, 0.055, 0.032])
btn_minus = mwidgets.Button(ax_minus, "◀  −1", color="#333333", hovercolor="#555555")
btn_minus.label.set_color("white")
btn_minus.label.set_fontsize(9)

ax_plus = fig.add_axes([0.48, 0.010, 0.055, 0.032])
btn_plus = mwidgets.Button(ax_plus, "+1  ▶", color="#333333", hovercolor="#555555")
btn_plus.label.set_color("white")
btn_plus.label.set_fontsize(9)

ax_save = fig.add_axes([0.575, 0.010, 0.12, 0.032])
btn_save = mwidgets.Button(ax_save, "[S]  Save", color="#334433", hovercolor="#446644")
btn_save.label.set_color("white")
btn_save.label.set_fontsize(9)

# ─── 5. State ─────────────────────────────────────────────────────────────────
playing   = False
_last_tick = [0.0]
_mp4_frac  = [0.0]   # fractional frame accumulators
_c3d_frac  = [0.0]

# Step sizes: 1 MP4 frame ≙ round(c3d_fps/mp4_fps) C3D frames
MP4_STEP = 1
C3D_STEP = max(1, round(c3d_fps / mp4_fps))   # = 2 for 60/~29.26


# ─── 6. Callbacks ─────────────────────────────────────────────────────────────
def compute_info(thresh):
    led = (medians > thresh).astype(np.float32)
    return (
        f"MP4  ON:{led.sum()/mp4_fps:.2f}s  "
        f"trans:{int(np.abs(np.diff(led)).sum())}   |   "
        f"C3D  ON:{led_c3d.sum()/c3d_fps:.2f}s  "
        f"trans:{int(np.abs(np.diff(led_c3d)).sum())}"
    )


info_text.set_text(compute_info(INIT_THRESH))


def _redraw_mp4(f):
    t   = f / mp4_fps
    med = medians[min(f, len(medians) - 1)]
    im_full.set_data(compose_frame(f))
    title_vid.set_text(f"LED1.mp4  t={t:.3f}s  |  frame {f}  |  median={med:.1f}")
    vline_mp4.set_xdata([t, t])
    print(f"MP4 frame {f:4d}  t={t:.3f}s  median={med:.1f}", flush=True)


def _redraw_c3d(f):
    t   = f / c3d_fps
    pts = get_visible_pts(f)
    if len(pts):
        sc._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
    else:
        sc._offsets3d = (np.array([]), np.array([]), np.array([]))
    title_c3d.set_text(
        f"C3D  t={t:.3f}s  |  frame {f}  |  LED={'ON' if led_c3d[f] else 'off'}")
    vline_c3d.set_xdata([t, t])


def update_mp4(val):
    _redraw_mp4(int(mp4_slider.val))
    fig.canvas.draw_idle()


def update_c3d(val):
    _redraw_c3d(int(c3d_slider.val))
    fig.canvas.draw_idle()


def on_thresh(val):
    thresh = int(thresh_slider.val)
    led    = (medians > thresh).astype(np.float32)
    line_mp4.set_ydata(led * 1.05)
    line_mp4.set_label(f"MP4 (thr={thresh})")
    ax_sig.legend(fontsize=8, facecolor="#222222", labelcolor="white", loc="upper right")
    info_text.set_text(compute_info(thresh))
    fig.canvas.draw_idle()


def on_play(event):
    global playing
    playing = not playing
    if playing:
        btn_play.label.set_text("⏸  Stop")
        btn_play.ax.set_facecolor("#552222")
        _last_tick[0] = time.perf_counter()
        _mp4_frac[0]  = float(mp4_slider.val)
        _c3d_frac[0]  = float(c3d_slider.val)
    else:
        btn_play.label.set_text("▶  Play")
        btn_play.ax.set_facecolor("#225522")
    fig.canvas.draw_idle()


def on_reset(event):
    global playing
    playing = False
    btn_play.label.set_text("▶  Play")
    btn_play.ax.set_facecolor("#225522")
    mp4_slider.set_val(0)
    c3d_slider.set_val(0)
    _mp4_frac[0] = 0.0
    _c3d_frac[0] = 0.0
    fig.canvas.draw_idle()


def _tick():
    """33 ms timer callback — advances both streams at their native fps."""
    global playing
    if not playing:
        return
    now  = time.perf_counter()
    dt   = now - _last_tick[0]
    _last_tick[0] = now

    _mp4_frac[0] += dt * mp4_fps
    _c3d_frac[0] += dt * c3d_fps

    new_mp4 = int(_mp4_frac[0])
    new_c3d = int(_c3d_frac[0])

    stopped = False
    if new_mp4 >= n_shared - 1:
        new_mp4 = n_shared - 1
        _mp4_frac[0] = float(new_mp4)
        stopped = True
    if new_c3d >= n_c3d - 1:
        new_c3d = n_c3d - 1
        _c3d_frac[0] = float(new_c3d)
        stopped = True

    mp4_changed = (new_mp4 != int(mp4_slider.val))
    c3d_changed = (new_c3d != int(c3d_slider.val))

    if mp4_changed:
        mp4_slider.eventson = False
        mp4_slider.set_val(new_mp4)
        mp4_slider.eventson = True
        _redraw_mp4(new_mp4)

    if c3d_changed:
        c3d_slider.eventson = False
        c3d_slider.set_val(new_c3d)
        c3d_slider.eventson = True
        _redraw_c3d(new_c3d)

    if mp4_changed or c3d_changed:
        fig.canvas.draw_idle()

    if stopped:
        playing = False
        btn_play.label.set_text("▶  Play")
        btn_play.ax.set_facecolor("#225522")
        fig.canvas.draw_idle()


def on_step(direction):
    """Advance (+1) or retreat (-1) both streams by 1 MP4 frame / C3D_STEP C3D frames."""
    new_mp4 = int(np.clip(mp4_slider.val + direction * MP4_STEP, 0, n_shared - 1))
    new_c3d = int(np.clip(c3d_slider.val + direction * C3D_STEP, 0, n_c3d  - 1))
    mp4_slider.set_val(new_mp4)
    c3d_slider.set_val(new_c3d)
    _mp4_frac[0] = float(new_mp4)
    _c3d_frac[0] = float(new_c3d)


def on_save(event):
    """Render the full GUI to an MP4 from current position until either stream ends."""
    global playing
    # stop any live playback first
    playing = False
    btn_play.label.set_text("▶  Play")
    btn_play.ax.set_facecolor("#225522")

    start_mp4 = int(mp4_slider.val)
    start_c3d = int(c3d_slider.val)

    mp4_rem_t = (n_shared - 1 - start_mp4) / mp4_fps
    c3d_rem_t = (n_c3d   - 1 - start_c3d) / c3d_fps
    total_t   = min(mp4_rem_t, c3d_rem_t)
    n_rec     = int(total_t * RECORD_FPS) + 1

    # Use savefig→RGBA — completely independent of TkAgg canvas stride / HiDPI
    def _grab_frame():
        bio = _io.BytesIO()
        fig.savefig(bio, format="rgba", dpi=fig.dpi)
        bio.seek(0)
        raw = np.frombuffer(bio.read(), dtype=np.uint8)
        fw  = int(round(fig.get_figwidth()  * fig.dpi))
        fh  = int(round(fig.get_figheight() * fig.dpi))
        return raw.reshape(fh, fw, 4), fw, fh

    # probe once to get W, H
    _redraw_mp4(start_mp4)
    _redraw_c3d(start_c3d)
    probe, W, H = _grab_frame()
    print(f"[save] frame size: {W}x{H}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(SAVE_PATH, fourcc, float(RECORD_FPS), (W, H))
    if not out.isOpened():
        print(f"ERROR: could not open VideoWriter for {SAVE_PATH}")
        return

    print(f"Saving {n_rec} frames @ {RECORD_FPS} fps  →  {SAVE_PATH}")
    btn_save.label.set_text(f"Saving 0/{n_rec}")
    btn_save.ax.set_facecolor("#553300")
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    for i in range(n_rec):
        t     = i / RECORD_FPS
        f_mp4 = min(start_mp4 + int(round(t * mp4_fps)), n_shared - 1)
        f_c3d = min(start_c3d + int(round(t * c3d_fps)), n_c3d  - 1)

        # update sliders silently (avoid re-entrant callbacks)
        mp4_slider.eventson = False
        c3d_slider.eventson = False
        mp4_slider.set_val(f_mp4)
        c3d_slider.set_val(f_c3d)
        mp4_slider.eventson = True
        c3d_slider.eventson = True

        _redraw_mp4(f_mp4)
        _redraw_c3d(f_c3d)

        if i % 15 == 0:
            btn_save.label.set_text(f"Saving {i}/{n_rec}")
            print(f"\r  frame {i:4d}/{n_rec}", end="", flush=True)

        frame_rgba, _, _ = _grab_frame()
        out.write(cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR))

        if f_mp4 >= n_shared - 1 or f_c3d >= n_c3d - 1:
            break

    out.release()
    print(f"\n  Done  →  {SAVE_PATH}")
    btn_save.label.set_text("[S]  Save")
    btn_save.ax.set_facecolor("#334433")
    _mp4_frac[0] = float(mp4_slider.val)
    _c3d_frac[0] = float(c3d_slider.val)
    fig.canvas.draw_idle()


mp4_slider.on_changed(update_mp4)
c3d_slider.on_changed(update_c3d)
thresh_slider.on_changed(on_thresh)
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
