"""
Visualize a .c3d motion capture file in 3D with a time scrubber.

Usage:
    conda activate orbbec
    python vis_c3d.py
"""

import numpy as np
import ezc3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

C3D_PATH = "/data/telept/my_scripts/orbbec_scripts/Take 2014-10-02 05.49.50 AM.c3d"

# ── Load data ─────────────────────────────────────────────────────────────────
c = ezc3d.c3d(C3D_PATH)
points   = c["data"]["points"]          # (4, n_markers, n_frames)
labels   = c["parameters"]["POINT"]["LABELS"]["value"]
fps      = c["header"]["points"]["frame_rate"]
n_frames = points.shape[2]
n_markers = points.shape[1]

# XYZ in metres
xyz = points[:3]   # (3, n_markers, n_frames)

# Replace zero-residual (occluded) markers with NaN
residuals = points[3]                   # (n_markers, n_frames)
occluded  = (residuals == 0)
xyz[:, occluded] = np.nan

# Axis limits (ignore NaN)
pad = 0.05
xmin, xmax = np.nanmin(xyz[0]), np.nanmax(xyz[0])
ymin, ymax = np.nanmin(xyz[1]), np.nanmax(xyz[1])
zmin, zmax = np.nanmin(xyz[2]), np.nanmax(xyz[2])
margin = max(xmax - xmin, ymax - ymin, zmax - zmin) * pad

# Colors per marker (cycle)
cmap   = plt.get_cmap("tab20")
colors = [cmap(i % 20) for i in range(n_markers)]

# ── Build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(bottom=0.18)

ax = fig.add_subplot(111, projection="3d")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_xlim(xmin - margin, xmax + margin)
ax.set_ylim(ymin - margin, ymax + margin)
ax.set_zlim(zmin - margin, zmax + margin)

# Initial scatter for each marker
scatters = []
for i in range(n_markers):
    sc = ax.scatter([], [], [], s=60, color=colors[i], label=labels[i], depthshade=True)
    scatters.append(sc)

ax.legend(loc="upper left", fontsize=6, ncol=2, framealpha=0.5)
title = ax.set_title(f"Frame 0 / {n_frames - 1}  |  t = 0.000 s")


def update_frame(frame_idx):
    frame_idx = int(frame_idx)
    for i, sc in enumerate(scatters):
        x = xyz[0, i, frame_idx]
        y = xyz[1, i, frame_idx]
        z = xyz[2, i, frame_idx]
        if np.isnan(x):
            sc._offsets3d = ([], [], [])
        else:
            sc._offsets3d = ([x], [y], [z])
    t = frame_idx / fps
    title.set_text(f"Frame {frame_idx} / {n_frames - 1}  |  t = {t:.3f} s")
    fig.canvas.draw_idle()


# ── Slider ────────────────────────────────────────────────────────────────────
ax_slider = fig.add_axes([0.12, 0.06, 0.76, 0.04])
slider = Slider(
    ax=ax_slider,
    label="Frame",
    valmin=0,
    valmax=n_frames - 1,
    valinit=0,
    valstep=1,
    color="steelblue",
)
slider.on_changed(update_frame)

# ── Play/Pause with spacebar ──────────────────────────────────────────────────
play_state = {"running": False, "ani": None}


def animate(_):
    if play_state["running"]:
        next_frame = int(slider.val + 1)
        if next_frame >= n_frames:
            next_frame = 0
        slider.set_val(next_frame)


ani = animation.FuncAnimation(fig, animate, interval=int(1000 / fps), blit=False)
play_state["ani"] = ani


def on_key(event):
    if event.key == " ":
        play_state["running"] = not play_state["running"]
        state_str = "Playing" if play_state["running"] else "Paused"
        print(f"{state_str} (frame {int(slider.val)})")


fig.canvas.mpl_connect("key_press_event", on_key)

update_frame(0)
print(f"Loaded: {n_frames} frames @ {fps} fps  |  {n_markers} markers")
print("Spacebar = play/pause  |  Drag slider to scrub")
plt.show()
