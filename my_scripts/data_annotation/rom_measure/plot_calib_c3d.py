"""Interactive 3D viewer for mocap_rgb_calib_sync01.c3d.

Draws all mocap markers as a 3D scatter, highlights the 6 calibration-board
markers (Board1..Board6), and lets you scrub through time with a frame slider
(+ play/pause button).

Usage:
    # interactive (needs a GUI display OR a Jupyter notebook w/ %matplotlib widget)
    conda run -n telept_server python plot_calib_c3d.py [path.c3d]

    # headless: no display needed
    conda run -n telept_server python plot_calib_c3d.py [path.c3d] --save out.mp4
    conda run -n telept_server python plot_calib_c3d.py [path.c3d] --save out.gif
    conda run -n telept_server python plot_calib_c3d.py [path.c3d] --frames png_dir

Headless options:
    --save FILE     render the whole recording to FILE (.mp4 or .gif) and exit
    --frames DIR    save one PNG per (sampled) frame into DIR and exit
    --every N       sample every N-th frame for --save/--frames (default 1)
    --view AZ,EL    fixed 3D view, azimuth,elevation degrees (default 45,25)

Interactive mode needs a GUI backend (TkAgg/QtAgg) or a Jupyter notebook
with `%matplotlib widget` (see plot_calib_c3d.ipynb).
"""
import sys
import os
import argparse

import numpy as np
import ezc3d
import matplotlib

# switch to the headless backend *before* importing pyplot when saving
try:
    _headless = "--save" in sys.argv or "--frames" in sys.argv
    if _headless:
        matplotlib.use("Agg")
except Exception:
    pass

import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.widgets import Slider, Button          # noqa: E402
from matplotlib.animation import FuncAnimation         # noqa: E402
from matplotlib.patches import Patch                   # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402

DEFAULT_PATH = "/data/haziq/telept/mocap_rgb_calib_sync01.c3d"


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
def load_c3d(path):
    c = ezc3d.c3d(path)
    pts = c["data"]["points"]          # [4, M, F] X,Y,Z,residual
    xyz = pts[:3].astype(float)        # [3, M, F]
    res = pts[3].astype(float)         # [M, F]
    M, F = xyz.shape[1], xyz.shape[2]
    rate = float(c["parameters"]["POINT"]["RATE"]["value"][0])
    units = c["parameters"]["POINT"]["UNITS"]["value"][0]

    label_keys = sorted(k for k in c["parameters"]["POINT"] if k.startswith("LABELS"))
    labels = []
    for k in label_keys:
        labels.extend(c["parameters"]["POINT"][k]["value"])
    labels = labels[:M]
    return xyz, res, labels, rate, units


def make_figure(xyz, res, labels, rate, units, view=(45, 25)):
    """Build the 3D scatter figure; returns (fig, update) where update(f) redraws frame f."""
    M, F = xyz.shape[1], xyz.shape[2]
    board_idx = [i for i, l in enumerate(labels) if str(l).startswith("Board")]
    noise_idx = [i for i in range(M) if i not in board_idx]
    # board markers in outline order: Board1 -> Board2 -> ... -> Board6
    board_order = [labels.index(f"Board{i}") for i in range(1, 7) if f"Board{i}" in labels]

    # ---- static axis limits from all valid points ----
    valid = np.isfinite(xyz).all(axis=0) & (res >= 0)
    if valid.any():
        lo = xyz[:, valid].min(axis=1)
        hi = xyz[:, valid].max(axis=1)
    else:
        lo, hi = np.zeros(3), np.ones(3)
    pad = (hi - lo) * 0.08 + 1e-3
    lims = [(lo[i] - pad[i], hi[i] + pad[i]) for i in range(3)]

    cmap = plt.get_cmap("tab10")
    board_colors = [cmap(i % 10) for i in range(len(board_idx))]

    fig = plt.figure(figsize=(11, 7.5))
    ax = fig.add_axes([0.08, 0.18, 0.80, 0.72], projection="3d")

    # seed with NaNs sized to each group so per-marker colors match data length
    nan_b = [np.nan] * len(board_idx)
    nan_n = [np.nan] * len(noise_idx)
    scat_board = ax.scatter(nan_b, nan_b, nan_b, s=70, color=board_colors,
                            depthshade=False)
    scat_noise = ax.scatter(nan_n, nan_n, nan_n, s=28, color="0.6", alpha=0.65,
                            depthshade=False)
    # board outline: connect Board1..Board6 in order and close back to Board1
    line_board = ax.plot([], [], [], color="black", lw=1.6, alpha=0.85)[0]
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1]); ax.set_zlim(lims[2])
    ax.set_xlabel(f"X ({units})"); ax.set_ylabel(f"Y ({units})"); ax.set_zlabel(f"Z ({units})")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=view[1], azim=view[0])

    # legend: one patch per board marker
    handles = [Patch(color=board_colors[i], label=str(labels[board_idx[i]]))
               for i in range(len(board_idx))]
    handles.append(Patch(color="0.6", alpha=0.65, label="other markers"))
    handles.append(Line2D([0], [0], color="black", lw=1.6, alpha=0.85,
                          label="board outline"))
    ax.legend(handles=handles, loc="upper left", fontsize=8, ncol=2)

    title = ax.set_title("", fontsize=11)

    def update(f):
        f = int(round(f)) % F
        for scat, idx in ((scat_board, board_idx), (scat_noise, noise_idx)):
            x = xyz[:, idx, f].copy()          # [3, n]
            ok = np.isfinite(x).all(axis=0) & (res[idx, f] >= 0)
            xx = x[0].copy(); yy = x[1].copy(); zz = x[2].copy()
            xx[~ok] = np.nan; yy[~ok] = np.nan; zz[~ok] = np.nan
            scat.set_offsets(np.column_stack([xx, yy]))
            scat.set_3d_properties(zz, zdir="z")
        # board outline: Board1..Board6 in order, then close back to Board1
        if board_order:
            x = xyz[:, board_order, f].copy()
            ok = np.isfinite(x).all(axis=0) & (res[board_order, f] >= 0)
            xs = list(x[0]) + [x[0, 0]]
            ys = list(x[1]) + [x[1, 0]]
            zs = list(x[2]) + [x[2, 0]]
            okc = list(ok) + [ok[0]]
            xs = [v if o else np.nan for v, o in zip(xs, okc)]
            ys = [v if o else np.nan for v, o in zip(ys, okc)]
            zs = [v if o else np.nan for v, o in zip(zs, okc)]
            line_board.set_data(xs, ys)
            line_board.set_3d_properties(zs)
        title.set_text(f"Frame {f} / {F-1}   ({f/rate:.2f} s of {F/rate:.2f} s)")
        fig.canvas.draw_idle()
        return scat_board, scat_noise, line_board

    return fig, update


def main():
    parser = argparse.ArgumentParser(
        description="Plot mocap markers from a .c3d file (interactive or headless).")
    parser.add_argument("path", nargs="?", default=DEFAULT_PATH,
                        help="path to .c3d file")
    parser.add_argument("--save", metavar="FILE", default=None,
                        help="headless: render recording to FILE (.mp4 or .gif)")
    parser.add_argument("--frames", metavar="DIR", default=None,
                        help="headless: save one PNG per frame into DIR")
    parser.add_argument("--every", type=int, default=1,
                        help="sample every N-th frame for --save/--frames (default 1)")
    parser.add_argument("--view", default="45,25",
                        help="3D view as azimuth,elevation degrees (default 45,25)")
    args = parser.parse_args()

    xyz, res, labels, rate, units = load_c3d(args.path)
    M, F = xyz.shape[1], xyz.shape[2]
    azim, elev = (float(x) for x in args.view.split(","))

    board_idx = [i for i, l in enumerate(labels) if str(l).startswith("Board")]
    noise_idx = [i for i in range(M) if i not in board_idx]
    print(f"{args.path}\n  {M} markers, {F} frames @ {rate:.0f} fps ({F/rate:.2f} s), units={units}")
    print(f"  board markers : {[labels[i] for i in board_idx]}")
    print(f"  other markers : {[labels[i] for i in noise_idx]}")

    fig, update = make_figure(xyz, res, labels, rate, units, view=(azim, elev))

    # ---------------- headless modes (no display required) ----------------
    if args.frames:
        os.makedirs(args.frames, exist_ok=True)
        frames = list(range(0, F, max(1, args.every)))
        print(f"rendering {len(frames)} frames -> {args.frames}/ ...")
        for i, f in enumerate(frames):
            update(f)
            fig.savefig(os.path.join(args.frames, f"frame_{f:05d}.png"), dpi=110)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(frames)}")
        print(f"done: {len(frames)} PNGs in {args.frames}")
        return

    if args.save:
        frames = list(range(0, F, max(1, args.every)))
        fps = rate / max(1, args.every)
        anim = FuncAnimation(fig, lambda i: update(frames[i]),
                             frames=range(len(frames)), interval=1000.0 / fps,
                             blit=False, repeat=False, cache_frame_data=False)
        ext = os.path.splitext(args.save)[1].lower()
        try:
            if ext == ".mp4":
                from matplotlib.animation import FFMpegWriter
                anim.save(args.save, writer=FFMpegWriter(fps=fps), dpi=110)
            else:
                from matplotlib.animation import PillowWriter
                anim.save(args.save, writer=PillowWriter(fps=fps), dpi=110)
        except Exception as e:
            print(f"WARN: could not write {args.save}: {e}")
            out2 = os.path.splitext(args.save)[0] + ".gif"
            print(f"falling back to GIF: {out2}")
            from matplotlib.animation import PillowWriter
            anim.save(out2, writer=PillowWriter(fps=fps), dpi=110)
            args.save = out2
        print(f"done: {args.save}")
        return

    # ---------------- interactive mode (needs a display) ----------------
    ax_slider = fig.add_axes([0.13, 0.06, 0.60, 0.035])
    slider = Slider(ax_slider, "Frame", 0, F - 1, valinit=0, valstep=1)
    slider.on_changed(lambda v: update(v))

    ax_play = fig.add_axes([0.76, 0.055, 0.09, 0.05])
    play_btn = Button(ax_play, "Play")
    ax_play_r = fig.add_axes([0.87, 0.055, 0.09, 0.05])
    reset_btn = Button(ax_play_r, "Reset")

    anim = FuncAnimation(fig, lambda _: (update(slider.val + 1),), None,
                         interval=1000.0 / rate, blit=False, repeat=True,
                         save_count=F, cache_frame_data=False)
    anim.event_source.stop()

    def toggle_play(_):
        if anim.event_source is None:
            return
        if anim.event_source.running:
            anim.event_source.stop()
            play_btn.label.set_text("Play")
        else:
            anim.event_source.start()
            play_btn.label.set_text("Pause")

    def reset(_):
        slider.set_val(0)
        anim.event_source.stop()
        play_btn.label.set_text("Play")

    play_btn.on_clicked(toggle_play)
    reset_btn.on_clicked(reset)

    update(0)
    plt.show()
    print("closed.")


if __name__ == "__main__":
    main()
