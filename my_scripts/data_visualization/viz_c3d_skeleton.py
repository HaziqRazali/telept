#!/usr/bin/env python3
"""
Interactive visualizer for Vicon C3D marker data (upper-limb capture).

What the file contains
----------------------
- 28 3-D markers (13 anatomical + 15 cluster "*NN") at 100 Hz.
  NOTE: the file stores *markers only* — the ROTATIONS (segments) section is
  empty, so there are NO saved body segments and NO saved joint angles.
- 12 analog channels (2 force plates: Fx/Fy/Fz + Mx/My/Mz) at 1000 Hz.

What this script does
---------------------
- Loads markers with ezc3d (it reads this file's labels/params/points fine).
- Builds a skeleton from the anatomical markers:
    spine     : T10 -- C7
    neck      : C7  -- CLAV
    clavicle  : CLAV -- RSHO
    upperarm  : RSHO -- RELB
    forearm   : RELB -- WR        (WR = centroid of RWRA, RWRB)
    hand      : WR  -- HF         (HF = centroid of RFRA, RFIN)
- COMPUTES joint angles from the marker geometry (3-D angle at each joint
  between the two adjacent bone segments):
    shoulder : at RSHO between (RSHO->CLAV) and (RSHO->RELB)
    elbow    : at RELB between (RELB->RSHO) and (RELB->WR)
    wrist    : at WR   between (WR->RELB)   and (WR->HF)
- Renders a matplotlib figure with:
    * a 3-D skeleton view (equal aspect, not squashed; drag to rotate)
    * joint-angle time series with a cursor on the current frame
    * a VERTICAL slider to scrub frames (drives both panels)
    * spacebar = play/pause animation, +/- = speed up/down, arrows = step

Usage
-----
    python viz_c3d_skeleton.py [--file PATH] [--start F] [--end F]

Run with an ezc3d-enabled interpreter, e.g.:
    /home/haziq/datasets/telept/my_scripts/data_calibration/pc_calib_tool/.venv/bin/python \
        /home/haziq/datasets/telept/my_scripts/data_visualization/viz_c3d_skeleton.py
"""

import argparse
import sys

import numpy as np

try:
    import ezc3d
except ImportError:
    sys.exit("ezc3d is not installed in this interpreter. Try the ezc3d-enabled "
             "python, e.g. the pc_calib_tool/.venv/bin/python.")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

DEFAULT_FILE = ("/home/haziq/datasets/telept/data/NUS/mocap/"
               "haziq_upperlimb_right2 dynamic 03.c3d")

# ---------------------------------------------------------------------------
# Skeleton definition: (name, markerA, markerB); a group is a list -> centroid
# ---------------------------------------------------------------------------
BONES = [
    ("spine",    ["T10"],            ["C7"]),
    ("neck",     ["C7"],             ["CLAV"]),
    ("clavicle", ["CLAV"],           ["RSHO"]),
    ("upperarm", ["RSHO"],           ["RELB"]),
    ("forearm",  ["RELB"],           ["RWRA", "RWRB"]),
    ("hand",     ["RWRA", "RWRB"],   ["RFRA", "RFIN"]),
]
BONE_COLORS = dict(
    spine="#444444", neck="#444444", clavicle="#1f77b4",
    upperarm="#d62728", forearm="#ff7f0e", hand="#2ca02c",
)
ANGLE_JOINTS = [
    ("shoulder", "#d62728"),
    ("elbow",    "#ff7f0e"),
    ("wrist",    "#2ca02c"),
]
KEY_LABELS = ["T10", "C7", "CLAV", "RSHO", "RELB"]


def _as_group(m):
    return m if isinstance(m, list) else [m]


def resolve_point(P_frame, marker_group, labels_idx):
    """P_frame: (3, M). marker_group: label or list. -> (3,) centroid or nan."""
    pts = []
    for name in _as_group(marker_group):
        i = labels_idx.get(name)
        if i is None:
            continue
        q = P_frame[:, i]
        if np.all(np.isfinite(q)):
            pts.append(q)
    if not pts:
        return np.full(3, np.nan)
    return np.mean(pts, axis=0)


def joint_angle(a, b, c):
    """3-D angle (deg) at b between b->a and b->c."""
    v1 = a - b
    v2 = c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_c3d(path):
    c = ezc3d.c3d(path)
    p = np.array(c.data.points)                 # (4, n_markers, n_frames)
    if p.ndim != 3:
        sys.exit("Unexpected point-data shape: %s" % (p.shape,))
    xyz = p[:3]                                 # (3, M, F)
    try:
        fps = float(c.parameters["POINT"]["RATE"]["value"][0])
    except Exception:
        fps = 100.0
    labels = list(c.parameters["POINT"]["LABELS"]["value"])
    return dict(xyz=xyz, labels=labels, fps=fps, M=p.shape[1], F=p.shape[2])


def compute_angles(xyz, labels):
    """Return dict name -> (F,) angles (rad->deg) for shoulder/elbow/wrist."""
    idx = {l: i for i, l in enumerate(labels)}
    F = xyz.shape[2]
    out = {n: np.full(F, np.nan) for n, _ in ANGLE_JOINTS}
    for f in range(F):
        P = xyz[:, :, f]
        rsho = resolve_point(P, ["RSHO"], idx)
        relb = resolve_point(P, ["RELB"], idx)
        clav = resolve_point(P, ["CLAV"], idx)
        wr   = resolve_point(P, ["RWRA", "RWRB"], idx)
        hf   = resolve_point(P, ["RFRA", "RFIN"], idx)
        if np.all(np.isfinite([rsho, clav, relb]).ravel()):
            out["shoulder"][f] = joint_angle(clav, rsho, relb)
        if np.all(np.isfinite([rsho, relb, wr]).ravel()):
            out["elbow"][f] = joint_angle(rsho, relb, wr)
        if np.all(np.isfinite([relb, wr, hf]).ravel()):
            out["wrist"][f] = joint_angle(relb, wr, hf)
    return out


def _fmt(v):
    return "  -- " if not np.isfinite(v) else "%5.1f" % v


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Visualize C3D markers + computed joint angles")
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    print("Loading:", args.file)
    data = load_c3d(args.file)
    xyz, labels, fps = data["xyz"], data["labels"], data["fps"]
    F = data["F"]
    labels_idx = {l: i for i, l in enumerate(labels)}

    s = max(0, args.start)
    e = args.end if args.end is not None else F - 1
    e = min(e, F - 1)

    print("  markers : %d  @ %.1f Hz" % (data["M"], fps))
    print("  frames  : %d  (showing %d..%d)" % (F, s, e))
    print("  NOTE    : file stores markers only (no segments/angles saved).")
    print("            Joint angles are COMPUTED from marker geometry.")

    angles = compute_angles(xyz, labels)

    # find a reference frame with most markers valid (for a stable 3-D box)
    # Use ONLY anatomical markers for the box: some cluster markers (*NN) can
    # be far-away calibration artefacts that would otherwise dwarf the body.
    anat_mask = np.array([not l.startswith("*") for l in labels])
    P0 = xyz[:, :, :].copy()                       # (3, M, F)
    P0[:, ~np.isfinite(P0[0]) | ~np.isfinite(P0[1]) | ~np.isfinite(P0[2])] = np.nan
    anat_valid = np.isfinite(P0).all(axis=0) & anat_mask[:, None]   # (M, F)
    valid_count = anat_valid.sum(axis=0)           # (F,) over anatomical only
    ref = int(np.argmax(valid_count[s:e + 1])) + s
    P0r = P0[:, anat_mask, ref].copy()             # (3, M_anat)
    P0r[:, ~np.isfinite(P0r).any(axis=0)] = np.nan
    center = np.nanmedian(P0r, axis=1)
    ext = float(np.nanmax(np.abs(P0r - center[:, None]))) * 1.35
    ext = max(ext, 1e-3)

    # --------------------------------------------------------------- figure
    fig = plt.figure(figsize=(14, 9))
    fig.canvas.manager.set_window_title(
        "C3D skeleton: " + args.file.rsplit("/", 1)[-1])
    gs = GridSpec(2, 1, height_ratios=[2.3, 1.0], hspace=0.22,
                  left=0.05, right=0.86, top=0.94, bottom=0.08)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    axt = fig.add_subplot(gs[1, 0])

    ax3d.set_xlim(center[0] - ext, center[0] + ext)
    ax3d.set_ylim(center[1] - ext, center[1] + ext)
    ax3d.set_zlim(center[2] - ext, center[2] + ext)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlabel("X (mm)"); ax3d.set_ylabel("Y (mm)"); ax3d.set_zlabel("Z (mm)")
    ax3d.set_title("3-D skeleton (drag to rotate)")

    bone_lines = []
    for name, a, b in BONES:
        (ln,) = ax3d.plot([], [], [], "-", color=BONE_COLORS[name], lw=4,
                          solid_capstyle="round", label=name)
        bone_lines.append((name, a, b, ln))

    anat_set = {l for l in labels if not l.startswith("*")}
    dot_anat = ax3d.scatter([], [], [], c="#111111", s=30, depthshade=False,
                            label="anatomical")
    dot_cluster = ax3d.scatter([], [], [], c="#bbbbbb", s=10, depthshade=False,
                               label="cluster")
    annots = {}
    for name in KEY_LABELS:
        if name in labels_idx:
            txt = ax3d.text(0, 0, 0, name, fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                                      alpha=0.6, ec="none"))
            annots[name] = txt
    ax3d.legend(loc="upper left", fontsize=7)

    # ------------------------------------------------------------ time plot
    tshow = np.arange(s, e + 1) / fps
    for name, col in ANGLE_JOINTS:
        axt.plot(tshow, angles[name][s:e + 1], label=name, color=col, lw=1.3)
    axt.set_xlim(s / fps, e / fps)
    axt.set_xlabel("time (s)")
    axt.set_ylabel("joint angle (deg)")
    axt.grid(alpha=0.3)
    axt.legend(loc="upper right", fontsize=8)
    cursor = axt.axvline(s / fps, color="k", lw=1.6, alpha=0.9)
    frame_text = axt.text(0.01, 0.95, "", transform=axt.transAxes, va="top",
                          fontsize=10,
                          bbox=dict(boxstyle="round", fc="lightyellow",
                                    alpha=0.85, ec="gray"))
    axt.set_title("Computed joint angles    "
                  "[space]=play/pause  [↑/↓]=step  [+/-]=speed  [esc]=quit")

    # -------------------------------------------------------- vertical slider
    axsl = fig.add_axes([0.895, 0.08, 0.05, 0.86])
    slider = Slider(axsl, "frame", float(s), float(e), valinit=float(s),
                    orientation="vertical", valstep=1)

    # ---------------------------------------------------------------- state
    state = dict(frame=s, playing=False, speed=1.0, last_t=None)

    def draw(_=None):
        f = int(round(state["frame"]))
        f = max(s, min(e, f))
        state["frame"] = f
        P = xyz[:, :, f]
        ok = np.isfinite(P).all(axis=0)

        for name, a, b, ln in bone_lines:
            pa = resolve_point(P, a, labels_idx)
            pb = resolve_point(P, b, labels_idx)
            if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                ln.set_data([pa[0], pb[0]], [pa[1], pb[1]])
                ln.set_3d_properties([pa[2], pb[2]])
                ln.set_visible(True)
            else:
                ln.set_visible(False)

        ia = [i for i in range(data["M"]) if labels[i] in anat_set and ok[i]]
        ic = [i for i in range(data["M"]) if labels[i] not in anat_set and ok[i]]
        dot_anat.set_offsets(np.column_stack((P[0, ia], P[1, ia])))
        dot_anat.set_3d_properties(P[2, ia], "z")
        dot_cluster.set_offsets(np.column_stack((P[0, ic], P[1, ic])))
        dot_cluster.set_3d_properties(P[2, ic], "z")
        for name, txt in annots.items():
            i = labels_idx[name]
            if ok[i]:
                txt.set_position((P[0, i], P[1, i], P[2, i]))
                txt.set_visible(True)
            else:
                txt.set_visible(False)

        cursor.set_xdata([f / fps, f / fps])
        frame_text.set_text(
            "frame %d / %d   t=%.2fs\n"
            "shoulder=%s  elbow=%s  wrist=%s" % (
                f, e, f / fps,
                _fmt(angles["shoulder"][f]), _fmt(angles["elbow"][f]),
                _fmt(angles["wrist"][f])))
        fig.canvas.draw_idle()

    def on_slider(val):
        if not state["playing"]:
            state["frame"] = int(val)
            draw()
    slider.on_changed(on_slider)

    import time as _time

    def step(_frame=None):
        if not state["playing"]:
            return
        now = _time.time()
        if state["last_t"] is None:
            state["last_t"] = now
        dt = now - state["last_t"]
        state["last_t"] = now
        f = int(state["frame"]) + max(1, int(round(dt * fps * state["speed"])))
        if f > e:
            f = s
        state["frame"] = f
        slider.set_val(f)
        draw()

    def toggle_play():
        state["playing"] = not state["playing"]
        state["last_t"] = None
        timer.start() if state["playing"] else timer.stop()

    def on_key(event):
        if event.key in (" ", "p"):
            toggle_play()
        elif event.key == "up":
            timer.stop(); state["playing"] = False
            state["frame"] = min(e, int(state["frame"]) + 1); draw()
        elif event.key == "down":
            timer.stop(); state["playing"] = False
            state["frame"] = max(s, int(state["frame"]) - 1); draw()
        elif event.key == "+":
            state["speed"] = min(10.0, state["speed"] * 1.5)
        elif event.key == "-":
            state["speed"] = max(0.1, state["speed"] / 1.5)
        elif event.key == "escape":
            plt.close(fig)
    fig.canvas.mpl_connect("key_press_event", on_key)

    timer = animation.FuncAnimation(fig, step, interval=40, blit=False,
                                    cache_frame_data=False)
    draw()
    print("GUI ready.")
    plt.show()


if __name__ == "__main__":
    main()
