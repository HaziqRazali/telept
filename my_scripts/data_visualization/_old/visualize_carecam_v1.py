"""
Three-panel carecam visualizer.

Produces a side-by-side video:
  [ raw input ] [ 2D skeleton ] [ 3D skeleton (matplotlib) ]

Usage
-----
    python visualize_carecam.py <video> <out.pkl> [output.mp4]

    python visualize_carecam.py \
        "/home/haziq/datasets/telept/data/Special Tests/videos/sd m neg 01.MP4" \
        "/home/haziq/datasets/telept/data/Special Tests/carecam/20260302_sd m neg 01/out.pkl" \
        --biomarker "right knee flexion" "right hip flexion"

    python visualize_carecam.py \
        "./sd m neg 01.MP4" \
        "./20260302_sd m neg 01/out.pkl" \
        --biomarker "right knee flexion" "right hip flexion"

Options
-------
    --height      Panel height in pixels (default: 720)
    --elev        3-D view elevation angle (default: 15)
    --azim        3-D view azimuth angle (default: -75)

Environment: conda activate mhr_new
"""

import argparse
import os
import pickle
import sys

import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Skeleton connections ──────────────────────────────────────────────────────
#
# Keypoint map (28 points):
#  0=R ear/eye   1=L ear/eye   2=nose       3=neck        4=chest
#  5=R shoulder  6=L shoulder  7=R elbow    8=L elbow
#  9=R wrist    10=L wrist    11=R hip     12=L hip      13=hip center
# 14=R knee(out)15=L knee(out)16=R knee(in)17=L knee(in)
# 18=R ankle    19=L ankle    20=R midfoot  21=L midfoot
# 22=R heel     23=L heel     24=R toe tip  25=L toe tip
# 26=R outer toe 27=L outer toe
#
SKELETON_CONNECTIONS = [
    # Head
    (0, 2), (1, 2),
    (2, 3),
    # Spine
    (3, 4), (4, 13),
    # Shoulders
    (3, 5), (3, 6), (5, 6),
    # Right arm
    (5, 7), (7, 9),
    # Left arm
    (6, 8), (8, 10),
    # Torso sides
    (5, 11), (6, 12),
    (11, 13), (12, 13),
    # Hips → knees
    (11, 14), (12, 15),
    (13, 16), (13, 17),
    (14, 16), (15, 17),
    # Knees → ankles
    (14, 18), (15, 19),
    (16, 20), (17, 21),
    (18, 20), (19, 21),
    # Right foot
    (18, 22), (22, 24), (22, 26), (24, 26),
    # Left foot
    (19, 23), (23, 25), (23, 27), (25, 27),
]

_LEFT_IDX  = {1, 6, 8, 10, 12, 15, 17, 19, 21, 23, 25, 27}
_RIGHT_IDX = {0, 5, 7,  9, 11, 14, 16, 18, 20, 22, 24, 26}

# 2-D overlay colours (BGR)
_BONE_LEFT   = (74, 144, 219)   # orange-ish
_BONE_RIGHT  = (74, 219, 144)   # green-ish
_BONE_CENTRE = (200, 200, 200)  # grey

# 3-D matplotlib colours (hex)
_3D_LEFT   = "#4a90d9"
_3D_RIGHT  = "#5cb85c"
_3D_CENTRE = "#aaaaaa"


def _bone_color_bgr(a, b):
    if a in _LEFT_IDX  and b in _LEFT_IDX:  return _BONE_LEFT
    if a in _RIGHT_IDX and b in _RIGHT_IDX: return _BONE_RIGHT
    return _BONE_CENTRE


def _bone_color_hex(a, b):
    if a in _LEFT_IDX  and b in _LEFT_IDX:  return _3D_LEFT
    if a in _RIGHT_IDX and b in _RIGHT_IDX: return _3D_RIGHT
    return _3D_CENTRE


# ── Colour palette for dots ───────────────────────────────────────────────────
_PALETTE = [
    (255,  56,  56), (255, 157,  56), (255, 225,  56), (129, 255,  56),
    ( 56, 255, 133), ( 56, 255, 255), ( 56, 133, 255), (129,  56, 255),
    (255,  56, 225), (255, 128, 128), (128, 255, 128), (128, 128, 255),
    (255, 200,   0), (  0, 200, 255), (200,   0, 255), (255,   0, 128),
    (  0, 255, 128), (128,   0, 255), (200, 255,   0), (  0, 128, 255),
    (255,  80,   0), (  0, 255,  80), ( 80,   0, 255), (255,   0,  80),
    (  0,  80, 255), ( 80, 255,   0), (200, 100, 100), (100, 200, 100),
]


# ── 2-D panel ─────────────────────────────────────────────────────────────────

def draw_2d_panel(frame, kpts2d, dot_radius=4):
    """Draw skeleton + coloured dots (no labels) onto frame."""
    out = frame.copy()
    pts = np.asarray(kpts2d, dtype=float)

    # Bones first
    for a, b in SKELETON_CONNECTIONS:
        if a >= len(pts) or b >= len(pts):
            continue
        xa, ya = int(round(pts[a, 0])), int(round(pts[a, 1]))
        xb, yb = int(round(pts[b, 0])), int(round(pts[b, 1]))
        cv2.line(out, (xa, ya), (xb, yb), _bone_color_bgr(a, b), 2, cv2.LINE_AA)

    # Dots on top
    for idx, pt in enumerate(pts):
        x, y = int(round(pt[0])), int(round(pt[1]))
        color = _PALETTE[idx % len(_PALETTE)]
        cv2.circle(out, (x, y), dot_radius, color, -1, cv2.LINE_AA)
        cv2.circle(out, (x, y), dot_radius, (255, 255, 255), 1, cv2.LINE_AA)

    return out


# ── 3-D panel ─────────────────────────────────────────────────────────────────

def compute_global_bounds(pose3d_series):
    """Stable axis limits from all frames."""
    all_x, all_y, all_z = [], [], []
    for pts in pose3d_series:
        arr = np.asarray(pts, dtype=float)
        all_x.extend(arr[:, 0].tolist())
        all_y.extend(arr[:, 2].tolist())        # depth → plot-y
        all_z.extend((-arr[:, 1]).tolist())     # up    → plot-z
    mins = np.array([min(all_x), min(all_y), min(all_z)])
    maxs = np.array([max(all_x), max(all_y), max(all_z)])
    mid  = (mins + maxs) / 2.0
    half = (maxs - mins).max() / 2.0 * 1.15
    return mid, half


def draw_3d_panel(ax, kpts3d, mid, half, frame_idx, n_frames):
    ax.cla()
    pts = np.asarray(kpts3d, dtype=float)
    xs = pts[:, 0]
    ys = pts[:, 2]          # depth
    zs = -pts[:, 1]         # up

    for a, b in SKELETON_CONNECTIONS:
        if a >= len(pts) or b >= len(pts):
            continue
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]],
                color=_bone_color_hex(a, b), linewidth=1.8, alpha=0.9)

    ax.scatter(xs, ys, zs, c="white", s=10, edgecolors="black",
               linewidths=0.4, zorder=5, depthshade=False)

    ax.set_xlim3d([mid[0] - half, mid[0] + half])
    ax.set_ylim3d([mid[1] - half, mid[1] + half])
    ax.set_zlim3d([mid[2] - half, mid[2] + half])

    ax.set_xlabel("X",     fontsize=7, labelpad=1)
    ax.set_ylabel("Depth", fontsize=7, labelpad=1)
    ax.set_zlabel("Up",    fontsize=7, labelpad=1)
    ax.set_title(f"3D  {frame_idx + 1}/{n_frames}", fontsize=8, pad=3)
    ax.tick_params(labelsize=5)
    ax.set_facecolor("#1a1a2e")
    ax.figure.patch.set_facecolor("#1a1a2e")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)


def fig_to_bgr(fig, w, h):
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    fw, fh = fig.canvas.get_width_height()
    img = buf.reshape(fh, fw, 4)[:, :, :3]
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def render_angle_graph_base(angles, total_w, graph_h, label="Angle"):
    """Pre-render the named angle curve (no slider) as BGR."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(total_w / dpi, graph_h / dpi), dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    xs = np.arange(len(angles))
    ax.plot(xs, angles, color="#4a90d9", linewidth=1.5)
    ax.fill_between(xs, angles, alpha=0.15, color="#4a90d9")
    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_xlim(0, len(angles) - 1)
    ax.set_ylabel(f"{label} (°)", fontsize=8, color="white")
    ax.set_xlabel("Frame", fontsize=8, color="white")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4, color="#888888")
    fig.tight_layout(pad=0.5)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    fw, fh = fig.canvas.get_width_height()
    img = buf.reshape(fh, fw, 4)[:, :, :3]
    img = cv2.resize(img, (total_w, graph_h), interpolation=cv2.INTER_AREA)
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("video",  help="Input video path")
    parser.add_argument("pkl",    help="carecam out.pkl file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output path (default: <video_stem>_vis3panel.mp4)")
    parser.add_argument("--height",       type=int,   default=720,
                        help="Panel height in pixels (default: 720)")
    parser.add_argument("--graph-height",  type=int,   default=160,
                        help="Angle-graph strip height per biomarker in pixels (default: 160)")
    parser.add_argument("--biomarker",     type=str,   nargs="+",
                        default=[],
                        help="One or more biomarker names to plot (case-insensitive partial "
                             "match). E.g. --biomarker 'left knee' 'right hip'")
    parser.add_argument("--elev",          type=float, default=15.0)
    parser.add_argument("--azim",          type=float, default=-75.0)
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    print(f"Loading: {args.pkl}")
    with open(args.pkl, "rb") as f:
        df = pickle.load(f)

    for col in ("person1_pose2D", "person1_pose3D"):
        if col not in df.columns:
            sys.exit(f"ERROR: '{col}' not in pkl. Available: {list(df.columns)}")

    print(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open '{args.video}'")

    fps   = cap.get(cv2.CAP_PROP_FPS)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_vid, len(df))

    # ── Load bioMarker.json ────────────────────────────────────────────────────
    bio_path = os.path.join(os.path.dirname(os.path.abspath(args.pkl)), "bioMarker.json")
    # List of (label, angles) for each requested biomarker
    bio_tracks = []   # [(label_str, [float, ...]), ...]
    if os.path.isfile(bio_path):
        print(f"Loading bioMarker.json: {bio_path}")
        with open(bio_path) as f:
            bio_data = json.load(f)
        available = [item["name"] for item in bio_data]
        print(f"  Available biomarkers: {available}")
        for query in args.biomarker:
            ql = query.lower()
            match = next((it for it in bio_data if ql in it["name"].lower()), None)
            if match:
                label  = match["name"]
                angles = match["measurementList"]["angleList"]
                bio_tracks.append((label, angles))
                print(f"  Selected: '{label}' → {len(angles)} frames")
                if len(angles) != n_frames:
                    print(f"    WARNING: frame count mismatch "
                          f"(bio={len(angles)}, vis={n_frames})")
            else:
                print(f"  WARNING: no biomarker matching '{query}'. "
                      f"Available: {available}")
    else:
        print(f"WARNING: bioMarker.json not found at {bio_path}. Skipping angle graphs.")

    # ── Layout ─────────────────────────────────────────────────────────────────
    out_h   = args.height
    panel_w = int(round(vid_w * out_h / vid_h))   # video aspect-ratio width
    skel_w  = out_h                                # square 3-D panel
    total_w = panel_w * 2 + skel_w                # left + middle + right
    graph_h = args.graph_height if bio_tracks else 0
    total_h = out_h + graph_h * len(bio_tracks)

    if args.output is None:
        pkl_dir = os.path.dirname(os.path.abspath(args.pkl))
        args.output = pkl_dir + ".mp4"
    print(f"Writing: {args.output}  ({total_w}x{total_h})")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (total_w, total_h))
    if not writer.isOpened():
        sys.exit("ERROR: could not open VideoWriter.")

    # Scale factors for 2-D keypoints (stored in original pixel space)
    sx = panel_w / vid_w
    sy = out_h   / vid_h

    # ── Pre-compute 3-D axis limits ────────────────────────────────────────────
    print("Computing 3-D axis bounds ...")
    mid, half = compute_global_bounds(df["person1_pose3D"])

    # ── Pre-render angle graph base images (one per biomarker) ────────────────
    graph_bases = []   # list of BGR numpy arrays
    if bio_tracks and graph_h > 0:
        _graph_x_left  = int(total_w * 0.065)
        _graph_x_right = int(total_w * 0.985)
        for label, angles in bio_tracks:
            print(f"Pre-rendering graph: {label} ...")
            base = render_angle_graph_base(angles, total_w, graph_h, label=label)
            graph_bases.append((base, len(angles)))

    def _slider_x(frame_idx, n_angles):
        """Map frame index to pixel x on a graph strip."""
        t = frame_idx / max(n_angles - 1, 1)
        return int(_graph_x_left + t * (_graph_x_right - _graph_x_left))

    # ── Matplotlib figure (reused) ─────────────────────────────────────────────
    dpi = 100
    fig = plt.figure(figsize=(skel_w / dpi, out_h / dpi), dpi=dpi)
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)

    # ── Frame loop ─────────────────────────────────────────────────────────────
    print(f"Rendering {n_frames} frames ...")
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: video ended at frame {i}")
            break

        # ── Panel 1: raw ──────────────────────────────────────────────────────
        raw_panel = cv2.resize(frame, (panel_w, out_h), interpolation=cv2.INTER_AREA)

        # ── Panel 2: 2-D skeleton ─────────────────────────────────────────────
        kpts2d_raw = df["person1_pose2D"].iloc[i]
        if kpts2d_raw is not None and len(kpts2d_raw) > 0:
            kpts2d = np.asarray(kpts2d_raw, dtype=float).copy()
            kpts2d[:, 0] *= sx
            kpts2d[:, 1] *= sy
            skel2d_panel = draw_2d_panel(raw_panel.copy(), kpts2d, dot_radius=4)
        else:
            skel2d_panel = raw_panel.copy()

        # ── Panel 3: 3-D matplotlib ───────────────────────────────────────────
        kpts3d_raw = df["person1_pose3D"].iloc[i]
        draw_3d_panel(ax, kpts3d_raw, mid, half, i, n_frames)
        skel3d_panel = fig_to_bgr(fig, skel_w, out_h)

        # ── Concatenate ───────────────────────────────────────────────────────
        top_row = np.hstack([raw_panel, skel2d_panel, skel3d_panel])

        rows = [top_row]
        for base, n_angles in graph_bases:
            strip = base.copy()
            sx_pos = _slider_x(i, n_angles)
            cv2.line(strip, (sx_pos, 0), (sx_pos, graph_h - 1),
                     (255, 230, 80), 2, cv2.LINE_AA)
            rows.append(strip)
        combined = np.vstack(rows)

        writer.write(combined)

        if (i + 1) % 30 == 0 or i == n_frames - 1:
            print(f"  {i + 1}/{n_frames} frames done")

    cap.release()
    writer.release()
    plt.close(fig)
    print(f"\nDone! Saved -> {args.output}")


if __name__ == "__main__":
    main()
