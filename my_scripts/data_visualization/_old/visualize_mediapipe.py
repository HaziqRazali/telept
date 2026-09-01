"""
Three-panel MediaPipe visualizer.

Produces a side-by-side video:
  [ raw input ] [ 2D skeleton ] [ 3D skeleton (matplotlib) ]
  [ angle graph strip (one per --biomarker) ]

Usage:
    python visualize_mediapipe.py <video_path> <json_path> [output_path]
    python visualize_mediapipe.py \
        "/home/haziq/datasets/telept/data/Special Tests/videos/sd m neg 01.MP4" \
        "/home/haziq/datasets/telept/data/Special Tests/mediapipe/sd m neg 01.json" \
        --biomarker "left knee flexion" "right hip flexion"

Supported biomarkers (case-insensitive partial match):
    left knee flexion   right knee flexion
    left hip flexion    right hip flexion

Options:
    --height        Panel height in pixels (default: 720)
    --graph-height  Angle-graph strip height per biomarker (default: 160)
    --elev          3-D view elevation (default: 15)
    --azim          3-D view azimuth   (default: -75)

If output_path is omitted, the result is saved next to the json with the same stem.

Environment: conda activate mhr_new
"""

import argparse
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ── MediaPipe landmark indices ─────────────────────────────────────────────────
# fmt: off
_MP = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow":    13, "r_elbow":    14,
    "l_wrist":    15, "r_wrist":    16,
    "l_hip":      23, "r_hip":      24,
    "l_knee":     25, "r_knee":     26,
    "l_ankle":    27, "r_ankle":    28,
    "l_heel":     29, "r_heel":     30,
    "l_foot":     31, "r_foot":     32,
}
# fmt: on

# ── Inlined math helpers (mirrors utils_math.py from sam-3d-body) ─────────────

def _normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return np.zeros_like(v) if n < eps else v / n


def _project_vec_to_plane(v, n):
    """Remove the component of v along plane-normal n."""
    v = np.asarray(v, dtype=np.float64)
    n = _normalize(n)
    return v - np.dot(v, n) * n


def _build_plane_basis(up, right):
    """Return orthonormal (right, forward, up) from approximate up + right."""
    up_u      = _normalize(up)
    right_u   = _normalize(right)
    forward_u = _normalize(np.cross(up_u, right_u))
    right_u   = _normalize(np.cross(forward_u, up_u))
    return right_u, forward_u, up_u


def _signed_angle_in_plane(a, b, plane_normal, eps=1e-8):
    """Signed angle from reference b → vector a in the given plane (radians)."""
    n  = _normalize(plane_normal)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < eps or nb < eps:
        return np.nan
    a_u = a / na;  b_u = b / nb
    ang = float(np.arccos(float(np.clip(np.dot(b_u, a_u), -1.0, 1.0))))
    s   = float(np.dot(n, np.cross(b_u, a_u)))
    return ang if abs(s) < 1e-12 else np.sign(s) * ang


# ── Per-biomarker compute functions ───────────────────────────────────────────

def _lm(frame, idx):
    """Extract landmark idx as float64 xyz array."""
    return np.array([frame[idx]["x"], frame[idx]["y"], frame[idx]["z"]], dtype=np.float64)


def _knee_flexion(frame, hip_idx, knee_idx, ankle_idx):
    """
    Hinge angle at knee: 0° = fully straight, positive = flexion.
    Mirrors render_smplx_video.py hinge(..., zero_when_straight=True).
    """
    pp = _lm(frame, hip_idx)
    pj = _lm(frame, knee_idx)
    pd = _lm(frame, ankle_idx)
    vr = pp - pj;  vm = pd - pj
    nr = np.linalg.norm(vr);  nm = np.linalg.norm(vm)
    if nr < 1e-8 or nm < 1e-8:
        return np.nan
    raw = float(np.degrees(np.arccos(np.clip(np.dot(vr / nr, vm / nm), -1., 1.))))
    return 180.0 - raw   # 0 when straight, ~90 at right angle, etc.


def _hip_flexion_sagittal(frame, hip_idx, knee_idx, side):
    """
    Sagittal-plane signed hip flexion angle, matching render_smplx_video.py's
    sagittal_signed() exactly.

    Pelvis frame approximation with MediaPipe landmarks:
      pelvis  → midpoint of l_hip (23) + r_hip (24)
      spine3  → midpoint of l_shoulder (11) + r_shoulder (12)
      lateral → r_hip - l_hip  (always left→right regardless of side)
    """
    l_hip  = _lm(frame, _MP["l_hip"])
    r_hip  = _lm(frame, _MP["r_hip"])
    l_sh   = _lm(frame, _MP["l_shoulder"])
    r_sh   = _lm(frame, _MP["r_shoulder"])
    mid_hip = (l_hip + r_hip) / 2.0       # ≈ pelvis
    mid_sh  = (l_sh  + r_sh)  / 2.0       # ≈ spine3

    up  = _normalize(mid_sh  - mid_hip)   # trunk-up axis
    rg  = _normalize(r_hip   - l_hip)     # lateral (left→right)

    right, _forward, _up = _build_plane_basis(up, rg)
    plane_normal = -right                  # sagittal plane normal (points left)

    p_hip  = _lm(frame, hip_idx)
    p_knee = _lm(frame, knee_idx)

    v_main = p_knee  - p_hip              # thigh vector (down when standing)
    v_ref  = mid_hip - mid_sh             # trunk-down (pelvis - spine3)

    vm_p = _project_vec_to_plane(v_main, plane_normal)
    vr_p = _project_vec_to_plane(v_ref,  plane_normal)

    nm = np.linalg.norm(vm_p)
    nr = np.linalg.norm(vr_p)
    if nm < 1e-8 or nr < 1e-8:
        return np.nan

    vr_p = _normalize(vr_p) * nm          # equalize magnitudes (keep direction)
    ang_rad = _signed_angle_in_plane(vm_p, vr_p, plane_normal)
    return float(np.degrees(ang_rad)) if np.isfinite(ang_rad) else np.nan


# ── Biomarker registry ────────────────────────────────────────────────────────
# Each entry: (display_name, compute_fn)
# compute_fn(frame) -> float degrees, where frame is a list of landmark dicts.

def _require(frame, *idxs):
    return frame and len(frame) > max(idxs)

BIOMARKER_REGISTRY = [
    ("Left Knee Flexion",
     lambda f: _knee_flexion(f, _MP["l_hip"], _MP["l_knee"], _MP["l_ankle"])
               if _require(f, _MP["l_hip"], _MP["l_knee"], _MP["l_ankle"]) else float("nan")),
    ("Right Knee Flexion",
     lambda f: _knee_flexion(f, _MP["r_hip"], _MP["r_knee"], _MP["r_ankle"])
               if _require(f, _MP["r_hip"], _MP["r_knee"], _MP["r_ankle"]) else float("nan")),
    ("Left Hip Flexion",
     lambda f: _hip_flexion_sagittal(f, _MP["l_hip"], _MP["l_knee"], "left")
               if _require(f, _MP["l_hip"], _MP["l_knee"], _MP["l_shoulder"],
                              _MP["r_hip"], _MP["r_shoulder"]) else float("nan")),
    ("Right Hip Flexion",
     lambda f: _hip_flexion_sagittal(f, _MP["r_hip"], _MP["r_knee"], "right")
               if _require(f, _MP["r_hip"], _MP["r_knee"], _MP["l_shoulder"],
                              _MP["l_hip"], _MP["r_shoulder"]) else float("nan")),
]


def compute_biomarker_series(pose_data, compute_fn):
    """
    compute_fn(frame) -> float degrees.
    Returns a list of one angle per frame.
    """
    return [compute_fn(frame) for frame in pose_data]


def render_angle_graph_base(angles, total_w, graph_h, label="Angle"):
    """Pre-render the named angle curve (no slider) as a BGR numpy array."""
    dpi = 100
    fig, ax = plt.subplots(figsize=(total_w / dpi, graph_h / dpi), dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    xs = np.arange(len(angles))
    valid = ~np.isnan(np.asarray(angles, dtype=float))
    ax.plot(xs[valid], np.asarray(angles, dtype=float)[valid], color="#4a90d9", linewidth=1.5)
    ax.fill_between(xs[valid], np.asarray(angles, dtype=float)[valid], alpha=0.15, color="#4a90d9")
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


# ── MediaPipe Pose landmark connections ──────────────────────────────────────
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Left arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Colour scheme: left side blue, right side green, centre/face grey
_LEFT  = [1, 2, 3, 7, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
_RIGHT = [4, 5, 6, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]

# 2-D overlay colours (BGR)
_BONE_LEFT   = (74, 144, 219)
_BONE_RIGHT  = (74, 219, 144)
_BONE_CENTRE = (200, 200, 200)

# Colour palette for dots
_PALETTE = [
    (255,  56,  56), (255, 157,  56), (255, 225,  56), (129, 255,  56),
    ( 56, 255, 133), ( 56, 255, 255), ( 56, 133, 255), (129,  56, 255),
    (255,  56, 225), (255, 128, 128), (128, 255, 128), (128, 128, 255),
    (255, 200,   0), (  0, 200, 255), (200,   0, 255), (255,   0, 128),
    (  0, 255, 128), (128,   0, 255), (200, 255,   0), (  0, 128, 255),
    (255,  80,   0), (  0, 255,  80), ( 80,   0, 255), (255,   0,  80),
    (  0,  80, 255), ( 80, 255,   0), (200, 100, 100), (100, 200, 100),
    (100, 100, 200), (200, 200,   0), (  0, 200, 200), (200,   0, 200),
    (255, 128,   0),
]


def _conn_color(a, b):
    if a in _LEFT  and b in _LEFT:  return "#4a90d9"   # blue  – left
    if a in _RIGHT and b in _RIGHT: return "#5cb85c"   # green – right
    return "#aaaaaa"                                    # grey  – centre


def _bone_color_bgr(a, b):
    if a in _LEFT  and b in _LEFT:  return _BONE_LEFT
    if a in _RIGHT and b in _RIGHT: return _BONE_RIGHT
    return _BONE_CENTRE


# ── 2-D panel ─────────────────────────────────────────────────────────────────

def draw_2d_panel(landmarks, mid2d, half2d, out_w, out_h, dot_radius=4):
    """
    Draw an orthographic front-view (X/Up) projection of world landmarks.

    Uses per-axis 2-D bounds so the skeleton fills the panel.
    """
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:] = (26, 26, 46)   # same dark background as 3-D panel

    def world_to_px(lm):
        wx =  lm["x"]
        wy = -lm["y"]   # flip: world -y = up
        px = int(round((wx - (mid2d[0] - half2d)) / (2 * half2d) * out_w))
        py = int(round((1.0 - (wy - (mid2d[1] - half2d)) / (2 * half2d)) * out_h))
        return px, py

    pts = [world_to_px(lm) for lm in landmarks]

    for a, b in POSE_CONNECTIONS:
        if a >= len(pts) or b >= len(pts):
            continue
        cv2.line(canvas, pts[a], pts[b], _bone_color_bgr(a, b), 2, cv2.LINE_AA)

    for idx, (x, y) in enumerate(pts):
        color = _PALETTE[idx % len(_PALETTE)]
        cv2.circle(canvas, (x, y), dot_radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), dot_radius, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_global_bounds(pose_data):
    """
    Return (mid, half_range) for the 3-D plot and (mid2d, half2d) for the
    2-D front-view panel.

    3-D plot mapping:
        plot-x = lm['x']
        plot-y = lm['z']   (depth)
        plot-z = -lm['y']  (up)

    2-D front-view:
        panel-x = lm['x']
        panel-y = -lm['y']  (up, flipped so head is at top)
    """
    all_px, all_py, all_pz = [], [], []
    all_2dx, all_2dy = [], []
    for frame in pose_data:
        for lm in frame:
            all_px.append(lm["x"])
            all_py.append(lm["z"])
            all_pz.append(-lm["y"])
            all_2dx.append(lm["x"])
            all_2dy.append(-lm["y"])

    # 3-D bounds (equal aspect, max span across all axes)
    mins = np.array([min(all_px), min(all_py), min(all_pz)])
    maxs = np.array([max(all_px), max(all_py), max(all_pz)])
    mid  = (mins + maxs) / 2.0
    half_range = (maxs - mins).max() / 2.0 * 1.15

    # 2-D bounds (per-axis so skeleton fills the panel)
    x_mid  = (min(all_2dx) + max(all_2dx)) / 2.0
    y_mid  = (min(all_2dy) + max(all_2dy)) / 2.0
    x_half = (max(all_2dx) - min(all_2dx)) / 2.0 * 1.2
    y_half = (max(all_2dy) - min(all_2dy)) / 2.0 * 1.2
    # keep square panels by using the larger of the two halves
    half2d = max(x_half, y_half)
    mid2d  = np.array([x_mid, y_mid])

    return mid, half_range, mid2d, half2d


def draw_skeleton(ax, landmarks, mid, half_range, frame_idx, n_frames):
    """Clear ax and draw one frame's 3D skeleton."""
    ax.cla()

    xs = [lm["x"] for lm in landmarks]
    ys = [lm["z"] for lm in landmarks]   # depth → plot-y
    zs = [-lm["y"] for lm in landmarks]  # up    → plot-z

    # Connections
    for a, b in POSE_CONNECTIONS:
        color = _conn_color(a, b)
        ax.plot([xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]],
                color=color, linewidth=1.8, alpha=0.9)

    # Joints
    ax.scatter(xs, ys, zs, c="white", s=12, edgecolors="black",
               linewidths=0.5, zorder=5, depthshade=False)

    # Equal axes
    ax.set_xlim3d([mid[0] - half_range, mid[0] + half_range])
    ax.set_ylim3d([mid[1] - half_range, mid[1] + half_range])
    ax.set_zlim3d([mid[2] - half_range, mid[2] + half_range])

    ax.set_xlabel("X",      fontsize=8, labelpad=2)
    ax.set_ylabel("Depth",  fontsize=8, labelpad=2)
    ax.set_zlabel("Up",     fontsize=8, labelpad=2)
    ax.set_title(f"3D Skeleton  {frame_idx + 1}/{n_frames}",
                 fontsize=9, pad=4)
    ax.tick_params(labelsize=6)
    ax.set_facecolor("#1a1a2e")
    ax.figure.patch.set_facecolor("#1a1a2e")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)


def fig_to_bgr(fig, w, h):
    """Render matplotlib figure to a BGR numpy array of size (h, w, 3)."""
    fig.canvas.draw()
    # buffer_rgba works across all recent matplotlib versions
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    fw, fh = fig.canvas.get_width_height()
    img = buf.reshape(fh, fw, 4)           # RGBA
    img = img[:, :, :3]                    # drop alpha → RGB
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video",  help="Path to input video (.MP4 / .mp4 …)")
    parser.add_argument("json",   help="Path to motion-data JSON file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output video path (default: <video_stem>_skeleton.mp4)")
    parser.add_argument("--height", type=int, default=720,
                        help="Output frame height in pixels (default: 720)")
    parser.add_argument("--elev",  type=float, default=15.0,
                        help="3-D view elevation angle (default: 15)")
    parser.add_argument("--azim",  type=float, default=-75.0,
                        help="3-D view azimuth angle (default: -75)")
    parser.add_argument("--biomarker", type=str, nargs="+", default=[],
                        help="One or more biomarker names to plot (case-insensitive partial "
                             "match). E.g. --biomarker 'left knee' 'right hip'")
    parser.add_argument("--graph-height", type=int, default=160,
                        help="Angle-graph strip height per biomarker in pixels (default: 160)")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading pose data from: {args.json}")
    with open(args.json) as f:
        pose_data = json.load(f)

    print(f"Opening video: {args.video}")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video '{args.video}'")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    vid_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_frames, len(pose_data))

    # ── Resolve requested biomarkers ──────────────────────────────────────────
    bio_tracks = []   # [(label, [float, ...]), ...]
    for query in args.biomarker:
        ql = query.lower()
        match = next(
            (entry for entry in BIOMARKER_REGISTRY if ql in entry[0].lower()), None
        )
        if match:
            label, compute_fn = match
            angles = compute_biomarker_series(pose_data, compute_fn)
            bio_tracks.append((label, angles))
            print(f"  Biomarker '{label}': {len(angles)} frames computed")
        else:
            available = [e[0] for e in BIOMARKER_REGISTRY]
            print(f"  WARNING: no biomarker matching '{query}'. Available: {available}")

    # ── Layout ────────────────────────────────────────────────────────────────
    out_h     = args.height
    vid_out_w = int(round(vid_w * out_h / vid_h))   # keep AR of original
    skel_w    = out_h                                # square 3-D plot panel
    total_w   = vid_out_w * 2 + skel_w              # left + middle + right
    graph_h   = args.graph_height if bio_tracks else 0
    total_h   = out_h + graph_h * len(bio_tracks)

    # ── Output path ───────────────────────────────────────────────────────────
    if args.output is None:
        json_stem = os.path.splitext(args.json)[0]
        args.output = json_stem + ".mp4"

    print(f"Writing: {args.output}  ({total_w}x{total_h})")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (total_w, total_h))
    if not writer.isOpened():
        sys.exit("ERROR: could not open VideoWriter – check codec support.")

    # ── Pre-compute axis bounds from all frames (stable axes) ─────────────────
    print("Computing global skeleton bounds …")
    mid, half_range, mid2d, half2d = compute_global_bounds(pose_data)

    # ── Pre-render angle graph base images (one per biomarker) ──────────────
    graph_bases = []
    if bio_tracks and graph_h > 0:
        _graph_x_left  = int(total_w * 0.065)
        _graph_x_right = int(total_w * 0.985)
        for label, angles in bio_tracks:
            print(f"Pre-rendering graph: {label} ...")
            base = render_angle_graph_base(angles, total_w, graph_h, label=label)
            graph_bases.append((base, len(angles)))

    def _slider_x(frame_idx, n_angles):
        t = frame_idx / max(n_angles - 1, 1)
        return int(_graph_x_left + t * (_graph_x_right - _graph_x_left))

    # ── Matplotlib figure (re-used across frames) ─────────────────────────────
    dpi = 100
    fig = plt.figure(figsize=(skel_w / dpi, out_h / dpi), dpi=dpi)
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=args.elev, azim=args.azim)

    # ── Frame loop ────────────────────────────────────────────────────────────
    print(f"Rendering {n_frames} frames …")
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: video ended at frame {i}")
            break

        # ── Panel 1: raw ──────────────────────────────────────────────────────
        raw_panel = cv2.resize(frame, (vid_out_w, out_h),
                               interpolation=cv2.INTER_AREA)

        # ── Panel 2: 2-D skeleton ─────────────────────────────────────────────
        landmarks = pose_data[i]
        if landmarks:
            skel2d_panel = draw_2d_panel(landmarks, mid2d, half2d, vid_out_w, out_h)
        else:
            skel2d_panel = np.zeros((out_h, vid_out_w, 3), dtype=np.uint8)

        # ── Panel 3: 3-D matplotlib ───────────────────────────────────────────
        draw_skeleton(ax, landmarks, mid, half_range, i, n_frames)
        skel_img = fig_to_bgr(fig, skel_w, out_h)

        # ── Concatenate ───────────────────────────────────────────────────────
        top_row = np.hstack([raw_panel, skel2d_panel, skel_img])

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
    print(f"\nDone! Saved → {args.output}")


if __name__ == "__main__":
    main()

