#!/usr/bin/env python3
"""
Render SMPL-X mesh (from JSON params) onto the original video.

Output: 3-panel video
    [ raw input ] [ SMPL-X mesh overlay – front ] [ SMPL-X mesh on white – side ]

Usage
-----
    python render_smplx_video.py <video> <smplx.json> <mhr_outputs.npz> <out.mp4>

Environment: conda activate sam_3d_body
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation

# ── SAM-3D-Body renderer ────────────────────────────────────────────────────
SAM3D_DIR = Path("/home/haziq/sam-3d-body")
sys.path.insert(0, str(SAM3D_DIR))
sys.path.insert(0, str(SAM3D_DIR / "my_scripts"))
from sam_3d_body.visualization.renderer import Renderer   # noqa: E402
from utils_math import (                                   # noqa: E402
    normalize,
    project_vec_to_plane,
    signed_angle_in_plane,
    build_plane_basis_from_up_and_right,
)
from utils_rom_config import JOINT_NAMES                   # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)

# ── SMPL-X model paths ───────────────────────────────────────────────────────
def find_smplx_path():
    candidates = [
        os.path.expanduser("/media/haziq/Haziq/mocap/data/models_smplx_v1_1/models/smplx"),
        os.path.expanduser("~/datasets/mocap/data/models_smplx_v1_1/models/smplx"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return None


# ── Rotation-matrix → axis-angle ────────────────────────────────────────────

def rotmat_to_aa(rotmats: np.ndarray) -> np.ndarray:
    """
    [..., 3, 3] rotation matrices → [..., 3] axis-angle vectors.
    Uses scipy for robustness.
    """
    orig = rotmats.shape[:-2]
    flat = rotmats.reshape(-1, 3, 3)
    aa = Rotation.from_matrix(flat).as_rotvec().astype(np.float32)
    return aa.reshape(*orig, 3)


# ── Biomarker angle computation ───────────────────────────────────────────────

def _jnt(joints, name):
    return joints[JOINT_NAMES.index(name)]


def compute_angle(task_name: str, joints: np.ndarray) -> float:
    """Return scalar angle (degrees) for the given task at a single frame."""

    def hinge(joint, proximal, distal, zero_when_straight=True):
        pj = _jnt(joints, joint)
        pp = _jnt(joints, proximal)
        pd = _jnt(joints, distal)
        vr, vm = pp - pj, pd - pj
        nr, nm = np.linalg.norm(vr), np.linalg.norm(vm)
        if nr < 1e-8 or nm < 1e-8:
            return np.nan
        ang = float(np.degrees(np.arccos(np.clip(np.dot(vr/nr, vm/nm), -1., 1.))))
        return 180. - ang if zero_when_straight else ang

    def sagittal_signed(hip_jnt, knee_jnt, up_from, up_to, lr_from, lr_to):
        """Signed angle in body sagittal plane (flexion positive)."""
        p_up_from = _jnt(joints, up_from)
        p_up_to   = _jnt(joints, up_to)
        p_lr_from = _jnt(joints, lr_from)
        p_lr_to   = _jnt(joints, lr_to)
        p_hip  = _jnt(joints, hip_jnt)
        p_knee = _jnt(joints, knee_jnt)
        up      = normalize(p_up_to   - p_up_from)
        rg      = normalize(p_lr_to   - p_lr_from)
        right, forward, up = build_plane_basis_from_up_and_right(up, rg)
        plane_normal = -right
        v_main = p_knee - p_hip
        v_ref  = p_up_from - p_up_to   # DOWN
        vm_p = project_vec_to_plane(v_main, plane_normal)
        vr_p = project_vec_to_plane(v_ref,  plane_normal)
        nm, nr = np.linalg.norm(vm_p), np.linalg.norm(vr_p)
        if nm < 1e-8 or nr < 1e-8:
            return np.nan
        vr_p = normalize(vr_p) * nm
        ang_rad = signed_angle_in_plane(vm_p, vr_p, plane_normal)
        return float(np.degrees(ang_rad)) if np.isfinite(ang_rad) else np.nan

    t = task_name.lower().replace(" ", "_")

    if t == "left_knee_flexion":
        return hinge("left_knee",  "left_hip",      "left_ankle",  zero_when_straight=True)
    elif t == "right_knee_flexion":
        return hinge("right_knee", "right_hip",     "right_ankle", zero_when_straight=True)
    elif t == "left_elbow_flexion":
        return hinge("left_elbow", "left_shoulder", "left_wrist",  zero_when_straight=False)
    elif t == "right_elbow_flexion":
        return hinge("right_elbow","right_shoulder","right_wrist", zero_when_straight=False)
    elif t == "left_hip_flexion":
        return sagittal_signed("left_hip",  "left_knee",
                               "pelvis", "spine3", "left_hip", "right_hip")
    elif t == "right_hip_flexion":
        return sagittal_signed("right_hip", "right_knee",
                               "pelvis", "spine3", "left_hip", "right_hip")
    else:
        raise ValueError(f"Unsupported biomarker task: '{task_name}'. "
                         f"Supported: left/right knee/elbow/hip flexion.")


# ── Angle graph strip ─────────────────────────────────────────────────────────

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
    ax.set_xlim(0, max(len(angles) - 1, 1))
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video",  help="Original input video path")
    parser.add_argument("json",   help="SMPL-X JSON parameter file")
    parser.add_argument("npz",    help="SAM-3D-Body NPZ (contains focal_length per frame)")
    parser.add_argument("output", help="Output rendered video path")
    parser.add_argument("--height",       type=int,   default=720,
                        help="Panel height in pixels (default: 720)")
    parser.add_argument("--graph-height",  type=int,   default=160,
                        help="Graph strip height per biomarker (default: 160)")
    parser.add_argument("--biomarker",     type=str,   nargs="+", default=[],
                        help="Biomarker(s) to plot, e.g. 'left knee flexion'")
    parser.add_argument("--smplx_path",    default=find_smplx_path(),
                        help="Path to SMPL-X model folder (contains SMPLX_*.pkl)")
    args = parser.parse_args()

    if args.smplx_path is None:
        sys.exit("ERROR: SMPL-X model path not found. Pass --smplx_path explicitly.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load SMPL-X JSON ─────────────────────────────────────────────────────
    print(f"Loading SMPL-X JSON: {args.json}")
    with open(args.json) as f:
        jdata = json.load(f)

    T = len(jdata["transl"])
    print(f"  Frames in JSON: {T}")

    # JSON stores rotation matrices; convert to axis-angle for smplx forward pass
    global_orient_aa = rotmat_to_aa(
        np.array(jdata["global_orient"], dtype=np.float32)[:, 0]   # [T,3,3] → [T,3]
    )
    body_pose_aa = rotmat_to_aa(
        np.array(jdata["body_pose"], dtype=np.float32)              # [T,21,3,3]
    ).reshape(T, 63)                                                 # → [T,63]
    left_hand_aa = rotmat_to_aa(
        np.array(jdata["left_hand_pose"], dtype=np.float32)         # [T,15,3,3]
    ).reshape(T, 45)                                                 # → [T,45]
    right_hand_aa = rotmat_to_aa(
        np.array(jdata["right_hand_pose"], dtype=np.float32)        # [T,15,3,3]
    ).reshape(T, 45)                                                 # → [T,45]
    betas      = np.array(jdata["betas"],      dtype=np.float32)            # [T,10]
    transl     = np.array(jdata["transl"],     dtype=np.float32)            # [T,3]  (body-local offset)
    expression = np.array(jdata.get("expression",
                           np.zeros((T, 10), dtype=np.float32)),
                          dtype=np.float32)                                  # [T,10]

    # ── SMPL-X forward pass (full batch) ────────────────────────────────────
    print("Running SMPL-X forward pass ...")
    smplx_model = smplx.SMPLX(
        model_path=args.smplx_path,
        gender="neutral",
        use_pca=False,
        num_betas=10,
        num_expression_coeffs=10,
    ).to(device)

    def _t(x): return torch.from_numpy(x).to(device)

    zeros3 = torch.zeros((T, 3), device=device, dtype=torch.float32)

    with torch.no_grad():
        out = smplx_model(
            global_orient    = _t(global_orient_aa),
            body_pose        = _t(body_pose_aa),
            left_hand_pose   = _t(left_hand_aa),
            right_hand_pose  = _t(right_hand_aa),
            betas            = _t(betas),
            transl           = _t(transl),
            expression       = _t(expression),
            jaw_pose         = zeros3,
            leye_pose        = zeros3,
            reye_pose        = zeros3,
        )
    all_verts  = out.vertices.cpu().numpy()                  # [T, V, 3]
    all_joints = out.joints.cpu().numpy()[:, :len(JOINT_NAMES)]  # [T, J, 3]
    faces      = smplx_model.faces                           # (F, 3) int32
    print(f"  Vertices shape: {all_verts.shape}")

    # ── Compute biomarker angles for all frames ───────────────────────────────
    bio_tracks = []   # [(label, [float * T]), ...]
    for query in args.biomarker:
        try:
            angles = []
            for fi in range(T):
                angles.append(compute_angle(query, all_joints[fi]))
            bio_tracks.append((query, angles))
            valid = [a for a in angles if np.isfinite(a)]
            print(f"  Biomarker '{query}': min={min(valid):.1f}° max={max(valid):.1f}°")
        except ValueError as e:
            print(f"  WARNING: {e}")

    # ── Load pred_cam_t + focal lengths from NPZ ─────────────────────────────
    print(f"Loading NPZ: {args.npz}")
    npz_data      = np.load(args.npz, allow_pickle=True)
    focal_lengths = npz_data["focal_length"]            # [T] float32
    pred_cam_t    = npz_data["pred_cam_t"]              # [T, 3] – camera-space translation
    if focal_lengths.ndim == 0:
        focal_lengths = np.full(T, float(focal_lengths))
    print(f"  focal_length sample : {focal_lengths[0]:.1f}")
    print(f"  pred_cam_t[0]       : {pred_cam_t[0]}")

    # ── Open video ───────────────────────────────────────────────────────────
    print(f"Opening video: {args.video}")
    cap   = cv2.VideoCapture(args.video)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), T)

    out_h    = args.height
    graph_h  = args.graph_height if bio_tracks else 0
    panel_w  = int(round(vid_w * out_h / vid_h))
    total_w  = panel_w * 3
    total_h  = out_h + graph_h * len(bio_tracks)

    # focal_length is stored for the original resolution; scale it to panel size
    fl_scale      = out_h / vid_h
    focal_lengths = focal_lengths * fl_scale
    print(f"  focal_length (scaled to {out_h}px): {focal_lengths[0]:.1f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (total_w, total_h))
    if not writer.isOpened():
        sys.exit(f"ERROR: cannot open VideoWriter for {args.output}")

    # ── Pre-render graph base images ─────────────────────────────────────────
    _graph_x_left  = int(total_w * 0.065)
    _graph_x_right = int(total_w * 0.985)

    def _slider_x(fi, n):
        t = fi / max(n - 1, 1)
        return int(_graph_x_left + t * (_graph_x_right - _graph_x_left))

    graph_bases = []
    for label, angles in bio_tracks:
        print(f"Pre-rendering graph: {label} ...")
        base = render_angle_graph_base(angles, total_w, graph_h, label=label)
        graph_bases.append((base, len(angles)))

    print(f"Writing {n_frames} frames → {args.output}  ({total_w}x{total_h})")

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: video ended at frame {i}")
            break

        # ── Panel 1: raw ─────────────────────────────────────────────────────
        raw_panel = cv2.resize(frame, (panel_w, out_h), interpolation=cv2.INTER_AREA)

        # ── SMPL-X vertices for this frame ───────────────────────────────────
        # SMPL-X verts already include transl (body-local offset).
        # pred_cam_t (from NPZ) is the camera-space translation from SAM-3D-Body.
        # Together they give the correct camera-space position.
        cam_t_npz    = pred_cam_t[i] if i < len(pred_cam_t) else pred_cam_t[-1]
        verts_in_cam = all_verts[i] + cam_t_npz                          # camera space
        cam_t        = (verts_in_cam.max(axis=0) + verts_in_cam.min(axis=0)) / 2
        verts_local  = verts_in_cam - cam_t                               # centred at origin

        fl = float(focal_lengths[i]) if i < len(focal_lengths) else float(focal_lengths[-1])
        renderer = Renderer(focal_length=fl, faces=faces)

        # ── Panel 2: mesh overlay on raw frame (front view) ──────────────────
        bg_front = raw_panel.copy().astype(np.float32)      # BGR uint8 → float
        front_rgb = renderer(
            verts_local, cam_t, bg_front,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
        ) * 255
        front_panel = cv2.cvtColor(front_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

        # ── Panel 3: mesh on white background (side view) ────────────────────
        white_bg = np.full_like(raw_panel, 255, dtype=np.float32)
        side_rgb = renderer(
            verts_local, cam_t, white_bg,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            side_view=True,
        ) * 255
        side_panel = cv2.cvtColor(side_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

        combined = np.hstack([raw_panel, front_panel, side_panel])

        rows = [combined]
        for (base, n_angles) in graph_bases:
            strip = base.copy()
            sx = _slider_x(i, n_angles)
            cv2.line(strip, (sx, 0), (sx, graph_h - 1), (255, 230, 80), 2, cv2.LINE_AA)
            rows.append(strip)
        writer.write(np.vstack(rows))

        if (i + 1) % 30 == 0 or i == n_frames - 1:
            print(f"  {i + 1}/{n_frames} frames done")

    cap.release()
    writer.release()
    print(f"\nDone! Saved → {args.output}")


if __name__ == "__main__":
    main()
