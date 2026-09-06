#!/usr/bin/env python3
"""Visualize an RGBD point cloud for one frame of a TelePT iPad recording.

Loads a single RGB frame plus its paired depth frame, lifts every valid depth pixel
to metric 3D (meters, pinhole backprojection using the depth-intrinsics), colors each
point from the RGB image, and renders an interactive 3D scatter plot. Optionally
overlays manually annotated and/or MMPose keypoints (also lifted to 3D) as markers.

This answers "what does the captured 3D look like?" and lets you sanity-check that the
depth/calibration lifting is producing plausible geometry for the annotated joints.

Usage:
    python visualize_rgbd_pointcloud.py \
        --data-root data/NUS/val \
        --subject haziq_upperlimb_right_24082026 \
        --session session_1787566918897 \
        --trial dynamic_03 \
        --frame 30 [--step 2] [--annotations PATH] [--mmpose] [--undistort]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# add repo root to path so my_scripts.* imports resolve
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

from my_scripts.data_evaluation.recordings import find_recording  # noqa: E402
from my_scripts.data_evaluation.rgbd_geometry import (  # noqa: E402
    DepthZipReader,
    lift_points_2d,
    read_video_frame,
)


def _load_annotations(path: str) -> dict | None:
    p = Path(path)
    if not p.is_file():
        print(f"[warn] no annotations file at {p}")
        return None
    import json
    return json.loads(p.read_text())


def _get_annotation_points(ann: dict, frame: int) -> dict[str, dict[str, float]]:
    """Extract {name: {x,y}} for the closest 2D frame annotation to `frame`."""
    tasks = ann.get("tasks", {})
    # flatten across task definitions and entries; pick any task whose frame matches
    points: dict[str, dict[str, float]] = {}
    for _tid, entry in tasks.items():
        if not isinstance(entry, dict):
            continue
        blind = entry.get("blind_points_2d")
        if isinstance(blind, dict):
            points.update(blind)
        for pose in ("t1", "t2"):
            by_pose = entry.get("blind_points_2d_by_pose", {})
            if isinstance(by_pose, dict) and isinstance(by_pose.get(pose), dict):
                points.update(by_pose[pose])
    return points if points else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data/NUS/val")
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--step", type=int, default=2, help="subsample stride for the point cloud")
    ap.add_argument("--annotations", default=None, help="annotation JSON to overlay keypoints")
    ap.add_argument("--mmpose", action="store_true", help="overlay MMPose keypoints")
    ap.add_argument("--undistort", action="store_true",
                    help="apply the LUT undistortion before lifting")
    ap.add_argument("--max-points", type=int, default=400000)
    ap.add_argument("--save", default=None,
                    help="save the figure to this PNG instead of opening a window")
    args = ap.parse_args()

    rec = find_recording(Path(args.data_root), args.subject, args.session, args.trial)
    if rec.depth_path is None:
        raise SystemExit("no depth archive for this recording")

    rgb = read_video_frame(str(rec.video_path), args.frame)
    print(f"RGB frame {args.frame}: {rgb.shape[1]}x{rgb.shape[0]}")

    reader = DepthZipReader(str(rec.depth_path))
    try:
        depth = reader.read(args.frame)
        if depth is None:
            raise SystemExit(f"depth frame {args.frame} unavailable")
        fx, fy, cx, cy = reader.fx, reader.fy, reader.cx, reader.cy
    finally:
        reader.close()
    print(f"depth frame: {depth.shape[1]}x{depth.shape[0]}  "
          f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

    # --- lift the full depth image to a metric point cloud -----------------
    h, w = depth.shape
    ys, xs = np.nonzero(np.isfinite(depth) & (depth > 0))
    d = depth[ys, xs]
    x = (xs - cx) * d / fx
    y = (ys - cy) * d / fy
    z = d
    # map depth pixel -> rgb pixel to sample color
    rw, rh = rgb.shape[1], rgb.shape[0]
    u_rgb = np.clip(np.round((xs + 0.5) * rw / w - 0.5).astype(int), 0, rw - 1)
    v_rgb = np.clip(np.round((ys + 0.5) * rh / h - 0.5).astype(int), 0, rh - 1)
    colors = rgb[v_rgb, u_rgb].astype(float) / 255.0  # BGR -> keep; matplotlib RGB
    colors = colors[:, ::-1]  # BGR -> RGB

    if args.step > 1:
        idx = np.arange(0, x.shape[0], args.step)
        x, y, z, colors = x[idx], y[idx], z[idx], colors[idx]
    if x.shape[0] > args.max_points:
        idx = np.random.default_rng(0).choice(x.shape[0], args.max_points, replace=False)
        x, y, z, colors = x[idx], y[idx], z[idx], colors[idx]

    print(f"point cloud: {x.shape[0]} points")

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, c=colors, s=1, marker=".", depthshade=False, alpha=0.7)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"RGBD point cloud - {args.subject} {args.trial} frame {args.frame}")

    # --- overlay keypoints -------------------------------------------------
    def plot_points(points: dict, marker: str, color: str, label: str) -> None:
        if not points:
            return
        reader = DepthZipReader(str(rec.depth_path))
        try:
            lifted = lift_points_2d(points, reader, args.frame,
                                    rgb.shape[1], rgb.shape[0], 2)
        finally:
            reader.close()
        good = {k: v for k, v in lifted.points.items() if np.isfinite(v).all()}
        if not good:
            print(f"[warn] no valid depth for {label} points")
            return
        ax.scatter([v[0] for v in good.values()], [v[1] for v in good.values()],
                   [v[2] for v in good.values()], marker=marker, c=color,
                   s=90, depthshade=False, label=label)
        for name, v in good.items():
            ax.text(v[0], v[1], v[2], name, color=color, fontsize=7)

    if args.annotations:
        ann = _load_annotations(args.annotations)
        if ann:
            pts = _get_annotation_points(ann, args.frame)
            plot_points(pts, "o", "red", "manual annotation")
    if args.mmpose and rec.mmpose_path is not None:
        from my_scripts.data_evaluation.rgbd_geometry import (  # noqa: PLC0415
            load_mmpose_json,
            mmpose_points_for_frame,
        )
        labels, keypoints, scores = load_mmpose_json(str(rec.mmpose_path), 0.5)
        pts = mmpose_points_for_frame(labels, keypoints, scores, args.frame, 0.5)
        plot_points(pts, "^", "cyan", "MMPose")

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=140)
        print(f"saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
