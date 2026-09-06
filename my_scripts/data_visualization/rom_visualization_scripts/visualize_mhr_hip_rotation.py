#!/usr/bin/env python3
"""Render an MHR pose with a hip external/internal rotation overlay.

The output contains:

    [original video frame] [MHR body-aligned front] [MHR thigh-axis rotation]

This is intended for the standing leg-raise setup in which the hip is flexed
and the knee is approximately 90 degrees.  The thigh is the rotation axis;
the knee-to-foot vector is projected into the plane perpendicular to the
thigh.  A downward lower-leg direction is used as neutral and the anatomical
outward direction is positive external rotation.

For the rendered SAM-3D-Body video timeline, run:

    /home/haziq/anaconda3/envs/mhr_new/bin/python \
        /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/visualize_mhr_hip_rotation.py \
        /data/haziq/mocap/data/brett/train/s01/sam3d/cam0/hip_osteoarthritis_mhr_outputs.npz \
        --time 12.611 --movement internal_rotation --side right

Use ``--time-domain original`` when the timestamp comes from the original
source video instead.
"""

import argparse
import os
import sys
from pathlib import Path

# Must be set before importing pyrender/OpenGL on a headless workstation.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
if "anaconda" in os.environ.get("__EGL_VENDOR_LIBRARY_DIRS", ""):
    os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = "/usr/share/glvnd/egl_vendor.d"

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_VISUALIZATION_DIR = SCRIPT_DIR.parent
for import_path in (DATA_VISUALIZATION_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

# Reuse the tested headless MHR mesh renderer and timestamp utilities.
from visualize_mhr_shoulder_flexion import (  # noqa: E402
    ACTUAL_ARM_COLOR,
    ARC_COLOR,
    JOINT_COLOR,
    MEASURED_ARM_COLOR,
    POSITIVE_AXIS_COLOR,
    REFERENCE_COLOR,
    SKELETON_COLOR,
    TEXT_COLOR,
    WHITE,
    _clip_point,
    _draw_arrow,
    _draw_dashed_line,
    _draw_rotation_mesh_inset,
    _letterbox,
    _load_mhr_faces,
    _load_video_frame,
    _make_rotation_mesh_spec,
    _make_view_spec,
    _put_text,
    _read_meta,
    _render_mesh,
    _resolve_frame,
)
from rom_visualization_scripts.mhr_hip import (  # noqa: E402
    MHR_BODY_EDGES,
    MHR_HIP_JOINTS,
    HipRotationResult,
    compute_hip_rotation,
    mhr_upleg_twist_deg,
)
from rom_visualization_scripts.mhr_shoulder import build_mhr_body_frame  # noqa: E402


def _draw_hip_rotation_measurement(
    image: np.ndarray,
    spec,
    joints: np.ndarray,
    result: HipRotationResult,
    mhr_twist_deg: float | None,
    show_result_text: bool = True,
) -> None:
    """Overlay the thigh/shank vectors and hip-rotation labels."""

    projected_joints = spec.project(joints)
    for a, b in MHR_BODY_EDGES:
        cv2.line(
            image,
            tuple(projected_joints[a]),
            tuple(projected_joints[b]),
            SKELETON_COLOR,
            2,
            cv2.LINE_AA,
        )

    key_joint_indices = [
        MHR_HIP_JOINTS["root"],
        MHR_HIP_JOINTS["c_spine3"],
        110,
        MHR_HIP_JOINTS["right_hip"],
        MHR_HIP_JOINTS["right_knee"],
        MHR_HIP_JOINTS["right_foot"],
        MHR_HIP_JOINTS["left_hip"],
        MHR_HIP_JOINTS["left_knee"],
    ]
    for index in key_joint_indices:
        cv2.circle(image, tuple(projected_joints[index]), 5, JOINT_COLOR, -1, cv2.LINE_AA)

    p_hip = tuple(spec.project(result.hip[None, :])[0])
    p_knee = tuple(spec.project(result.knee[None, :])[0])
    p_foot = tuple(spec.project(result.foot[None, :])[0])
    p_measurement = tuple(spec.project(result.measurement_end[None, :])[0])
    p_reference = tuple(spec.project(result.reference_end[None, :])[0])
    positive_end = result.knee + result.positive_axis * result.measurement_length
    p_positive = tuple(spec.project(positive_end[None, :])[0])

    # Red follows the reconstructed thigh/shank.  Cyan is the shank after
    # removing its component parallel to the thigh, which is used for axial
    # rotation.  Yellow is the downward neutral reference; green only marks
    # the positive external-rotation direction.
    _draw_arrow(image, p_hip, p_knee, ACTUAL_ARM_COLOR, 5)
    _draw_arrow(image, p_knee, p_foot, ACTUAL_ARM_COLOR, 5)
    _draw_arrow(image, p_knee, p_measurement, MEASURED_ARM_COLOR, 5)
    _draw_dashed_line(image, p_knee, p_reference, REFERENCE_COLOR, 4)
    _draw_dashed_line(image, p_knee, p_positive, POSITIVE_AXIS_COLOR, 3, dash=10, gap=8)
    cv2.circle(image, p_hip, 9, WHITE, -1, cv2.LINE_AA)
    cv2.circle(image, p_hip, 9, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.circle(image, p_knee, 8, WHITE, -1, cv2.LINE_AA)
    cv2.circle(image, p_knee, 8, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.circle(image, p_foot, 7, ACTUAL_ARM_COLOR, -1, cv2.LINE_AA)

    if show_result_text:
        _put_text(
            image,
            f"{result.side.capitalize()} {result.rotation_name} rotation: "
            f"{result.requested_angle_deg:+.1f} deg",
            (22, 34),
            ARC_COLOR,
            0.72,
            2,
        )
    _put_text(image, "red = actual 3-D thigh + shank", (22, spec.height - 84), ACTUAL_ARM_COLOR, 0.50, 1)
    _put_text(image, "cyan = transverse shank vector used", (22, spec.height - 62), MEASURED_ARM_COLOR, 0.52, 1)
    _put_text(image, "yellow = shank-down neutral reference", (22, spec.height - 40), (0, 150, 190), 0.50, 1)
    _put_text(image, "green = external-rotation positive direction", (22, spec.height - 18), POSITIVE_AXIS_COLOR, 0.50, 1)


def _draw_rotation_plane_diagram(
    width: int,
    height: int,
    result: HipRotationResult,
    mhr_twist_deg: float | None,
    show_result_text: bool = True,
    mesh_vertices: np.ndarray | None = None,
    mesh_faces: np.ndarray | None = None,
) -> np.ndarray:
    """Create a transverse-plane diagram viewed along the thigh."""

    image = np.full((height, width, 3), (249, 249, 249), dtype=np.uint8)
    has_mesh = mesh_vertices is not None and mesh_faces is not None
    center = np.array(
        [width * (0.68 if has_mesh else 0.52), height * 0.53],
        dtype=np.float64,
    )
    radius = min(width * (0.24 if has_mesh else 0.29), height * 0.28)

    def point(angle: float, length: float) -> tuple[int, int]:
        # Positive external rotation is drawn upward from the neutral vector.
        p = center + np.array([np.cos(angle) * length, -np.sin(angle) * length])
        return _clip_point(p, width, height)

    theta = float(np.radians(result.angle_deg))
    ref_end = point(0.0, radius)
    positive_end = point(np.pi / 2.0, radius)
    actual_end = point(theta, radius)
    arc_radius = radius * 0.72
    arc_angles = np.linspace(0.0, theta, max(20, int(abs(theta) * 30.0)))
    arc = np.array([point(float(a), arc_radius) for a in arc_angles], dtype=np.int32)

    cv2.circle(image, tuple(np.rint(center).astype(int)), int(radius), (218, 222, 228), 2, cv2.LINE_AA)
    _draw_dashed_line(
        image,
        tuple(np.rint(center).astype(int)),
        ref_end,
        REFERENCE_COLOR,
        5,
        dash=14,
        gap=9,
    )
    _draw_dashed_line(
        image,
        tuple(np.rint(center).astype(int)),
        positive_end,
        POSITIVE_AXIS_COLOR,
        4,
        dash=12,
        gap=8,
    )
    _draw_arrow(image, tuple(np.rint(center).astype(int)), actual_end, MEASURED_ARM_COLOR, 7)
    if len(arc) >= 2:
        cv2.polylines(image, [arc.reshape(-1, 1, 2)], False, ARC_COLOR, 6, cv2.LINE_AA)
    cv2.circle(image, tuple(np.rint(center).astype(int)), 10, TEXT_COLOR, -1, cv2.LINE_AA)

    label_point = point(theta / 2.0, arc_radius + 34.0)
    _put_text(image, f"{result.angle_deg:+.1f} deg", (label_point[0] - 28, label_point[1]), ARC_COLOR, 0.75, 2)
    if show_result_text:
        _put_text(
            image,
            f"{result.side.capitalize()} {result.rotation_name} rotation: "
            f"{result.requested_angle_deg:+.1f} deg",
            (22, 34),
            ARC_COLOR,
            0.70,
            2,
        )
    _put_text(image, "yellow = shank-down / neutral reference", (22, height - 84), (0, 150, 190), 0.50, 1)
    _put_text(image, "cyan = transverse shank direction", (22, height - 62), MEASURED_ARM_COLOR, 0.52, 1)
    _put_text(image, "green = external-rotation positive direction", (22, height - 40), POSITIVE_AXIS_COLOR, 0.50, 1)
    _put_text(image, "magenta = measured axial angle", (22, height - 18), ARC_COLOR, 0.52, 1)
    _put_text(image, "neutral / shank-down", (ref_end[0] - 150, ref_end[1] - 24), REFERENCE_COLOR, 0.47, 1)
    _put_text(image, "external +", (positive_end[0] + 10, positive_end[1] - 8), POSITIVE_AXIS_COLOR, 0.50, 1)
    _draw_rotation_mesh_inset(image, mesh_vertices, mesh_faces, result)
    return image


def _default_output(npz_path: Path, time_domain: str, time_s: float, movement: str) -> Path:
    stamp = f"{time_s:.3f}".replace(".", "p")
    return Path(__file__).resolve().parent / "results" / (
        f"{npz_path.stem}_hip_{movement}_{time_domain}_t{stamp}.png"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="SAM-3D-Body MHR output NPZ")
    parser.add_argument(
        "--movement",
        "--motion",
        dest="movement",
        choices=("external_rotation", "internal_rotation"),
        default="external_rotation",
        help="Hip rotation direction to measure",
    )
    parser.add_argument("--time", type=float, default=15.814, help="Timestamp in seconds")
    parser.add_argument(
        "--time-domain",
        choices=("rendered", "original"),
        default="rendered",
        help="Use the valid-frame rendered-video timeline or original-video timeline",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Original zero-based video/NPZ frame; overrides --time",
    )
    parser.add_argument(
        "--side",
        choices=("right", "left"),
        default="right",
        help="Hip to annotate (default: right)",
    )
    parser.add_argument("--video", type=Path, default=None, help="Original video; inferred from NPZ meta")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument(
        "--mhr-root",
        type=Path,
        default=Path("/home/haziq/MHR"),
        help="MHR repository root containing the mhr package",
    )
    parser.add_argument("--height", type=int, default=900, help="Panel height in pixels")
    parser.add_argument("--panel-width", type=int, default=800, help="Mesh panel width")
    parser.add_argument("--no-source", action="store_true", help="Omit the original-video panel")
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    meta = _read_meta(data)
    selection = _resolve_frame(data, args.time, args.time_domain, args.frame)
    row = selection["row"]
    vertices = np.asarray(data["vertices"][row], dtype=np.float64)
    joints = np.asarray(data["pred_joint_coords"][row], dtype=np.float64)
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise RuntimeError(f"Selected NPZ row {row} contains NaN/Inf values.")

    body_frame = build_mhr_body_frame(joints)
    rotation_name = "external" if args.movement == "external_rotation" else "internal"
    result = compute_hip_rotation(joints, args.side, rotation_name, body_frame)
    mhr_twist_deg = mhr_upleg_twist_deg(data["body_pose_params"][row], args.side)
    faces = _load_mhr_faces(args.mhr_root)
    if faces.max(initial=-1) >= len(vertices):
        raise RuntimeError(
            f"MHR topology references vertex {faces.max()}, but selected mesh has {len(vertices)} vertices."
        )

    print(f"NPZ:             {args.npz}")
    print(f"NPZ row:         {row}")
    print(f"Original frame:  {selection['original_frame']} ({selection['original_time']:.3f}s)")
    print(f"Rendered frame:  {selection['rendered_frame']} ({selection['rendered_time']:.3f}s)")
    print(f"Movement:        {result.measurement_name}")
    print(f"Hip side:        {result.side}")
    print(f"ROM angle:       {result.requested_angle_deg:+.2f} deg ({result.angle_deg:+.2f} signed; external +)")
    print(f"MHR upleg twist: {mhr_twist_deg:+.2f} deg (model-local)")

    # Conventional anterior view: anatomical right appears on the viewer's
    # left, so reverse the horizontal screen axis.
    front_spec = _make_view_spec(
        vertices,
        joints,
        result,
        args.panel_width,
        args.height,
        -body_frame.right,
        body_frame.up,
        body_frame.forward,
    )
    front_panel = _render_mesh(vertices, faces, front_spec)
    rotation_spec = _make_rotation_mesh_spec(
        vertices,
        result,
        args.panel_width,
        args.height,
    )
    rotation_panel = _render_mesh(vertices, faces, rotation_spec)
    _draw_hip_rotation_measurement(
        front_panel,
        front_spec,
        joints,
        result,
        mhr_twist_deg,
        show_result_text=True,
    )
    _draw_hip_rotation_measurement(
        rotation_panel,
        rotation_spec,
        joints,
        result,
        mhr_twist_deg,
        show_result_text=False,
    )

    # Keep the same three-panel layout as the flexion visualizations: original
    # frame, body-aligned front mesh, and a full-body thigh-axis mesh view.
    panels = []
    if not args.no_source:
        video_path = args.video
        if video_path is None and meta.get("video_path"):
            video_path = Path(str(meta["video_path"]))
        source_frame = _load_video_frame(video_path, selection["original_frame"]) if video_path else None
        source_width = int(round(args.height * 576.0 / 1024.0))
        if source_frame is not None:
            source_width = int(round(args.height * source_frame.shape[1] / source_frame.shape[0]))
        source_panel = _letterbox(source_frame, source_width, args.height)
        if source_frame is None:
            _put_text(source_panel, "source video unavailable", (18, args.height // 2), (0, 0, 200), 0.6, 2)
        panels.append(source_panel)

    panels.extend((front_panel, rotation_panel))
    output = args.output or _default_output(args.npz, args.time_domain, args.time, args.movement)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.hconcat(panels)):
        raise RuntimeError(f"Could not write output image: {output}")
    print(f"Saved:            {output}")


if __name__ == "__main__":
    main()
