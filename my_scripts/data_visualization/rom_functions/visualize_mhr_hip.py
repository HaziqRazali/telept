#!/usr/bin/env python3
"""Render an MHR pose with a hip range-of-motion measurement overlay.

The output contains:

    [original video frame] [MHR body-aligned front] [MHR sagittal view]

Hip flexion and extension use the hip-to-knee (thigh) vector projected into
the subject's body sagittal plane.  Hip abduction uses the same vector
projected into the body coronal plane.  In every case the angle starts at the
hip-to-trunk-down reference.  This is appropriate for the standing leg-raise
and side-leg-raise tasks: the contralateral leg remains supporting the body
while the selected thigh moves.

For the rendered SAM-3D-Body video timeline, run:

    /home/haziq/anaconda3/envs/mhr_new/bin/python \
        /data/haziq/telept/my_scripts/data_visualization/rom_functions/visualize_mhr_hip.py \
        /data/haziq/mocap/data/brett/train/s01/sam3d/cam0/hip_osteoarthritis_mhr_outputs.npz \
        --time 6.139 --side right --movement flexion

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

# Reuse the tested headless MHR mesh renderer and frame/timestamp utilities.
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
    _finite_rows,
    _letterbox,
    _load_mhr_faces,
    _load_video_frame,
    _make_view_spec,
    _put_text,
    _read_meta,
    _render_mesh,
    _resolve_frame,
)
from rom_functions.mhr_hip import (  # noqa: E402
    MHR_BODY_EDGES,
    MHR_HIP_JOINTS,
    HipROMResult,
    compute_hip_abduction,
    compute_hip_extension,
    compute_hip_flexion,
    select_hip_side,
)
from rom_functions.mhr_shoulder import build_mhr_body_frame  # noqa: E402


def _draw_hip_arc(image: np.ndarray, spec, result: HipROMResult) -> None:
    """Draw the selected planar hip-ROM arc."""

    down = result.body_frame.down
    positive = result.positive_axis
    up = result.body_frame.up
    thigh = result.measurement_unit

    a0 = float(np.arctan2(np.dot(down, up), np.dot(down, positive)))
    a1 = float(np.arctan2(np.dot(thigh, up), np.dot(thigh, positive)))
    delta = (a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi
    radius = max(0.08, 0.45 * result.thigh_length)
    samples = max(20, int(abs(delta) * 30.0))
    angles = np.linspace(a0, a0 + delta, samples)
    arc_world = np.vstack(
        [
            result.hip + radius * (np.cos(angle) * positive + np.sin(angle) * up)
            for angle in angles
        ]
    )
    arc_pixels = spec.project(arc_world)
    cv2.polylines(image, [arc_pixels.reshape(-1, 1, 2)], False, ARC_COLOR, 5, cv2.LINE_AA)

    midpoint = a0 + delta / 2.0
    label_world = result.hip + (radius + 0.06) * (
        np.cos(midpoint) * positive + np.sin(midpoint) * up
    )
    label = _clip_point(spec.project(label_world[None, :])[0], spec.width, spec.height)
    _put_text(image, f"{result.angle_deg:+.1f} deg", label, ARC_COLOR, 0.72, 2)


def _draw_hip_measurement(
    image: np.ndarray,
    spec,
    joints: np.ndarray,
    result: HipROMResult,
    title: str,
    draw_arc: bool,
    draw_positive_axis: bool = False,
    show_result_text: bool = True,
) -> None:
    """Overlay the skeleton, thigh/reference vectors, labels, and arc."""

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
        MHR_HIP_JOINTS["left_hip"],
        MHR_HIP_JOINTS["left_knee"],
    ]
    for index in key_joint_indices:
        cv2.circle(image, tuple(projected_joints[index]), 5, JOINT_COLOR, -1, cv2.LINE_AA)

    p_hip = tuple(spec.project(result.hip[None, :])[0])
    p_knee = tuple(spec.project(result.knee[None, :])[0])
    p_measurement = tuple(spec.project(result.measurement_end[None, :])[0])
    p_reference = tuple(spec.project(result.reference_end[None, :])[0])

    # Red is the actual 3-D thigh.  Cyan is its body-plane projection, which
    # is the vector used for the selected hip-ROM angle.
    _draw_arrow(image, p_hip, p_knee, ACTUAL_ARM_COLOR, 5)
    _draw_arrow(image, p_hip, p_measurement, MEASURED_ARM_COLOR, 5)
    _draw_dashed_line(image, p_hip, p_reference, REFERENCE_COLOR, 4)
    if draw_positive_axis:
        positive_end = result.hip + result.positive_axis * (0.72 * result.thigh_length)
        p_positive = tuple(spec.project(positive_end[None, :])[0])
        _draw_dashed_line(image, p_hip, p_positive, POSITIVE_AXIS_COLOR, 3, dash=10, gap=8)
        positive_label = {
            "hip flexion": "forward +",
            "hip extension": "posterior +",
            "hip abduction": "outward +",
        }.get(result.measurement_name, "positive +")
        _put_text(image, positive_label, (p_positive[0] + 8, p_positive[1]), POSITIVE_AXIS_COLOR, 0.46, 1)
    cv2.circle(image, p_hip, 9, WHITE, -1, cv2.LINE_AA)
    cv2.circle(image, p_hip, 9, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.circle(image, p_knee, 7, ACTUAL_ARM_COLOR, -1, cv2.LINE_AA)

    if show_result_text:
        _put_text(
            image,
            f"{result.side.capitalize()} {result.measurement_name}: {result.angle_deg:+.1f} deg",
            (22, 34),
            ARC_COLOR,
            0.72,
            2,
        )
    _put_text(image, "red = actual 3-D thigh", (22, spec.height - 62), ACTUAL_ARM_COLOR, 0.52, 1)
    _put_text(image, f"cyan = {result.plane_name}-plane thigh vector used", (22, spec.height - 40), MEASURED_ARM_COLOR, 0.50, 1)
    _put_text(image, "yellow = trunk-down reference", (22, spec.height - 18), (0, 150, 190), 0.52, 1)
    if draw_positive_axis:
        _put_text(image, "green = positive movement direction", (22, spec.height - 84), POSITIVE_AXIS_COLOR, 0.52, 1)

    if draw_arc:
        _draw_hip_arc(image, spec, result)


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
        choices=("flexion", "abduction", "extension"),
        default="flexion",
        help="Hip movement to measure (default: flexion)",
    )
    parser.add_argument("--time", type=float, default=6.139, help="Timestamp in seconds")
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
        choices=("auto", "right", "left"),
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
    selected_side = select_hip_side(joints, args.side, body_frame)
    if args.movement == "abduction":
        result = compute_hip_abduction(joints, selected_side, body_frame)
        primary_title = "MHR body-aligned CORONAL / FRONT"
        secondary_title = "MHR body-aligned SAGITTAL"
        primary_arc = True
        secondary_arc = False
        primary_positive_axis = True
        secondary_positive_axis = False
    elif args.movement == "extension":
        result = compute_hip_extension(joints, selected_side, body_frame)
        primary_title = "MHR body-aligned FRONT"
        secondary_title = "MHR body-aligned SAGITTAL"
        primary_arc = False
        secondary_arc = True
        primary_positive_axis = False
        secondary_positive_axis = True
    else:
        result = compute_hip_flexion(joints, selected_side, body_frame)
        primary_title = "MHR body-aligned FRONT"
        secondary_title = "MHR body-aligned SAGITTAL"
        primary_arc = False
        secondary_arc = True
        primary_positive_axis = False
        secondary_positive_axis = False
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
    print(f"ROM angle:       {result.angle_deg:+.2f} deg")

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
    sagittal_spec = _make_view_spec(
        vertices,
        joints,
        result,
        args.panel_width,
        args.height,
        body_frame.forward,
        body_frame.up,
        body_frame.right,
    )
    front_panel = _render_mesh(vertices, faces, front_spec)
    sagittal_panel = _render_mesh(vertices, faces, sagittal_spec)
    _draw_hip_measurement(
        front_panel,
        front_spec,
        joints,
        result,
        primary_title,
        draw_arc=primary_arc,
        draw_positive_axis=primary_positive_axis,
    )
    _draw_hip_measurement(
        sagittal_panel,
        sagittal_spec,
        joints,
        result,
        secondary_title,
        draw_arc=secondary_arc,
        draw_positive_axis=secondary_positive_axis,
        show_result_text=False,
    )

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

    panels.extend((front_panel, sagittal_panel))
    output = args.output or _default_output(args.npz, args.time_domain, args.time, args.movement)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.hconcat(panels)):
        raise RuntimeError(f"Could not write output image: {output}")
    print(f"Saved:            {output}")


if __name__ == "__main__":
    main()
