#!/usr/bin/env python3
"""Render one Synthium MHR JSON frame with a body-aligned ROM overlay.

The JSON runner stores MHR pose parameters rather than mesh vertices.  This
script reconstructs the mesh and 127-joint skeleton, then reuses the same ROM
calculations and annotation style as the SAM-3D NPZ visualizers.

Examples:

    /home/haziq/anaconda3/envs/mhr_new/bin/python \
        /data/haziq/telept/my_scripts/data_visualization/rom_functions/visualize_mhr_json_rom.py \
        --json /path/to/s01_cam0_brett_adhesive_capsulitis.json \
        --joint shoulder --movement flexion --time 10.076 --side right

JSON timestamps use the original video timeline.  The selected frame is
``round(time * fps)`` from the JSON metadata.
"""

from __future__ import annotations

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

from rom_functions.mhr_hip import (  # noqa: E402
    compute_hip_abduction,
    compute_hip_extension,
    compute_hip_flexion,
    compute_hip_rotation,
    mhr_upleg_twist_deg,
    select_hip_side,
)
from rom_functions.mhr_json import (  # noqa: E402
    load_mhr_json,
    load_mhr_model,
    reconstruct_mhr_json_frame,
    select_mhr_json_frame,
)
from rom_functions.mhr_shoulder import (  # noqa: E402
    compute_shoulder_abduction,
    compute_shoulder_flexion,
    compute_shoulder_rotation,
    mhr_uparm_twist_deg,
    select_shoulder_side,
)
from rom_functions.mhr_shoulder import build_mhr_body_frame  # noqa: E402
from visualize_mhr_hip import _draw_hip_measurement  # noqa: E402
from visualize_mhr_hip_rotation import _draw_hip_rotation_measurement  # noqa: E402
from visualize_mhr_shoulder_flexion import (  # noqa: E402
    ARC_COLOR,
    TEXT_COLOR,
    WHITE,
    _clip_point,
    _draw_measurement,
    _draw_rotation_measurement,
    _draw_rotation_plane_diagram as _draw_shoulder_rotation_plane_diagram,
    _letterbox,
    _make_rotation_mesh_spec,
    _make_view_spec,
    _put_text,
    _render_mesh,
    _load_video_frame,
)


SHOULDER_MOVEMENTS = {
    "flexion",
    "abduction",
    "external_rotation",
    "internal_rotation",
}
HIP_MOVEMENTS = {
    "flexion",
    "abduction",
    "extension",
    "external_rotation",
    "internal_rotation",
}


def _planar_panel_pair(
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    result,
    body_frame,
    joint: str,
    movement: str,
    width: int,
    height: int,
) -> list[np.ndarray]:
    """Render front/coronal and sagittal panels for a planar ROM result."""

    # Conventional anterior view: anatomical right appears on the viewer's
    # left, so reverse the horizontal screen axis.
    front_spec = _make_view_spec(
        vertices,
        joints,
        result,
        width,
        height,
        -body_frame.right,
        body_frame.up,
        body_frame.forward,
    )
    sagittal_spec = _make_view_spec(
        vertices,
        joints,
        result,
        width,
        height,
        body_frame.forward,
        body_frame.up,
        body_frame.right,
    )
    front_panel = _render_mesh(vertices, faces, front_spec)
    sagittal_panel = _render_mesh(vertices, faces, sagittal_spec)

    is_abduction = movement == "abduction"
    primary_title = (
        "MHR body-aligned CORONAL / FRONT"
        if is_abduction
        else "MHR body-aligned FRONT"
    )
    secondary_title = "MHR body-aligned SAGITTAL"
    primary_arc = is_abduction
    secondary_arc = not is_abduction
    primary_positive = is_abduction
    secondary_positive = movement == "extension"

    if joint == "shoulder":
        draw = _draw_measurement
    else:
        draw = _draw_hip_measurement

    draw(
        front_panel,
        front_spec,
        joints,
        result,
        primary_title,
        draw_arc=primary_arc,
        draw_positive_axis=primary_positive,
    )
    draw(
        sagittal_panel,
        sagittal_spec,
        joints,
        result,
        secondary_title,
        draw_arc=secondary_arc,
        draw_positive_axis=secondary_positive,
        show_result_text=False,
    )
    return [front_panel, sagittal_panel]


def _rotation_panel_pair(
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    result,
    body_frame,
    body_pose: np.ndarray,
    joint: str,
    width: int,
    height: int,
) -> list[np.ndarray]:
    """Render the body-aligned mesh and full-body rotation-axis view."""

    # Keep the same anterior-view convention for rotation panels.
    front_spec = _make_view_spec(
        vertices,
        joints,
        result,
        width,
        height,
        -body_frame.right,
        body_frame.up,
        body_frame.forward,
    )
    front_panel = _render_mesh(vertices, faces, front_spec)

    axis_spec = _make_rotation_mesh_spec(vertices, result, width, height)
    axis_panel = _render_mesh(vertices, faces, axis_spec)

    if joint == "shoulder":
        twist_deg = mhr_uparm_twist_deg(body_pose, result.side)
        _draw_rotation_measurement(
            front_panel,
            front_spec,
            joints,
            result,
            twist_deg,
            show_result_text=True,
        )
        _draw_rotation_measurement(
            axis_panel,
            axis_spec,
            joints,
            result,
            twist_deg,
            show_result_text=False,
        )
    else:
        twist_deg = mhr_upleg_twist_deg(body_pose, result.side)
        _draw_hip_rotation_measurement(
            front_panel,
            front_spec,
            joints,
            result,
            twist_deg,
            show_result_text=True,
        )
        _draw_hip_rotation_measurement(
            axis_panel,
            axis_spec,
            joints,
            result,
            twist_deg,
            show_result_text=False,
        )
    return [front_panel, axis_panel]


def _source_panel(
    video_path: Path | None,
    frame_idx: int,
    actual_time_s: float,
    requested_time_s: float | None,
    width: int,
    height: int,
) -> np.ndarray:
    """Load the original video frame without a timestamp overlay."""

    source = _load_video_frame(video_path, frame_idx) if video_path else None
    source_width = int(round(height * 576.0 / 1024.0))
    if source is not None:
        source_width = int(round(height * source.shape[1] / source.shape[0]))
    panel = _letterbox(source, source_width, height)
    if source is None:
        _put_text(
            panel,
            "source video unavailable",
            (18, height // 2),
            (0, 0, 200),
            0.6,
            2,
        )
    return panel


def _default_output(json_path: Path, joint: str, movement: str, time_s: float) -> Path:
    stamp = f"{time_s:.3f}".replace(".", "p")
    return Path(__file__).resolve().parent / "results" / (
        f"{json_path.stem}_{joint}_{movement}_t{stamp}.png"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path, help="Synthium MHR JSON timeseries")
    parser.add_argument(
        "--joint",
        "--body-part",
        choices=("shoulder", "hip"),
        required=True,
        help="Joint complex to annotate",
    )
    parser.add_argument(
        "--movement",
        "--motion",
        required=True,
        choices=("flexion", "abduction", "extension", "external_rotation", "internal_rotation"),
        help="ROM movement to measure",
    )
    parser.add_argument("--time", type=float, default=None, help="Timestamp in seconds")
    parser.add_argument("--frame", type=int, default=None, help="Zero-based JSON/source frame")
    parser.add_argument(
        "--side",
        choices=("auto", "right", "left"),
        default="right",
        help="Anatomical side to annotate (default: right)",
    )
    parser.add_argument("--video", type=Path, default=None, help="Override video path from JSON")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument(
        "--mhr-root",
        type=Path,
        default=Path("/home/haziq/MHR"),
        help="MHR repository root containing the assets",
    )
    parser.add_argument("--height", type=int, default=900, help="Panel height in pixels")
    parser.add_argument("--panel-width", type=int, default=800, help="Mesh panel width")
    parser.add_argument("--no-source", action="store_true", help="Omit the original-video panel")
    args = parser.parse_args()

    valid_movements = SHOULDER_MOVEMENTS if args.joint == "shoulder" else HIP_MOVEMENTS
    if args.movement not in valid_movements:
        choices = ", ".join(sorted(valid_movements))
        parser.error(f"{args.joint} supports: {choices}")
    if args.time is None and args.frame is None:
        parser.error("provide --time or --frame")

    data = load_mhr_json(args.json)
    frame, frame_idx, actual_time_s = select_mhr_json_frame(
        data,
        time_s=args.time,
        frame_override=args.frame,
    )
    model, faces = load_mhr_model(args.mhr_root)
    vertices, joints, body_pose = reconstruct_mhr_json_frame(frame, model)
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise RuntimeError(f"Reconstructed JSON frame {frame_idx} contains NaN/Inf values.")

    body_frame = build_mhr_body_frame(joints)
    if args.joint == "shoulder":
        selected_side = select_shoulder_side(joints, args.side, args.movement)
        if args.movement == "abduction":
            result = compute_shoulder_abduction(joints, selected_side, body_frame)
        elif args.movement in {"external_rotation", "internal_rotation"}:
            rotation = "external" if args.movement == "external_rotation" else "internal"
            result = compute_shoulder_rotation(joints, selected_side, rotation, body_frame)
        else:
            result = compute_shoulder_flexion(joints, selected_side, body_frame)
    else:
        selected_side = select_hip_side(joints, args.side, body_frame)
        if args.movement == "abduction":
            result = compute_hip_abduction(joints, selected_side, body_frame)
        elif args.movement == "extension":
            result = compute_hip_extension(joints, selected_side, body_frame)
        elif args.movement in {"external_rotation", "internal_rotation"}:
            rotation = "external" if args.movement == "external_rotation" else "internal"
            result = compute_hip_rotation(joints, selected_side, rotation, body_frame)
        else:
            result = compute_hip_flexion(joints, selected_side, body_frame)

    if not np.isfinite(faces).all() or faces.max(initial=-1) >= len(vertices):
        raise RuntimeError("MHR topology is incompatible with the reconstructed mesh.")

    if args.movement in {"external_rotation", "internal_rotation"}:
        panels = _rotation_panel_pair(
            vertices,
            joints,
            faces,
            result,
            body_frame,
            body_pose,
            args.joint,
            args.panel_width,
            args.height,
        )
        angle = result.requested_angle_deg
    else:
        panels = _planar_panel_pair(
            vertices,
            joints,
            faces,
            result,
            body_frame,
            args.joint,
            args.movement,
            args.panel_width,
            args.height,
        )
        angle = result.angle_deg

    video_path = args.video
    if video_path is None and data.get("video"):
        video_path = Path(str(data["video"]))
    if video_path is not None:
        video_path = video_path.expanduser().resolve()

    output_panels = []
    if not args.no_source:
        output_panels.append(
            _source_panel(
                video_path,
                frame_idx,
                actual_time_s,
                args.time,
                args.panel_width,
                args.height,
            )
        )
    output_panels.extend(panels)
    output = args.output or _default_output(args.json, args.joint, args.movement, actual_time_s)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.hconcat(output_panels)):
        raise RuntimeError(f"Could not write output image: {output}")

    print(f"JSON frame:       {frame_idx}")
    print(f"JSON time:        {actual_time_s:.3f}s")
    print(f"Joint/movement:   {args.joint} {args.movement}")
    print(f"Side:             {result.side}")
    print(f"ROM angle:        {angle:+.2f} deg")
    print(f"Saved:            {output}")


if __name__ == "__main__":
    main()
