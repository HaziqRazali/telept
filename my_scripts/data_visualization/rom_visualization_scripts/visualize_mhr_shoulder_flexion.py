#!/usr/bin/env python3
"""Render one MHR pose with a shoulder ROM measurement overlay.

The output is a single PNG containing:

    [original video frame] [MHR body-aligned front] [MHR sagittal view]

For flexion, the measurement is made in the sagittal plane.  For abduction,
it is made in the coronal plane.  In both cases the shoulder-to-elbow (humerus)
vector is projected into the relevant body plane and measured from the
subject-centered trunk-down vector.  For external/internal rotation, the
forearm is projected into the plane perpendicular to the humerus and measured
from the body-forward direction; the third panel is a view down the humerus
axis so the rotation angle is visible.

For a SAM-3D-Body rendered video, which contains only frames with valid MHR
detections, run:

    /home/haziq/anaconda3/envs/mhr_new/bin/python \
        /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/visualize_mhr_shoulder_flexion.py \
        /data/haziq/mocap/data/brett/train/s01/sam3d/cam0/adhesive_capsulitis_mhr_outputs.npz \
        --time 7.374 --movement flexion

For shoulder rotation, for example:

    /home/haziq/anaconda3/envs/mhr_new/bin/python \
        /data/haziq/telept/my_scripts/data_visualization/rom_visualization_scripts/visualize_mhr_shoulder_flexion.py \
        /data/haziq/mocap/data/brett/train/s01/sam3d/cam0/adhesive_capsulitis_mhr_outputs.npz \
        --time 15.982 --movement external_rotation --side right

``--time`` defaults to the valid-frame/rendered-video timeline.  Use
``--time-domain original`` when the timestamp comes from the original source
video instead.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Must be set before importing pyrender/OpenGL on a headless workstation.  The
# workstation's EGL device nodes are not available to every user, so use
# Mesa's surfaceless software path by default.  Users with a configured EGL
# device can override these variables from the shell.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
if "anaconda" in os.environ.get("__EGL_VENDOR_LIBRARY_DIRS", ""):
    os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = "/usr/share/glvnd/egl_vendor.d"

import cv2
import numpy as np
import pyrender
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_VISUALIZATION_DIR = SCRIPT_DIR.parent
for import_path in (DATA_VISUALIZATION_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from rom_visualization_scripts.mhr_shoulder import (  # noqa: E402
    MHR_BODY_EDGES,
    MHR_JOINTS,
    ShoulderROMResult,
    ShoulderRotationResult,
    build_mhr_body_frame,
    compute_shoulder_abduction,
    compute_shoulder_flexion,
    compute_shoulder_rotation,
    mhr_uparm_twist_deg,
    select_shoulder_side,
)


MESH_COLOR = np.array([174, 197, 226, 255], dtype=np.uint8)
SKELETON_COLOR = (105, 115, 130)
JOINT_COLOR = (55, 75, 95)
ACTUAL_ARM_COLOR = (40, 70, 225)       # BGR: red/orange
MEASURED_ARM_COLOR = (215, 105, 25)    # BGR: cyan/blue
REFERENCE_COLOR = (0, 205, 245)        # BGR: yellow
POSITIVE_AXIS_COLOR = (65, 170, 65)    # BGR: green
ARC_COLOR = (170, 40, 220)             # BGR: purple
TEXT_COLOR = (25, 30, 40)
WHITE = (255, 255, 255)


@dataclass
class ViewSpec:
    """2-D orthographic projection used both by pyrender and annotations."""

    x_axis: np.ndarray
    y_axis: np.ndarray
    z_axis: np.ndarray
    origin: np.ndarray
    x_center: float
    y_center: float
    xmag: float
    ymag: float
    width: int
    height: int

    def coordinates(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        relative = points - self.origin[None, :]
        return np.column_stack(
            (
                relative @ self.x_axis,
                relative @ self.y_axis,
                relative @ self.z_axis,
            )
        )

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project 3-D points to integer pixel coordinates (x, y)."""

        q = self.coordinates(points)
        px = (q[:, 0] - self.x_center + self.xmag / 2.0) / self.xmag
        py = (self.ymag / 2.0 - (q[:, 1] - self.y_center)) / self.ymag
        pixels = np.column_stack((px * (self.width - 1), py * (self.height - 1)))
        return np.rint(pixels).astype(np.int32)


def _read_meta(data) -> dict:
    if "meta" not in data:
        return {}
    meta = np.asarray(data["meta"])
    if meta.size == 0:
        return {}
    value = meta.reshape(-1)[0]
    if hasattr(value, "item"):
        value = value.item()
    return value if isinstance(value, dict) else {}


def _finite_rows(data) -> np.ndarray:
    vertices = np.asarray(data["vertices"])
    joints = np.asarray(data["pred_joint_coords"])
    valid = np.isfinite(vertices).all(axis=(1, 2))
    valid &= np.isfinite(joints).all(axis=(1, 2))
    return np.flatnonzero(valid)


def _resolve_frame(data, time_s: float, time_domain: str, frame_override: int | None):
    """Resolve a requested time to an NPZ row and original frame number."""

    frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
    meta = _read_meta(data)
    fps = float(meta.get("fps", 30.0))
    valid_rows = _finite_rows(data)
    if len(valid_rows) == 0:
        raise RuntimeError("The NPZ contains no finite mesh/joint rows.")

    if frame_override is not None:
        requested_original = int(frame_override)
        distances = np.abs(frame_indices[valid_rows] - requested_original)
        valid_position = int(np.argmin(distances))
        row = int(valid_rows[valid_position])
        requested_domain_time = valid_position / fps
    elif time_domain == "rendered":
        # The rendered MP4 is built from finite rows only, so its frame 0 is
        # valid_rows[0], not NPZ row 0.
        valid_position = int(round(float(time_s) * fps))
        if valid_position < 0 or valid_position >= len(valid_rows):
            raise IndexError(
                f"Rendered time {time_s:.3f}s maps to valid frame {valid_position}, "
                f"but the NPZ has {len(valid_rows)} valid frames."
            )
        row = int(valid_rows[valid_position])
        requested_domain_time = valid_position / fps
    elif time_domain == "original":
        requested_original = int(round(float(time_s) * fps))
        distances = np.abs(frame_indices[valid_rows] - requested_original)
        valid_position = int(np.argmin(distances))
        row = int(valid_rows[valid_position])
        requested_domain_time = requested_original / fps
    else:
        raise ValueError(f"Unknown time domain: {time_domain}")

    original_frame = int(frame_indices[row])
    rendered_position = int(np.flatnonzero(valid_rows == row)[0])
    return {
        "row": row,
        "fps": fps,
        "original_frame": original_frame,
        "original_time": original_frame / fps,
        "rendered_frame": rendered_position,
        "rendered_time": rendered_position / fps,
        "requested_domain_time": requested_domain_time,
        "valid_rows": valid_rows,
    }


def _load_mhr_faces(mhr_root: Path) -> np.ndarray:
    """Load the LOD1 topology used by the stored 18,439-vertex MHR mesh."""

    if str(mhr_root) not in sys.path:
        sys.path.insert(0, str(mhr_root))
    try:
        import torch
        from mhr.mhr import MHR
    except ImportError as exc:
        raise RuntimeError(
            "MHR Python package is unavailable. Run this script in the mhr_new "
            "environment or pass an environment containing /home/haziq/MHR."
        ) from exc

    model = MHR.from_files(device=torch.device("cpu"), lod=1)
    faces = np.asarray(model.character.mesh.faces, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"Unexpected MHR face array shape: {faces.shape}")
    return faces


def _make_view_spec(
    vertices: np.ndarray,
    joints: np.ndarray,
    result: ShoulderROMResult,
    width: int,
    height: int,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> ViewSpec:
    origin = joints[MHR_JOINTS["root"]].astype(np.float64)
    provisional = ViewSpec(
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        origin=origin,
        x_center=0.0,
        y_center=0.0,
        xmag=1.0,
        ymag=1.0,
        width=width,
        height=height,
    )
    bounds_points = np.vstack(
        (
            vertices,
            joints,
            result.reference_end[None, :],
            result.measurement_end[None, :],
        )
    )
    q = provisional.coordinates(bounds_points)
    xmin, ymin = q[:, 0].min(), q[:, 1].min()
    xmax, ymax = q[:, 0].max(), q[:, 1].max()
    raw_x = max(float(xmax - xmin), 0.05)
    raw_y = max(float(ymax - ymin), 0.05)

    # Add a little whitespace around the mesh and then preserve the panel's
    # aspect ratio.  xmag/ymag are the same extents used by the camera.
    raw_x *= 1.16
    raw_y *= 1.16
    aspect = width / float(height)
    ymag = max(raw_y, raw_x / aspect)
    xmag = ymag * aspect
    return ViewSpec(
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        origin=origin,
        x_center=float((xmin + xmax) / 2.0),
        y_center=float((ymin + ymax) / 2.0),
        xmag=xmag,
        ymag=ymag,
        width=width,
        height=height,
    )


def _render_mesh(vertices: np.ndarray, faces: np.ndarray, spec: ViewSpec) -> np.ndarray:
    q_vertices = spec.coordinates(vertices)
    q_vertices[:, 0] -= spec.x_center
    q_vertices[:, 1] -= spec.y_center

    # A body-aligned basis can be a reflection of the stored MHR basis.  Flip
    # winding in that case so lighting remains consistent in the render.
    basis = np.column_stack((spec.x_axis, spec.y_axis, spec.z_axis))
    render_faces = faces[:, ::-1] if np.linalg.det(basis) < 0.0 else faces
    vertex_colors = np.tile(MESH_COLOR[None, :], (len(q_vertices), 1))
    mesh_trimesh = trimesh.Trimesh(
        vertices=q_vertices,
        faces=render_faces,
        vertex_colors=vertex_colors,
        process=False,
    )

    scene = pyrender.Scene(
        bg_color=[0.97, 0.97, 0.97, 1.0],
        ambient_light=[0.35, 0.35, 0.35],
    )
    scene.add(pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True))

    camera = pyrender.OrthographicCamera(
        # pyrender's orthographic magnifications are half-extents (+/-),
        # whereas ViewSpec stores the full pixel-visible extents used by the
        # annotation projection below.
        xmag=spec.xmag / 2.0,
        ymag=spec.ymag / 2.0,
        znear=0.001,
        zfar=100.0,
    )
    qz = q_vertices[:, 2]
    depth_span = max(float(qz.max() - qz.min()), 0.1)
    camera_pose = np.eye(4, dtype=np.float64)
    camera_pose[:3, 3] = [0.0, 0.0, float(qz.max() + depth_span + 2.0)]
    scene.add(camera, pose=camera_pose)

    # A directional light gives stable shading; point lights add gentle form
    # contrast without requiring a particular camera distance.
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
        pose=np.eye(4),
    )
    for position, intensity in [
        ([2.0, 3.0, 4.0], 18.0),
        ([-2.0, 1.0, 2.0], 8.0),
    ]:
        light_pose = np.eye(4, dtype=np.float64)
        light_pose[:3, 3] = position
        scene.add(
            pyrender.PointLight(color=np.ones(3), intensity=intensity),
            pose=light_pose,
        )

    renderer = pyrender.OffscreenRenderer(spec.width, spec.height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    finally:
        renderer.delete()
    return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)


def _rotation_limb_segments(result) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Return the proximal/distal segments used by a rotation result."""

    if hasattr(result, "shoulder"):
        return (
            (result.shoulder, result.elbow),
            (result.elbow, result.wrist),
        )
    return (
        (result.hip, result.knee),
        (result.knee, result.foot),
    )


def _select_rotation_limb_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    result,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract the selected arm/leg surface for the axis-view inset."""

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        return None

    valid_faces = (faces >= 0).all(axis=1) & (faces < len(vertices)).all(axis=1)
    faces = faces[valid_faces]
    if len(faces) == 0:
        return None

    segments = _rotation_limb_segments(result)
    lengths = [float(np.linalg.norm(b - a)) for a, b in segments]
    scale = max(min(lengths), 1e-8)

    # Try a tight crop first, then widen it slightly if the topology leaves
    # too few complete triangles around the limb.
    selected_faces = None
    for fraction in (0.28, 0.40, 0.55):
        radius = fraction * scale
        vertex_distance_sq = np.full(len(vertices), np.inf, dtype=np.float64)
        for start, end in segments:
            direction = end - start
            denominator = float(np.dot(direction, direction))
            if denominator < 1e-12:
                distance_sq = np.sum((vertices - start[None, :]) ** 2, axis=1)
            else:
                t = np.clip(
                    ((vertices - start[None, :]) @ direction) / denominator,
                    0.0,
                    1.0,
                )
                closest = start[None, :] + t[:, None] * direction[None, :]
                distance_sq = np.sum((vertices - closest) ** 2, axis=1)
            vertex_distance_sq = np.minimum(vertex_distance_sq, distance_sq)

        vertex_mask = vertex_distance_sq <= radius * radius
        complete = vertex_mask[faces].all(axis=1)
        candidate = faces[complete]
        if len(candidate) >= 20:
            selected_faces = candidate
            break

    if selected_faces is None:
        # A small one-ring boundary is preferable to an empty inset when the
        # mesh topology cuts through the geometric crop.
        candidate = faces[vertex_mask[faces].any(axis=1)]
        if len(candidate) == 0:
            return None
        selected_faces = candidate

    used_vertices = np.unique(selected_faces)
    remap = np.full(len(vertices), -1, dtype=np.int32)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=np.int32)
    return vertices[used_vertices], remap[selected_faces]


def _make_rotation_mesh_spec(
    vertices: np.ndarray,
    result,
    width: int,
    height: int,
) -> ViewSpec:
    """Build a view perpendicular to the humerus/thigh rotation axis."""

    origin = result.elbow if hasattr(result, "shoulder") else result.knee
    x_axis = np.asarray(result.reference_unit, dtype=np.float64)
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
    y_axis = np.asarray(result.positive_axis, dtype=np.float64)
    y_axis -= x_axis * float(np.dot(y_axis, x_axis))
    y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
    # Construct the normal from the displayed in-plane axes so the mesh
    # renderer sees a right-handed basis while preserving the diagram's axes.
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-12)

    provisional = ViewSpec(
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        origin=np.asarray(origin, dtype=np.float64),
        x_center=0.0,
        y_center=0.0,
        xmag=1.0,
        ymag=1.0,
        width=width,
        height=height,
    )
    bounds = np.vstack(
        (
            np.asarray(vertices, dtype=np.float64),
            np.asarray(origin, dtype=np.float64)[None, :],
            result.reference_end[None, :],
            result.measurement_end[None, :],
            (
                origin
                + np.asarray(result.positive_axis, dtype=np.float64)
                * float(result.measurement_length)
            )[None, :],
        )
    )
    q = provisional.coordinates(bounds)
    xmin, ymin = q[:, 0].min(), q[:, 1].min()
    xmax, ymax = q[:, 0].max(), q[:, 1].max()
    raw_x = max(float(xmax - xmin), 0.05) * 1.14
    raw_y = max(float(ymax - ymin), 0.05) * 1.14
    aspect = width / float(height)
    ymag = max(raw_y, raw_x / aspect)
    xmag = ymag * aspect
    return ViewSpec(
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        origin=np.asarray(origin, dtype=np.float64),
        x_center=float((xmin + xmax) / 2.0),
        y_center=float((ymin + ymax) / 2.0),
        xmag=xmag,
        ymag=ymag,
        width=width,
        height=height,
    )


def _draw_rotation_mesh_inset(
    image: np.ndarray,
    vertices: np.ndarray | None,
    faces: np.ndarray | None,
    result,
) -> None:
    """Place a selected-limb mesh viewed along its rotation axis in a panel."""

    if vertices is None or faces is None:
        return
    selected = _select_rotation_limb_mesh(vertices, faces, result)
    if selected is None:
        return
    limb_vertices, limb_faces = selected

    height, width = image.shape[:2]
    inset_width = max(180, int(width * 0.34))
    inset_height = max(300, int(height * 0.58))
    inset_spec = _make_rotation_mesh_spec(
        limb_vertices,
        result,
        inset_width,
        inset_height,
    )
    inset = _render_mesh(limb_vertices, limb_faces, inset_spec)

    # Show the actual distal segment over the surface so its orientation can
    # be related directly to the cyan vector in the circular diagram.
    for start, end in _rotation_limb_segments(result):
        p_start = tuple(inset_spec.project(start[None, :])[0])
        p_end = tuple(inset_spec.project(end[None, :])[0])
        cv2.line(inset, p_start, p_end, ACTUAL_ARM_COLOR, 3, cv2.LINE_AA)
    joint = result.elbow if hasattr(result, "shoulder") else result.knee
    p_joint = tuple(inset_spec.project(joint[None, :])[0])
    cv2.circle(inset, p_joint, 7, WHITE, -1, cv2.LINE_AA)
    cv2.circle(inset, p_joint, 7, TEXT_COLOR, 2, cv2.LINE_AA)

    x0 = 12
    y0 = 88
    x1 = min(x0 + inset_width, width)
    y1 = min(y0 + inset_height, height)
    image[y0:y1, x0:x1] = inset[: y1 - y0, : x1 - x0]
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (218, 222, 228), 2, cv2.LINE_AA)


def _clip_point(point: np.ndarray, width: int, height: int) -> tuple[int, int]:
    x = int(np.clip(point[0], 0, width - 1))
    y = int(np.clip(point[1], 0, height - 1))
    return x, y


def _put_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    color=TEXT_COLOR,
    scale: float = 0.62,
    thickness: int = 2,
):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _draw_dashed_line(
    image: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color,
    thickness: int = 4,
    dash: int = 16,
    gap: int = 10,
):
    start_f = np.asarray(start, dtype=np.float64)
    end_f = np.asarray(end, dtype=np.float64)
    vector = end_f - start_f
    length = float(np.linalg.norm(vector))
    if length < 1.0:
        return
    direction = vector / length
    distance = 0.0
    while distance < length:
        a = start_f + direction * distance
        b = start_f + direction * min(distance + dash, length)
        cv2.line(
            image,
            tuple(np.rint(a).astype(int)),
            tuple(np.rint(b).astype(int)),
            color,
            thickness,
            cv2.LINE_AA,
        )
        distance += dash + gap


def _draw_arrow(image: np.ndarray, start, end, color, thickness: int = 5):
    cv2.arrowedLine(
        image,
        tuple(start),
        tuple(end),
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.12,
    )


def _draw_arc(image: np.ndarray, spec: ViewSpec, result: ShoulderROMResult):
    """Draw the signed angle arc in the selected ROM plane."""

    down = result.body_frame.down
    positive = result.positive_axis
    up = result.body_frame.up
    arm = result.measurement_unit

    a0 = float(np.arctan2(np.dot(down, up), np.dot(down, positive)))
    a1 = float(np.arctan2(np.dot(arm, up), np.dot(arm, positive)))
    delta = (a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi
    radius = max(0.08, 0.45 * result.arm_length)
    samples = max(20, int(abs(delta) * 30.0))
    angles = np.linspace(a0, a0 + delta, samples)
    arc_world = np.vstack(
        [
            result.shoulder + radius * (np.cos(angle) * positive + np.sin(angle) * up)
            for angle in angles
        ]
    )
    arc_pixels = spec.project(arc_world)
    cv2.polylines(image, [arc_pixels.reshape(-1, 1, 2)], False, ARC_COLOR, 5, cv2.LINE_AA)

    midpoint = a0 + delta / 2.0
    label_world = result.shoulder + (radius + 0.06) * (
        np.cos(midpoint) * positive + np.sin(midpoint) * up
    )
    label = _clip_point(spec.project(label_world[None, :])[0], spec.width, spec.height)
    _put_text(image, f"{result.angle_deg:+.1f} deg", label, ARC_COLOR, 0.72, 2)


def _draw_rotation_arc(
    image: np.ndarray,
    spec: ViewSpec,
    result: ShoulderRotationResult,
):
    """Draw the signed forearm rotation arc around the humerus axis."""

    theta = float(np.radians(result.angle_deg))
    radius = max(0.06, 0.70 * result.measurement_length)
    samples = max(20, int(abs(theta) * 30.0))
    angles = np.linspace(0.0, theta, samples)
    positive_transverse = np.cross(result.rotation_axis, result.reference_unit)
    arc_world = np.vstack(
        [
            result.elbow
            + radius
            * (
                np.cos(angle) * result.reference_unit
                + np.sin(angle) * positive_transverse
            )
            for angle in angles
        ]
    )
    arc_pixels = spec.project(arc_world)
    cv2.polylines(image, [arc_pixels.reshape(-1, 1, 2)], False, ARC_COLOR, 5, cv2.LINE_AA)


def _draw_rotation_measurement(
    image: np.ndarray,
    spec: ViewSpec,
    joints: np.ndarray,
    result: ShoulderRotationResult,
    mhr_twist_deg: float | None,
    show_result_text: bool = True,
):
    """Overlay the humerus, forearm/reference vectors, and rotation labels."""

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
        MHR_JOINTS["root"],
        MHR_JOINTS["c_spine3"],
        MHR_JOINTS["c_neck"],
        MHR_JOINTS["r_shoulder"],
        MHR_JOINTS["r_elbow"],
        MHR_JOINTS["r_wrist"],
        MHR_JOINTS["l_shoulder"],
        MHR_JOINTS["l_elbow"],
        MHR_JOINTS["l_wrist"],
    ]
    for index in key_joint_indices:
        cv2.circle(image, tuple(projected_joints[index]), 5, JOINT_COLOR, -1, cv2.LINE_AA)

    p_shoulder = tuple(spec.project(result.shoulder[None, :])[0])
    p_elbow = tuple(spec.project(result.elbow[None, :])[0])
    p_wrist = tuple(spec.project(result.wrist[None, :])[0])
    p_measurement = tuple(spec.project(result.measurement_end[None, :])[0])
    p_reference = tuple(spec.project(result.reference_end[None, :])[0])
    positive_end = result.elbow + result.positive_axis * result.measurement_length
    p_positive = tuple(spec.project(positive_end[None, :])[0])

    # Red follows the reconstructed humerus/forearm.  Cyan is the forearm
    # direction after removing the component parallel to the humerus, which is
    # the vector used by the axial-rotation calculation.
    _draw_arrow(image, p_shoulder, p_elbow, ACTUAL_ARM_COLOR, 5)
    _draw_arrow(image, p_elbow, p_wrist, ACTUAL_ARM_COLOR, 5)
    _draw_arrow(image, p_elbow, p_measurement, MEASURED_ARM_COLOR, 5)
    _draw_dashed_line(image, p_elbow, p_reference, REFERENCE_COLOR, 4)
    _draw_dashed_line(image, p_elbow, p_positive, POSITIVE_AXIS_COLOR, 3, dash=10, gap=8)
    cv2.circle(image, p_shoulder, 9, WHITE, -1, cv2.LINE_AA)
    cv2.circle(image, p_shoulder, 9, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.circle(image, p_elbow, 8, WHITE, -1, cv2.LINE_AA)
    cv2.circle(image, p_elbow, 8, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.circle(image, p_wrist, 7, ACTUAL_ARM_COLOR, -1, cv2.LINE_AA)

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
    _put_text(image, "red = actual humerus + forearm", (22, spec.height - 84), ACTUAL_ARM_COLOR, 0.52, 1)
    _put_text(image, "cyan = forearm transverse vector used", (22, spec.height - 62), MEASURED_ARM_COLOR, 0.52, 1)
    _put_text(image, "yellow = body-forward reference", (22, spec.height - 40), (0, 150, 190), 0.52, 1)
    _put_text(image, "green = external-rotation positive direction", (22, spec.height - 18), POSITIVE_AXIS_COLOR, 0.50, 1)


def _draw_rotation_plane_diagram(
    width: int,
    height: int,
    result: ShoulderRotationResult,
    mhr_twist_deg: float | None,
    show_result_text: bool = True,
    mesh_vertices: np.ndarray | None = None,
    mesh_faces: np.ndarray | None = None,
) -> np.ndarray:
    """Create a clean transverse-plane diagram viewed along the humerus."""

    image = np.full((height, width, 3), (249, 249, 249), dtype=np.uint8)
    has_mesh = mesh_vertices is not None and mesh_faces is not None
    center = np.array(
        [width * (0.68 if has_mesh else 0.52), height * 0.53],
        dtype=np.float64,
    )
    radius = min(width * (0.24 if has_mesh else 0.29), height * 0.28)

    def point(angle: float, length: float) -> tuple[int, int]:
        # The local positive transverse direction is drawn upward, so positive
        # external rotation appears counter-clockwise in this diagram.
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
    cv2.line(
        image,
        tuple(np.rint(center).astype(int)),
        tuple(positive_end),
        (205, 225, 205),
        1,
        cv2.LINE_AA,
    )
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
    _put_text(image, "yellow = body-forward / neutral reference", (22, height - 84), (0, 150, 190), 0.52, 1)
    _put_text(image, "cyan = forearm direction after perpendicular projection", (22, height - 62), MEASURED_ARM_COLOR, 0.50, 1)
    _put_text(image, "green = external-rotation positive direction", (22, height - 40), POSITIVE_AXIS_COLOR, 0.52, 1)
    _put_text(image, "magenta = measured axial angle", (22, height - 18), ARC_COLOR, 0.52, 1)
    _put_text(image, "neutral / body-forward", (ref_end[0] - 150, ref_end[1] - 24), REFERENCE_COLOR, 0.47, 1)
    _put_text(image, "external +", (positive_end[0] + 10, positive_end[1] - 8), POSITIVE_AXIS_COLOR, 0.50, 1)
    _draw_rotation_mesh_inset(image, mesh_vertices, mesh_faces, result)
    return image


def _draw_measurement(
    image: np.ndarray,
    spec: ViewSpec,
    joints: np.ndarray,
    result: ShoulderROMResult,
    title: str,
    draw_arc: bool = False,
    draw_positive_axis: bool = False,
    show_result_text: bool = True,
):
    """Overlay skeleton, humerus/reference vectors, labels, and optional arc."""

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
        MHR_JOINTS["root"],
        MHR_JOINTS["c_spine3"],
        MHR_JOINTS["c_neck"],
        MHR_JOINTS["r_shoulder"],
        MHR_JOINTS["r_elbow"],
        MHR_JOINTS["l_shoulder"],
        MHR_JOINTS["l_elbow"],
    ]
    for index in key_joint_indices:
        cv2.circle(image, tuple(projected_joints[index]), 5, JOINT_COLOR, -1, cv2.LINE_AA)

    p_shoulder = tuple(spec.project(result.shoulder[None, :])[0])
    p_elbow = tuple(spec.project(result.elbow[None, :])[0])
    p_measurement = tuple(spec.project(result.measurement_end[None, :])[0])
    p_reference = tuple(spec.project(result.reference_end[None, :])[0])

    # Red is the actual 3-D shoulder-to-elbow vector.  Cyan is the projected
    # humerus vector actually used in the ROM calculation.
    _draw_arrow(image, p_shoulder, p_elbow, ACTUAL_ARM_COLOR, 5)
    _draw_arrow(image, p_shoulder, p_measurement, MEASURED_ARM_COLOR, 5)
    _draw_dashed_line(image, p_shoulder, p_reference, REFERENCE_COLOR, 4)
    if draw_positive_axis:
        positive_end = result.shoulder + result.positive_axis * (0.72 * result.arm_length)
        p_positive = tuple(spec.project(positive_end[None, :])[0])
        _draw_dashed_line(image, p_shoulder, p_positive, POSITIVE_AXIS_COLOR, 3, dash=10, gap=8)
        _put_text(image, "outward +", (p_positive[0] + 8, p_positive[1]), POSITIVE_AXIS_COLOR, 0.46, 1)
    cv2.circle(image, p_shoulder, 9, WHITE, -1, cv2.LINE_AA)
    cv2.circle(image, p_shoulder, 9, TEXT_COLOR, 2, cv2.LINE_AA)
    cv2.circle(image, p_elbow, 7, ACTUAL_ARM_COLOR, -1, cv2.LINE_AA)

    if show_result_text:
        _put_text(
            image,
            f"{result.side.capitalize()} {result.measurement_name}: {result.angle_deg:+.1f} deg",
            (22, 34),
            ARC_COLOR,
            0.72,
            2,
        )
    _put_text(image, "red = actual humerus", (22, spec.height - 62), ACTUAL_ARM_COLOR, 0.52, 1)
    _put_text(image, f"cyan = {result.plane_name} vector used", (22, spec.height - 40), MEASURED_ARM_COLOR, 0.52, 1)
    _put_text(image, "yellow = trunk-down reference", (22, spec.height - 18), (0, 150, 190), 0.52, 1)
    if draw_positive_axis:
        _put_text(image, "green = positive/outward axis", (22, spec.height - 84), POSITIVE_AXIS_COLOR, 0.52, 1)

    if draw_arc:
        _draw_arc(image, spec, result)


def _letterbox(image: np.ndarray, width: int, height: int, background=WHITE) -> np.ndarray:
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    if image is None or image.size == 0:
        return canvas
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, int(round(image.shape[1] * scale))), max(1, int(round(image.shape[0] * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _load_video_frame(video_path: Path, frame_number: int) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, frame = capture.read()
        return frame if ok else None
    finally:
        capture.release()


def _default_output(npz_path: Path, time_domain: str, time_s: float, movement: str) -> Path:
    stamp = f"{time_s:.3f}".replace(".", "p")
    return Path(__file__).resolve().parent / "results" / (
        f"{npz_path.stem}_shoulder_{movement}_{time_domain}_t{stamp}.png"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path, help="SAM-3D-Body MHR output NPZ")
    parser.add_argument(
        "--movement",
        "--motion",
        dest="movement",
        choices=("flexion", "abduction", "external_rotation", "internal_rotation"),
        default="flexion",
        help="Shoulder movement to measure (default: flexion)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=7.374,
        help="Timestamp in seconds (default: 7.374)",
    )
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
        default="auto",
        help="Shoulder to annotate (default: auto selects largest absolute angle)",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Original video; inferred from NPZ meta when omitted",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path",
    )
    parser.add_argument(
        "--mhr-root",
        type=Path,
        default=Path("/home/haziq/MHR"),
        help="MHR repository root containing the mhr package",
    )
    parser.add_argument("--height", type=int, default=900, help="Panel height in pixels")
    parser.add_argument("--panel-width", type=int, default=800, help="Mesh panel width")
    parser.add_argument(
        "--no-source",
        action="store_true",
        help="Omit the original-video panel",
    )
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
    selected_side = select_shoulder_side(joints, args.side, args.movement)
    if args.movement == "abduction":
        result = compute_shoulder_abduction(joints, selected_side, body_frame)
        primary_title = "MHR body-aligned CORONAL / FRONT"
        secondary_title = "MHR body-aligned SAGITTAL"
        primary_arc = True
        secondary_arc = False
        primary_positive_axis = True
        rotation_result = None
        mhr_twist_deg = None
    elif args.movement in {"external_rotation", "internal_rotation"}:
        rotation_name = "external" if args.movement == "external_rotation" else "internal"
        rotation_result = compute_shoulder_rotation(
            joints,
            selected_side,
            rotation_name,
            body_frame,
        )
        result = rotation_result
        mhr_twist_deg = mhr_uparm_twist_deg(data["body_pose_params"][row], selected_side)
        primary_title = "MHR body-aligned FRONT"
        secondary_title = "MHR HUMERUS-AXIS ROTATION"
        primary_arc = False
        secondary_arc = False
        primary_positive_axis = False
    else:
        result = compute_shoulder_flexion(joints, selected_side, body_frame)
        primary_title = "MHR body-aligned FRONT"
        secondary_title = "MHR body-aligned SAGITTAL"
        primary_arc = False
        secondary_arc = True
        primary_positive_axis = False
        rotation_result = None
        mhr_twist_deg = None
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
    print(f"Shoulder side:   {result.side}")
    if rotation_result is not None:
        print(f"ROM angle:       {rotation_result.requested_angle_deg:+.2f} deg ({rotation_result.angle_deg:+.2f} signed; external +)")
        print(f"MHR uparm twist: {mhr_twist_deg:+.2f} deg (model-local)")
    else:
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
    if rotation_result is None:
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
    if rotation_result is not None:
        rotation_spec = _make_rotation_mesh_spec(
            vertices,
            rotation_result,
            args.panel_width,
            args.height,
        )
        rotation_panel = _render_mesh(vertices, faces, rotation_spec)
        _draw_rotation_measurement(
            front_panel,
            front_spec,
            joints,
            rotation_result,
            mhr_twist_deg,
            show_result_text=True,
        )
        _draw_rotation_measurement(
            rotation_panel,
            rotation_spec,
            joints,
            rotation_result,
            mhr_twist_deg,
            show_result_text=False,
        )
    else:
        _draw_measurement(
            front_panel,
            front_spec,
            joints,
            result,
            primary_title,
            draw_arc=primary_arc,
            draw_positive_axis=primary_positive_axis,
        )
        sagittal_panel = _render_mesh(vertices, faces, sagittal_spec)
        _draw_measurement(
            sagittal_panel,
            sagittal_spec,
            joints,
            result,
            secondary_title,
            draw_arc=secondary_arc,
            draw_positive_axis=False,
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

    panels.append(front_panel)
    panels.append(rotation_panel if rotation_result is not None else sagittal_panel)
    output = args.output or _default_output(args.npz, args.time_domain, args.time, args.movement)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.hconcat(panels)):
        raise RuntimeError(f"Could not write output image: {output}")
    print(f"Saved:            {output}")


if __name__ == "__main__":
    main()
