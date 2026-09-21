#!/usr/bin/env python3
"""Visualize the Momentum Human Rig at its default rest pose.

The script uses the Python MHR model directly and never invokes Unreal. The
``original`` and ``reskinned`` modes intentionally use the same MHR vertices
and faces; the reskinned mode changes only the render appearance with the
Python port of ``M_SkinFresnel`` in
``rom_visualization_scripts/mhr_skin.py``.

Examples
--------

Render the original rest pose:

    /home/haziq/anaconda3/envs/mhr_new/bin/python \
        /data/haziq/telept/my_scripts/data_visualization/visualize_mhr_mesh.py \
        --skin original --view front

Render the Python reskin:

    /home/haziq/anaconda3/envs/mhr_new/bin/python \
        /data/haziq/telept/my_scripts/data_visualization/visualize_mhr_mesh.py \
        --skin reskinned --view front

Render the same high-resolution LOD as the supplied Unreal ``skin.uasset``:

    /home/haziq/anaconda3/envs/mhr_new/bin/python \
        /data/haziq/telept/my_scripts/data_visualization/visualize_mhr_mesh.py \
        --skin reskinned --mhr-lod 0 --view front

Render both and run the appearance-only geometry validation:

    /home/haziq/anaconda3/envs/mhr_new/bin/python \
        /data/haziq/telept/my_scripts/data_visualization/visualize_mhr_mesh.py \
        --skin both --view both

Validation runs by default; pass ``--no-validate`` only when it is not needed.

The pose-following Unreal-style joint overlay is enabled by default. Disable
it with ``--pose-overlay none`` or use ``--pose-overlay legacy`` for the older
neutral Python-renderer colours.

An exported OBJ/PLY/glTF rest mesh can optionally be compared vertex-by-vertex
with ``--reference-mesh``.  Unreal ``.uasset`` files are not directly readable
by the Python mesh libraries.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Set these before importing pyrender/OpenGL on a headless workstation.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
if "anaconda" in os.environ.get("__EGL_VENDOR_LIBRARY_DIRS", ""):
    os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = "/usr/share/glvnd/egl_vendor.d"

import cv2
import numpy as np
import torch
import trimesh


SCRIPT_DIR = Path(__file__).resolve().parent
ROM_DIR = SCRIPT_DIR / "rom_visualization_scripts"
for import_path in (SCRIPT_DIR, ROM_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from rom_visualization_scripts.mhr_json import load_mhr_model  # noqa: E402
from rom_visualization_scripts.mhr_shoulder import (  # noqa: E402
    MHR_JOINTS,
    build_mhr_body_frame,
)
from rom_visualization_scripts.mhr_pose_overlay import (  # noqa: E402
    POSE_OVERLAY_CHOICES,
    draw_pose_overlay,
)
from rom_visualization_scripts.mhr_skin import (  # noqa: E402
    SKIN_CHOICES,
    compare_same_geometry,
)
from rom_visualization_scripts.visualize_mhr_shoulder_flexion import (  # noqa: E402
    TEXT_COLOR,
    ViewSpec,
    _render_mesh,
)


VIEW_CHOICES = ("front", "sagittal", "both")
SKIN_DISPLAY_CHOICES = ("original", "reskinned", "both")


def _reconstruct_default_rest_pose(model) -> tuple[np.ndarray, np.ndarray]:
    """Run MHR with zero identity, pose, scale, and expression parameters."""

    identity = torch.zeros(1, 45, dtype=torch.float32)
    model_parameters = torch.zeros(1, 204, dtype=torch.float32)
    expression = torch.zeros(1, 72, dtype=torch.float32)
    with torch.no_grad():
        vertices_cm, skeleton_state = model(identity, model_parameters, expression)

    vertices_m = (vertices_cm[0] / 100.0).cpu().numpy().astype(np.float64)
    joints_m = (skeleton_state[0, :, :3] / 100.0).cpu().numpy().astype(np.float64)
    return vertices_m, joints_m


def _make_rest_view_spec(
    vertices: np.ndarray,
    joints: np.ndarray,
    width: int,
    height: int,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    z_axis: np.ndarray,
) -> ViewSpec:
    """Create the same orthographic view convention used by the ROM figures."""

    origin = joints[MHR_JOINTS["root"]].astype(np.float64)
    provisional = ViewSpec(
        x_axis=np.asarray(x_axis, dtype=np.float64),
        y_axis=np.asarray(y_axis, dtype=np.float64),
        z_axis=np.asarray(z_axis, dtype=np.float64),
        origin=origin,
        x_center=0.0,
        y_center=0.0,
        xmag=1.0,
        ymag=1.0,
        width=width,
        height=height,
    )
    bounds_points = np.vstack((vertices, joints))
    q = provisional.coordinates(bounds_points)
    xmin, ymin = q[:, 0].min(), q[:, 1].min()
    xmax, ymax = q[:, 0].max(), q[:, 1].max()

    raw_x = max(float(xmax - xmin), 0.05) * 1.16
    raw_y = max(float(ymax - ymin), 0.05) * 1.16
    aspect = width / float(height)
    ymag = max(raw_y, raw_x / aspect)
    xmag = ymag * aspect
    return ViewSpec(
        x_axis=provisional.x_axis,
        y_axis=provisional.y_axis,
        z_axis=provisional.z_axis,
        origin=origin,
        x_center=float((xmin + xmax) / 2.0),
        y_center=float((ymin + ymax) / 2.0),
        xmag=xmag,
        ymag=ymag,
        width=width,
        height=height,
    )


def _label_panel(panel: np.ndarray, title: str) -> np.ndarray:
    """Add a small readable title without changing the mesh render."""

    output = panel.copy()
    cv2.putText(
        output,
        title,
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    return output


def _load_reference_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an exported reference mesh using trimesh."""

    if path.suffix.lower() == ".uasset":
        raise ValueError(
            "Unreal .uasset files are not directly readable; export the asset "
            "to OBJ, PLY, or glTF first."
        )
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [geometry for geometry in loaded.geometry.values()]
        if not geometries:
            raise ValueError(f"Reference scene contains no geometry: {path}")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported reference mesh type in {path}: {type(loaded)!r}")
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    faces = np.asarray(loaded.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Reference vertices have unexpected shape: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Reference faces have unexpected shape: {faces.shape}")
    return vertices, faces


def _center_scale_align(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align units and translation while preserving vertex correspondence."""

    reference_min = reference.min(axis=0)
    reference_max = reference.max(axis=0)
    target_min = target.min(axis=0)
    target_max = target.max(axis=0)
    reference_extent = reference_max - reference_min
    target_extent = target_max - target_min
    valid = reference_extent > 1e-12
    if not np.any(valid):
        raise ValueError("Reference mesh has zero extent.")
    scale = float(np.median(target_extent[valid] / reference_extent[valid]))
    reference_center = (reference_min + reference_max) / 2.0
    target_center = (target_min + target_max) / 2.0
    return (reference - reference_center[None, :]) * scale + target_center[None, :]


def _face_signatures(faces: np.ndarray) -> np.ndarray:
    """Canonicalize triangle vertex order for winding-insensitive comparison."""

    canonical = np.sort(np.asarray(faces, dtype=np.int64), axis=1)
    order = np.lexsort((canonical[:, 2], canonical[:, 1], canonical[:, 0]))
    return canonical[order]


def compare_reference_mesh(
    reference_path: Path,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
) -> dict[str, float | int | bool | str]:
    """Compare an exported reference mesh with the MHR rest mesh.

    This is a direct vertex-order comparison after centering and unit-scale
    normalization.  It deliberately does not silently solve a permutation or
    rotation, because that would no longer be a vertex-to-vertex validation.
    """

    reference_vertices, reference_faces = _load_reference_mesh(reference_path)
    report: dict[str, float | int | bool | str] = {
        "reference_mesh": str(reference_path),
        "reference_vertex_count": int(len(reference_vertices)),
        "mhr_vertex_count": int(len(target_vertices)),
        "reference_face_count": int(len(reference_faces)),
        "mhr_face_count": int(len(target_faces)),
    }
    if reference_vertices.shape != target_vertices.shape:
        report.update(
            {
                "same_vertex_shape": False,
                "same_face_shape": bool(reference_faces.shape == target_faces.shape),
                "faces_equal_ignoring_winding": False,
                "max_abs_vertex_delta_m": float("inf"),
                "mean_abs_vertex_delta_m": float("inf"),
                "passed": False,
            }
        )
        return report

    aligned_reference = _center_scale_align(reference_vertices, target_vertices)
    delta = np.abs(aligned_reference - target_vertices)
    faces_equal = bool(
        reference_faces.shape == target_faces.shape
        and np.array_equal(_face_signatures(reference_faces), _face_signatures(target_faces))
    )
    report.update(
        {
            "same_vertex_shape": True,
            "same_face_shape": bool(reference_faces.shape == target_faces.shape),
            "faces_equal_ignoring_winding": faces_equal,
            "max_abs_vertex_delta_m": float(delta.max(initial=0.0)),
            "mean_abs_vertex_delta_m": float(delta.mean()),
            "passed": bool(
                faces_equal
                and float(delta.max(initial=0.0)) <= 1e-5
            ),
        }
    )
    return report


def _print_report(name: str, report: dict[str, float | int | bool | str]) -> None:
    print(f"\n{name}")
    for key, value in report.items():
        print(f"  {key}: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skin",
        choices=SKIN_DISPLAY_CHOICES,
        default="original",
        help="Render original, reskinned, or both side by side (default: original)",
    )
    parser.add_argument(
        "--view",
        choices=VIEW_CHOICES,
        default="front",
        help="Front, sagittal, or both views (default: front)",
    )
    parser.add_argument(
        "--pose-overlay",
        choices=POSE_OVERLAY_CHOICES,
        default="unreal",
        help="Pose-following joint/bone overlay style (default: unreal)",
    )
    parser.add_argument(
        "--mhr-root",
        type=Path,
        default=Path("/home/haziq/MHR"),
        help="MHR repository root containing the Python package and assets",
    )
    parser.add_argument(
        "--mhr-lod",
        type=int,
        choices=tuple(range(7)),
        default=1,
        help="MHR mesh LOD (LOD1 matches the ROM renderer; LOD0 matches skin.uasset)",
    )
    parser.add_argument("--height", type=int, default=900, help="Panel height in pixels")
    parser.add_argument("--panel-width", type=int, default=800, help="Panel width in pixels")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path")
    parser.add_argument(
        "--reference-mesh",
        type=Path,
        default=None,
        help="Optional exported OBJ/PLY/glTF rest mesh for vertex comparison",
    )
    parser.add_argument(
        "--validation-json",
        type=Path,
        default=None,
        help="Optional path for a JSON validation report",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip the internal appearance-only geometry validation",
    )
    args = parser.parse_args()

    mhr_root = args.mhr_root.expanduser().resolve()
    if str(mhr_root) not in sys.path:
        sys.path.insert(0, str(mhr_root))
    model, faces = load_mhr_model(
        mhr_root,
        lod=args.mhr_lod,
        # The default rest pose has zero pose-corrective activation, so there
        # is no need to load the much larger corrective-blendshape file.
        wants_pose_correctives=False,
    )
    vertices, joints = _reconstruct_default_rest_pose(model)
    if not np.isfinite(vertices).all() or not np.isfinite(joints).all():
        raise RuntimeError("Default MHR rest pose contains NaN/Inf values.")
    if faces.max(initial=-1) >= len(vertices):
        raise RuntimeError(
            f"MHR topology references vertex {faces.max()}, but mesh has {len(vertices)} vertices."
        )

    body_frame = build_mhr_body_frame(joints)
    view_axes = {
        "front": (-body_frame.right, body_frame.up, body_frame.forward),
        "sagittal": (body_frame.forward, body_frame.up, body_frame.right),
    }
    views = ("front", "sagittal") if args.view == "both" else (args.view,)
    skins = ("original", "reskinned") if args.skin == "both" else (args.skin,)

    panels: list[np.ndarray] = []
    for skin in skins:
        for view in views:
            spec = _make_rest_view_spec(
                vertices,
                joints,
                args.panel_width,
                args.height,
                *view_axes[view],
            )
            panel = _render_mesh(vertices, faces, spec, skin=skin)
            draw_pose_overlay(
                panel,
                spec,
                joints,
                style=args.pose_overlay,
                line_thickness=2,
                joint_radius=5,
            )
            panels.append(_label_panel(panel, f"MHR REST - {skin.upper()} - {view.upper()}"))

    if not panels:
        raise RuntimeError("No panels were selected.")
    output = args.output
    if output is None:
        output = SCRIPT_DIR / "results" / f"mhr_rest_{args.skin}_{args.view}.png"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.hconcat(panels)):
        raise RuntimeError(f"Could not write output image: {output}")

    reports: dict[str, dict[str, float | int | bool | str]] = {}
    if not args.no_validate:
        # Both modes deliberately receive the same arrays.  This validates the
        # core contract: reskinning changes appearance, never MHR geometry.
        reports["appearance_only_geometry"] = compare_same_geometry(
            vertices,
            vertices.copy(),
            faces,
            faces.copy(),
        )
        _print_report("Appearance-only geometry validation", reports["appearance_only_geometry"])

    if args.reference_mesh is not None:
        reports["external_reference"] = compare_reference_mesh(
            args.reference_mesh.expanduser().resolve(),
            vertices,
            faces,
        )
        _print_report("External reference mesh validation", reports["external_reference"])

    if args.validation_json is not None:
        validation_json = args.validation_json.expanduser().resolve()
        validation_json.parent.mkdir(parents=True, exist_ok=True)
        with validation_json.open("w") as handle:
            json.dump(reports, handle, indent=2)
            handle.write("\n")

    print(f"MHR root:        {mhr_root}")
    print(f"Vertices:        {len(vertices)}")
    print(f"Faces:           {len(faces)}")
    print("Pose:            default zero-parameter rest pose")
    print(f"Skin/view:       {args.skin} / {args.view}")
    print(f"Pose overlay:    {args.pose_overlay} (driven by current MHR joints)")
    print(f"Saved:           {output}")


if __name__ == "__main__":
    main()
