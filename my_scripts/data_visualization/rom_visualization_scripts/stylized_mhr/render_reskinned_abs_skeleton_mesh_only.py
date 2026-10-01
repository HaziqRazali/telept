#!/usr/bin/env python3
"""Render the standard lean-abs MHR mesh with skeleton markers only.

This follows ``render_torchscript_lean_abs_rom.py`` for geometry, sculpt
transfer, pose reconstruction, and view framing.  It intentionally omits the
video panel and ROM annotations, and changes only the mesh appearance and
panel background for this deliverable.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
if "anaconda" in os.environ.get("__EGL_VENDOR_LIBRARY_DIRS", ""):
    os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = "/usr/share/glvnd/egl_vendor.d"

import cv2
import numpy as np
import pyrender
import torch
import trimesh


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
ROM_DIR = ROOT.parent
for import_path in (ROM_DIR.parent, ROM_DIR, ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import render_torchscript_lean_abs_rom as reference  # noqa: E402
import render_torchscript_lean_rom as direct  # noqa: E402
from rom_visualization_scripts import visualize_mhr_shoulder_flexion as presentation  # noqa: E402
from rom_visualization_scripts.mhr_pose_overlay import draw_pose_overlay  # noqa: E402
from rom_visualization_scripts.mhr_skin import (  # noqa: E402
    make_render_material,
    make_vertex_colors,
)


BODY_MODEL_ROOT = REPO_ROOT / "data" / "body_models" / "mhr"
DEFAULT_SCENE = BODY_MODEL_ROOT / "blender" / "stylized_mhr_lean_abs_outline.blend"
DEFAULT_MODEL = BODY_MODEL_ROOT / "native" / "mhr_model.pt"
DEFAULT_OUTPUT = ROM_DIR / "results" / (
    "adhesive_capsulitis_right_shoulder_flexion_reskinned_dark_gray.png"
)
DEFAULT_WORK_DIR = BODY_MODEL_ROOT / "derived" / "lean_abs_standard"
DEFAULT_BACKGROUND = (0.16, 0.18, 0.21)

# These are the same lower-limb endpoints used by the reference renderer.
LOWER_LIMB_JOINTS = {
    "l_hip": 2,
    "l_knee": 3,
    "l_ankle": 4,
    "l_toe": 8,
    "r_hip": 18,
    "r_knee": 19,
    "r_ankle": 20,
    "r_toe": 24,
}


def _render_mesh(vertices, faces, spec, background):
    """Render one mesh panel using the existing reskinned MHR material."""

    q_vertices = spec.coordinates(vertices)
    q_vertices[:, 0] -= spec.x_center
    q_vertices[:, 1] -= spec.y_center

    basis = np.column_stack((spec.x_axis, spec.y_axis, spec.z_axis))
    render_faces = faces[:, ::-1] if np.linalg.det(basis) < 0.0 else faces
    mesh_trimesh = trimesh.Trimesh(
        vertices=q_vertices,
        faces=render_faces,
        process=False,
    )
    mesh_trimesh.visual.vertex_colors = make_vertex_colors(
        q_vertices,
        render_faces,
        skin="reskinned",
        normals=mesh_trimesh.vertex_normals,
    )

    scene = pyrender.Scene(
        bg_color=[*background, 1.0],
        ambient_light=[0.35, 0.35, 0.35],
    )
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)
    material = make_render_material("reskinned")
    for primitive in pyrender_mesh.primitives:
        primitive.material = material
    scene.add(pyrender_mesh)

    camera = pyrender.OrthographicCamera(
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
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=3.0),
        pose=np.eye(4),
    )
    for position, intensity in (([2.0, 3.0, 4.0], 18.0), ([-2.0, 1.0, 2.0], 8.0)):
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


def skeleton_panels(vertices, joints, faces, result, body_frame, width, height, background):
    """Create the same front/sagittal skeleton-only panels as the reference."""

    key_joints = [
        presentation.MHR_JOINTS[name]
        for name in (
            "root",
            "c_spine3",
            "c_neck",
            "r_shoulder",
            "r_elbow",
            "r_wrist",
            "l_shoulder",
            "l_elbow",
            "l_wrist",
        )
    ]
    key_joints.extend(LOWER_LIMB_JOINTS.values())

    panels = []
    for horizontal, depth in (
        (-body_frame.right, body_frame.forward),
        (body_frame.forward, body_frame.right),
    ):
        spec = presentation._make_view_spec(
            vertices,
            joints,
            result,
            width,
            height,
            horizontal,
            body_frame.up,
            depth,
        )
        panel = _render_mesh(vertices, faces, spec, background)
        draw_pose_overlay(
            panel,
            spec,
            joints,
            style="unreal",
            line_thickness=2,
            joint_radius=5,
            joint_indices=key_joints,
        )
        panels.append(panel)
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--time", type=float, default=10.076)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sculpt-scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument(
        "--blender",
        type=Path,
        default=Path(os.environ.get("MHR_BLENDER", "/home/haziq/blender-3.6.17-linux-x64/blender")),
    )
    parser.add_argument("--sculpt-strength", type=float, default=1.0)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--panel-width", type=int, default=800)
    parser.add_argument("--background", nargs=3, type=float, default=DEFAULT_BACKGROUND)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not 0.0 <= args.sculpt_strength <= 1.0:
        raise ValueError("--sculpt-strength must be in [0, 1]")
    for path in (args.json, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    cache = args.work_dir / "sculpt_bind.npz"
    if not cache.is_file():
        for path in (args.sculpt_scene, args.blender):
            if not path.is_file():
                raise FileNotFoundError(path)
        extraction_command = [
            str(args.blender),
            "-b",
            "--threads",
            "8",
            "--python-exit-code",
            "1",
            "--python",
            str(ROOT / "extract_mhr_sculpt.py"),
            "--",
            "--scene",
            str(args.sculpt_scene.resolve()),
            "--output",
            str(cache.resolve()),
            "--sculpt-keys",
            "Lean_Abs_Definition",
        ]
        process = subprocess.run(extraction_command, capture_output=True, text=True)
        (args.work_dir / "extraction_reskinned_mesh_only.log").write_text(
            process.stdout + process.stderr
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"Geometry extraction failed; see {args.work_dir / 'extraction_reskinned_mesh_only.log'}"
            )

    with np.load(cache, allow_pickle=False) as data:
        sculpt = {name: data[name] for name in data.files}

    torch.set_num_threads(8)
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    compatibility = reference.verify_compatibility(model, sculpt)
    parameters, frame_index, actual_time = reference.load_pose(args.json.resolve(), args.time)
    with torch.no_grad():
        posed, baseline, state, rest, delta = reference.skin_sculpt(
            model,
            parameters,
            sculpt["sculpt_delta_cm"],
            args.sculpt_strength,
        )
        transfer = reference.validate_transfer(
            model,
            parameters,
            sculpt,
            args.sculpt_strength,
            posed,
            baseline,
            state,
            rest,
            delta,
        )

    vertices = (posed[0] / 100.0).numpy().astype(np.float64)
    joints = (state[0, :, :3] / 100.0).numpy().astype(np.float64)
    faces = sculpt["faces"]
    body_frame = direct.build_mhr_body_frame(joints)
    side = direct.select_shoulder_side(joints, args.side, "flexion")
    result = direct.compute_shoulder_flexion(joints, side, body_frame)

    panels = skeleton_panels(
        vertices,
        joints,
        faces,
        result,
        body_frame,
        args.panel_width,
        args.height,
        tuple(float(value) for value in args.background),
    )

    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), cv2.hconcat(panels)):
        raise RuntimeError(f"Could not write {args.output}")

    for label, panel in (("front", panels[0]), ("side", panels[1])):
        panel_path = args.output.with_name(f"{args.output.stem}_{label}.png")
        if not cv2.imwrite(str(panel_path), panel):
            raise RuntimeError(f"Could not write {panel_path}")

    report = {
        "renderer": "render_torchscript_lean_abs_rom.py skeleton_panels path",
        "mesh_only": True,
        "video_panel": False,
        "rom_annotations": False,
        "appearance": "reskinned",
        "background": list(args.background),
        "sculpt_scene": str(args.sculpt_scene.resolve()),
        "sculpt_key": "Lean_Abs_Definition",
        "sculpt_strength": args.sculpt_strength,
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
        "frame_index": int(frame_index),
        "actual_time": float(actual_time),
        "shoulder_flexion_used_for_framing_only": float(result.angle_deg),
        "lower_limb_joint_markers": [int(value) for value in LOWER_LIMB_JOINTS.values()],
        "compatibility": compatibility,
        "transfer": transfer,
        "output": str(args.output),
        "output_front": str(args.output.with_name(f"{args.output.stem}_front.png")),
        "output_side": str(args.output.with_name(f"{args.output.stem}_side.png")),
    }
    (args.work_dir / "reskinned_mesh_only_validation.json").write_text(
        __import__("json").dumps(report, indent=2)
    )
    print(f"JSON frame:       {frame_index}")
    print(f"JSON time:        {actual_time:.3f}s")
    print("Geometry:         standard lean + Lean_Abs_Definition")
    print("Appearance:       reskinned")
    print(f"Background:       {tuple(args.background)}")
    print("Panels:           front + sagittal, mesh and skeleton only")
    print("Lower-limb dots:  enabled")
    print(f"ROM angle used for framing only: {result.angle_deg:+.2f} deg")
    print(f"Saved:            {args.output}")


if __name__ == "__main__":
    main()
