#!/usr/bin/env python3
"""Render the copied MHR body in its zero-parameter default pose.

The geometry is the same lean MHR identity plus the standard
``Lean_Abs_Definition`` layer used by the ROM examples.  Only the appearance
changes between ``blue`` and ``reskinned``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
if "anaconda" in os.environ.get("__EGL_VENDOR_LIBRARY_DIRS", ""):
    os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = "/usr/share/glvnd/egl_vendor.d"

BODY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BODY_ROOT.parents[2]
ROM_ROOT = REPO_ROOT / "my_scripts" / "data_visualization" / "rom_visualization_scripts"
STYLIZED_ROOT = ROM_ROOT / "stylized_mhr"
for import_path in (ROM_ROOT.parent, ROM_ROOT, STYLIZED_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import cv2
import numpy as np
import pyrender
import torch
import trimesh

import render_torchscript_lean_abs_rom as abs_renderer  # noqa: E402
import render_torchscript_lean_rom as direct  # noqa: E402
from rom_visualization_scripts import visualize_mhr_shoulder_flexion as presentation  # noqa: E402
from rom_visualization_scripts.mhr_shoulder import MHR_BODY_EDGES  # noqa: E402
from rom_visualization_scripts.mhr_pose_overlay import draw_pose_overlay  # noqa: E402
from rom_visualization_scripts.mhr_skin import (  # noqa: E402
    make_render_material,
    make_vertex_colors,
)


DEFAULT_MODEL = BODY_ROOT / "native" / "mhr_model.pt"
DEFAULT_SCENE = BODY_ROOT / "blender" / "stylized_mhr_lean_abs_outline.blend"
DEFAULT_SCULPT_CACHE = BODY_ROOT / "derived" / "lean_abs_standard" / "sculpt_bind.npz"
DEFAULT_BLENDER = Path(
    os.environ.get("MHR_BLENDER", "/home/haziq/blender-3.6.17-linux-x64/blender")
)
PREVIEW_ROOT = BODY_ROOT / "previews"

APPEARANCES = {
    "blue": {
        "skin": "dark_fresnel",
        "background": (0.012, 0.016, 0.028),
        "default_name": "mhr_default_pose_blue_dark_fresnel.png",
    },
    "reskinned": {
        "skin": "reskinned",
        "background": (0.16, 0.18, 0.21),
        "default_name": "mhr_default_pose_reskinned_dark_gray.png",
    },
}

LOWER_LIMB_JOINTS = (2, 3, 4, 8, 18, 19, 20, 24)
# Stop the central chain at c_head. The MHR body definition continues with
# c_head -> c_jaw, but that final facial segment is omitted from this compact
# default-pose presentation.
DEFAULT_OVERLAY_EDGES = tuple(edge for edge in MHR_BODY_EDGES if edge != (113, 114))


def _ensure_sculpt_cache(cache_path: Path, blender_path: Path, scene_path: Path) -> None:
    if cache_path.is_file():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(blender_path),
        "-b",
        "--threads",
        "8",
        "--python-exit-code",
        "1",
        "--python",
        str(STYLIZED_ROOT / "extract_mhr_sculpt.py"),
        "--",
        "--scene",
        str(scene_path.resolve()),
        "--output",
        str(cache_path.resolve()),
        "--sculpt-keys",
        "Lean_Abs_Definition",
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        raise RuntimeError(
            "Could not regenerate the sculpt cache:\n"
            + process.stdout
            + process.stderr
        )


def _load_default_geometry(model_path: Path, sculpt_cache: Path):
    with np.load(sculpt_cache, allow_pickle=False) as data:
        sculpt = {name: data[name] for name in data.files}

    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    compatibility = abs_renderer.verify_compatibility(model, sculpt)

    # All 204 model/pose parameters are zero: native MHR default/rest pose.
    parameters = torch.zeros(1, 204, dtype=torch.float32)
    with torch.no_grad():
        posed, _, state, _, _ = abs_renderer.skin_sculpt(
            model,
            parameters,
            sculpt["sculpt_delta_cm"],
            strength=1.0,
        )

    vertices = (posed[0] / 100.0).numpy().astype(np.float64)
    joints = (state[0, :, :3] / 100.0).numpy().astype(np.float64)
    faces = sculpt["faces"].astype(np.int32, copy=False)
    return vertices, joints, faces, compatibility


def _make_default_view_spec(vertices, joints, body_frame, width, height, horizontal, depth):
    # _make_view_spec only needs these two points from its ROM result object
    # to establish the same generous framing used by the ROM renderer.
    root = joints[presentation.MHR_JOINTS["root"]]
    placeholder = SimpleNamespace(reference_end=root, measurement_end=root)
    return presentation._make_view_spec(
        vertices,
        joints,
        placeholder,
        width,
        height,
        horizontal,
        body_frame.up,
        depth,
    )


def _render_mesh(vertices, faces, spec, skin, background):
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
        skin=skin,
        normals=mesh_trimesh.vertex_normals,
    )

    scene = pyrender.Scene(
        bg_color=[*background, 1.0],
        ambient_light=[0.35, 0.35, 0.35],
    )
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)
    material = make_render_material(skin)
    if material is not None:
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


def _render_panels(vertices, joints, faces, appearance, width, height, background):
    body_frame = direct.build_mhr_body_frame(joints)
    key_joints = [
        presentation.MHR_JOINTS[name]
        for name in (
            "root",
            "c_spine3",
            "c_neck",
            "c_head",
            "r_shoulder",
            "r_elbow",
            "r_wrist",
            "l_shoulder",
            "l_elbow",
            "l_wrist",
        )
    ]
    key_joints.extend(LOWER_LIMB_JOINTS)

    panels = []
    for horizontal, depth in (
        (-body_frame.right, body_frame.forward),
        (body_frame.forward, body_frame.right),
    ):
        spec = _make_default_view_spec(
            vertices,
            joints,
            body_frame,
            width,
            height,
            horizontal,
            depth,
        )
        panel = _render_mesh(
            vertices,
            faces,
            spec,
            appearance["skin"],
            background,
        )
        draw_pose_overlay(
            panel,
            spec,
            joints,
            style="unreal",
            line_thickness=2,
            joint_radius=5,
            joint_indices=key_joints,
            edge_indices=DEFAULT_OVERLAY_EDGES,
        )
        panels.append(panel)
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appearance", choices=tuple(APPEARANCES), required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--sculpt-cache", type=Path, default=DEFAULT_SCULPT_CACHE)
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--background", nargs=3, type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    for path in (args.model,):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.sculpt_cache.is_file():
        for path in (args.scene, args.blender):
            if not path.is_file():
                raise FileNotFoundError(path)
        _ensure_sculpt_cache(args.sculpt_cache, args.blender, args.scene)

    appearance = APPEARANCES[args.appearance]
    background = tuple(
        appearance["background"] if args.background is None else
        (float(value) for value in args.background)
    )
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (PREVIEW_ROOT / appearance["default_name"]).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    vertices, joints, faces, compatibility = _load_default_geometry(
        args.model.resolve(),
        args.sculpt_cache.resolve(),
    )
    panels = _render_panels(
        vertices,
        joints,
        faces,
        appearance,
        args.width,
        args.height,
        background,
    )
    if not cv2.imwrite(str(output), cv2.hconcat(panels)):
        raise RuntimeError(f"Could not write {output}")
    for label, panel in (("front", panels[0]), ("side", panels[1])):
        panel_path = output.with_name(f"{output.stem}_{label}.png")
        if not cv2.imwrite(str(panel_path), panel):
            raise RuntimeError(f"Could not write {panel_path}")

    print("Pose:              native MHR zero-parameter default pose")
    print("Geometry:          lean identity + Lean_Abs_Definition")
    print(f"Appearance:        {args.appearance}")
    print(f"Vertices/triangles: {len(vertices)}/{len(faces)}")
    print(f"Compatibility:     {compatibility}")
    print(f"Saved:             {output}")


if __name__ == "__main__":
    main()
