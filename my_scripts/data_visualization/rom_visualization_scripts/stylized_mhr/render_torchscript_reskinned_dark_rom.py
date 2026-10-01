#!/usr/bin/env python3
"""Render the original JSON-driven reskinned MHR ROM image on dark gray."""

from __future__ import annotations

import argparse
import os
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


SCRIPT_DIR = Path(__file__).resolve().parent
ROM_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BODY_MODEL_ROOT = REPO_ROOT / "data" / "body_models" / "mhr"
for import_path in (ROM_DIR.parent, ROM_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from rom_visualization_scripts.mhr_json import (  # noqa: E402
    _finite_vector,
    _mat_to_euler_zyx,
    _rot6d_to_matrix,
    _scale_params_68,
    load_mhr_json,
    select_mhr_json_frame,
)
from rom_visualization_scripts.mhr_shoulder import (  # noqa: E402
    build_mhr_body_frame,
    compute_shoulder_flexion,
    select_shoulder_side,
)
from rom_visualization_scripts.mhr_pose_overlay import draw_pose_overlay  # noqa: E402,F401
from rom_visualization_scripts.mhr_skin import (  # noqa: E402
    make_render_material,
    make_vertex_colors,
)
from rom_visualization_scripts.visualize_mhr_json_rom import _source_panel  # noqa: E402
from rom_visualization_scripts.visualize_mhr_shoulder_flexion import (  # noqa: E402
    _draw_measurement,
    _make_view_spec,
)


def _load_frame(json_path: Path, time_s: float, model_path: Path):
    data = load_mhr_json(json_path)
    frame, frame_idx, actual_time_s = select_mhr_json_frame(data, time_s=time_s)

    params = np.asarray(frame["pred_body_params"], dtype=np.float32).reshape(-1)
    if params.size < 136:
        raise ValueError(f"MHR pose requires 136 values, got {params.size}")

    model_params = torch.zeros(1, 204, dtype=torch.float32)
    model_params[0, 3:6] = torch.from_numpy(
        _mat_to_euler_zyx(_rot6d_to_matrix(params[:6]))
    )
    model_params[0, 6:136] = torch.from_numpy(params[6:136])
    model_params[0, 136:204] = torch.from_numpy(
        _scale_params_68(frame.get("pred_scale_params"))
    )
    shape = torch.from_numpy(
        _finite_vector(frame.get("shape_params"), 45)
    ).unsqueeze(0)
    expression = torch.zeros(1, 72, dtype=torch.float32)

    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    with torch.no_grad():
        vertices_cm, skeleton_state = model(shape, model_params, expression)

    vertices = (vertices_cm[0] / 100.0).cpu().numpy().astype(np.float64)
    joints = (skeleton_state[0, :, :3] / 100.0).cpu().numpy().astype(np.float64)
    faces = model.character_torch.mesh.faces.cpu().numpy().astype(np.int32)
    return data, frame_idx, actual_time_s, vertices, joints, faces


def _render_mesh(vertices, faces, spec, background):
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
    for position, intensity in [([2.0, 3.0, 4.0], 18.0), ([-2.0, 1.0, 2.0], 8.0)]:
        light_pose = np.eye(4, dtype=np.float64)
        light_pose[:3, 3] = position
        scene.add(pyrender.PointLight(color=np.ones(3), intensity=intensity), pose=light_pose)

    renderer = pyrender.OffscreenRenderer(spec.width, spec.height)
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
    finally:
        renderer.delete()
    return cv2.cvtColor(color, cv2.COLOR_RGB2BGR)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--time", required=True, type=float)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=BODY_MODEL_ROOT / "native" / "mhr_model.pt",
    )
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--panel-width", type=int, default=800)
    parser.add_argument("--background", nargs=3, type=float, default=(0.16, 0.18, 0.21))
    args = parser.parse_args()

    for path in (args.json, args.video, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)

    data, frame_idx, actual_time_s, vertices, joints, faces = _load_frame(
        args.json.resolve(), args.time, args.model.resolve()
    )
    body_frame = build_mhr_body_frame(joints)
    side = select_shoulder_side(joints, args.side, "flexion")
    result = compute_shoulder_flexion(joints, side, body_frame)

    front_spec = _make_view_spec(
        vertices, joints, result, args.panel_width, args.height,
        -body_frame.right, body_frame.up, body_frame.forward,
    )
    sagittal_spec = _make_view_spec(
        vertices, joints, result, args.panel_width, args.height,
        body_frame.forward, body_frame.up, body_frame.right,
    )
    front_panel = _render_mesh(vertices, faces, front_spec, args.background)
    sagittal_panel = _render_mesh(vertices, faces, sagittal_spec, args.background)
    _draw_measurement(
        front_panel, front_spec, joints, result,
        "MHR body-aligned FRONT", draw_arc=False, pose_overlay="unreal",
    )
    _draw_measurement(
        sagittal_panel, sagittal_spec, joints, result,
        "MHR body-aligned SAGITTAL", draw_arc=True,
        show_result_text=False, pose_overlay="unreal",
    )

    source_panel = _source_panel(
        args.video.resolve(), frame_idx, actual_time_s, args.time,
        args.panel_width, args.height,
    )
    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(args.output), cv2.hconcat([source_panel, front_panel, sagittal_panel])
    ):
        raise RuntimeError(f"Could not write {args.output}")

    print(f"JSON frame:       {frame_idx}")
    print(f"JSON time:        {actual_time_s:.3f}s")
    print("Shape:            original JSON frame shape")
    print("Skin:             reskinned")
    print(f"Background:       {tuple(args.background)}")
    print(f"ROM angle:        {result.angle_deg:+.2f} deg")
    print(f"Saved:            {args.output}")


if __name__ == "__main__":
    main()
