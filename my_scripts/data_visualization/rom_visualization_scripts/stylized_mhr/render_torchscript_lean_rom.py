#!/usr/bin/env python3
"""Render a lean TorchScript MHR JSON ROM frame with the original renderer."""

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
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROM_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
BODY_MODEL_ROOT = REPO_ROOT / "data" / "body_models" / "mhr"
DEFAULT_MODEL = BODY_MODEL_ROOT / "native" / "mhr_model.pt"
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
from rom_visualization_scripts.visualize_mhr_json_rom import (  # noqa: E402
    _planar_panel_pair,
    _source_panel,
)


LEAN_SHAPE = np.zeros(45, dtype=np.float32)
LEAN_SHAPE[:3] = (-1.5, -0.75, 0.75)


def _load_torchscript_frame(json_path: Path, time_s: float, model_path: Path):
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
    expression = torch.zeros(1, 72, dtype=torch.float32)

    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    with torch.no_grad():
        vertices_cm, skeleton_state = model(
            torch.from_numpy(LEAN_SHAPE).unsqueeze(0),
            model_params,
            expression,
        )

    vertices = (vertices_cm[0] / 100.0).cpu().numpy().astype(np.float64)
    joints = (skeleton_state[0, :, :3] / 100.0).cpu().numpy().astype(np.float64)
    faces = model.character_torch.mesh.faces.cpu().numpy().astype(np.int32)
    return data, frame_idx, actual_time_s, vertices, joints, faces


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
        default=DEFAULT_MODEL,
    )
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--panel-width", type=int, default=800)
    args = parser.parse_args()

    for path in (args.json, args.video, args.model):
        if not path.is_file():
            raise FileNotFoundError(path)

    data, frame_idx, actual_time_s, vertices, joints, faces = _load_torchscript_frame(
        args.json.resolve(), args.time, args.model.resolve()
    )
    body_frame = build_mhr_body_frame(joints)
    side = select_shoulder_side(joints, args.side, "flexion")
    result = compute_shoulder_flexion(joints, side, body_frame)

    front_spec = __import__(
        "rom_visualization_scripts.visualize_mhr_shoulder_flexion",
        fromlist=["_make_view_spec"],
    )._make_view_spec(
        vertices,
        joints,
        result,
        args.panel_width,
        args.height,
        -body_frame.right,
        body_frame.up,
        body_frame.forward,
    )
    sagittal_spec = __import__(
        "rom_visualization_scripts.visualize_mhr_shoulder_flexion",
        fromlist=["_make_view_spec"],
    )._make_view_spec(
        vertices,
        joints,
        result,
        args.panel_width,
        args.height,
        body_frame.forward,
        body_frame.up,
        body_frame.right,
    )

    panels = _planar_panel_pair(
        vertices,
        joints,
        faces,
        result,
        body_frame,
        "shoulder",
        "flexion",
        "dark_fresnel",
        "unreal",
        args.panel_width,
        args.height,
    )
    source_panel = _source_panel(
        args.video.resolve(),
        frame_idx,
        actual_time_s,
        args.time,
        args.panel_width,
        args.height,
    )

    args.output = args.output.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), cv2.hconcat([source_panel, *panels])):
        raise RuntimeError(f"Could not write {args.output}")

    print(f"JSON frame:       {frame_idx}")
    print(f"JSON time:        {actual_time_s:.3f}s")
    print("Shape:            lean [-1.5, -0.75, +0.75, 0...]")
    print("Renderer:         direct TorchScript MHR + PyRender dark_fresnel")
    print(f"ROM angle:        {result.angle_deg:+.2f} deg")
    print(f"Saved:            {args.output}")


if __name__ == "__main__":
    main()
