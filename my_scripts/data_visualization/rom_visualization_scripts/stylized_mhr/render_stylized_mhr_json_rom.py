#!/usr/bin/env python3
"""Render a JSON ROM frame using Astra's stylized MHR Blender character.

This is intentionally separate from ``visualize_mhr_json_rom.py``.  It keeps
that existing renderer unchanged, transfers the selected JSON pose to
``stylized_mhr.blend``, renders the character in Blender, then reuses the
existing ROM measurement overlay and local source-video panel.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
if "anaconda" in os.environ.get("__EGL_VENDOR_LIBRARY_DIRS", ""):
    os.environ["__EGL_VENDOR_LIBRARY_DIRS"] = "/usr/share/glvnd/egl_vendor.d"

import cv2
import numpy as np
import torch


STYLIZED_DIR = Path(__file__).resolve().parent
ROM_DIR = STYLIZED_DIR.parent
DATA_VISUALIZATION_DIR = ROM_DIR.parent
REPO_ROOT = STYLIZED_DIR.parents[3]
BODY_MODEL_ROOT = REPO_ROOT / "data" / "body_models" / "mhr"
for import_path in (DATA_VISUALIZATION_DIR, ROM_DIR):
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
from rom_visualization_scripts.visualize_mhr_shoulder_flexion import (  # noqa: E402
    _draw_measurement,
    _letterbox,
    _make_view_spec,
    _load_video_frame,
)


def _load_frame_with_state(json_path: Path, time_s: float, model_path: Path):
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
    state = skeleton_state[0].cpu().numpy().astype(np.float64)
    joints = (skeleton_state[0, :, :3] / 100.0).cpu().numpy().astype(np.float64)
    joint_names = list(model.character_torch.skeleton.joint_names)
    return data, frame_idx, actual_time_s, vertices, joints, state, joint_names


def _view_payload(spec, output_path: Path):
    return {
        "x_axis": np.asarray(spec.x_axis).tolist(),
        "y_axis": np.asarray(spec.y_axis).tolist(),
        "z_axis": np.asarray(spec.z_axis).tolist(),
        "origin": np.asarray(spec.origin).tolist(),
        "x_center": float(spec.x_center),
        "y_center": float(spec.y_center),
        "xmag": float(spec.xmag),
        "ymag": float(spec.ymag),
        "width": int(spec.width),
        "height": int(spec.height),
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--joint", choices=("shoulder",), default="shoulder")
    parser.add_argument("--movement", choices=("flexion",), default="flexion")
    parser.add_argument("--time", required=True, type=float)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--blend",
        type=Path,
        default=BODY_MODEL_ROOT / "blender" / "stylized_mhr.blend",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=BODY_MODEL_ROOT / "native" / "mhr_model.pt",
    )
    parser.add_argument(
        "--blender",
        type=Path,
        default=Path("/home/haziq/blender-3.6.17-linux-x64/blender"),
    )
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--panel-width", type=int, default=800)
    parser.add_argument("--pose-overlay", choices=("unreal", "legacy"), default="unreal")
    args = parser.parse_args()

    if not args.json.is_file():
        raise FileNotFoundError(args.json)
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if not args.blend.is_file():
        raise FileNotFoundError(args.blend)
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.blender.is_file():
        raise FileNotFoundError(args.blender)

    data, frame_idx, actual_time_s, vertices, joints, state, joint_names = _load_frame_with_state(
        args.json, args.time, args.model
    )
    body_frame = build_mhr_body_frame(joints)
    side = select_shoulder_side(joints, args.side, args.movement)
    result = compute_shoulder_flexion(joints, side, body_frame)

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

    args.output = args.output.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="stylized-mhr-rom-") as temp_dir:
        temp_dir = Path(temp_dir)
        front_path = temp_dir / "stylized_front.png"
        sagittal_path = temp_dir / "stylized_sagittal.png"
        config_path = temp_dir / "blender_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "state": state.tolist(),
                    "joint_names": joint_names,
                    "views": {
                        "front": _view_payload(front_spec, front_path),
                        "sagittal": _view_payload(sagittal_spec, sagittal_path),
                    },
                }
            )
        )

        blender_script = STYLIZED_DIR / "blender_render_stylized_rom.py"
        command = [
            str(args.blender),
            "-b",
            str(args.blend),
            "--python-exit-code",
            "1",
            "--python",
            str(blender_script),
            "--",
            str(config_path),
        ]
        subprocess.run(command, check=True)

        front_panel = cv2.imread(str(front_path), cv2.IMREAD_COLOR)
        sagittal_panel = cv2.imread(str(sagittal_path), cv2.IMREAD_COLOR)
        if front_panel is None or sagittal_panel is None:
            raise RuntimeError("Blender did not produce both stylized ROM panels")

        _draw_measurement(
            front_panel,
            front_spec,
            joints,
            result,
            "Stylized MHR body-aligned FRONT",
            draw_arc=False,
            draw_positive_axis=False,
            pose_overlay=args.pose_overlay,
        )
        _draw_measurement(
            sagittal_panel,
            sagittal_spec,
            joints,
            result,
            "Stylized MHR body-aligned SAGITTAL",
            draw_arc=True,
            draw_positive_axis=False,
            show_result_text=False,
            pose_overlay=args.pose_overlay,
        )

        source = _load_video_frame(args.video.resolve(), frame_idx)
        source_width = int(round(args.height * 576.0 / 1024.0))
        if source is not None:
            source_width = int(round(args.height * source.shape[1] / source.shape[0]))
        source_panel = _letterbox(source, source_width, args.height)
        if source is None:
            cv2.putText(
                source_panel,
                "source video unavailable",
                (18, args.height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 200),
                2,
                cv2.LINE_AA,
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(
            str(args.output), cv2.hconcat([source_panel, front_panel, sagittal_panel])
        ):
            raise RuntimeError(f"Could not write {args.output}")

    print(f"JSON frame:       {frame_idx}")
    print(f"JSON time:        {actual_time_s:.3f}s")
    print("Character source: Astra stylized_mhr.blend")
    print("Joint/movement:   shoulder flexion")
    print(f"Side:             {result.side}")
    print(f"ROM angle:        {result.angle_deg:+.2f} deg")
    print(f"Saved:            {args.output}")


if __name__ == "__main__":
    main()
