#!/usr/bin/env python3
"""Skin attached Blender sculpt keys with native MHR, then render with PyRender."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import render_torchscript_lean_rom as direct
import cv2
import numpy as np
import torch
from rom_visualization_scripts import visualize_mhr_shoulder_flexion as presentation


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
BODY_MODEL_ROOT = REPO_ROOT / "data" / "body_models" / "mhr"
DEFAULT_SCENE = BODY_MODEL_ROOT / "blender" / "stylized_mhr_lean_abs_outline.blend"
DEFAULT_MODEL = BODY_MODEL_ROOT / "native" / "mhr_model.pt"
DEFAULT_WORK_DIR = BODY_MODEL_ROOT / "derived" / "lean_abs_standard"
DEFAULT_OUTPUT = direct.ROM_DIR / "results" / "adhesive_capsulitis_right_shoulder_flexion_standard_abs_skeleton_dark_fresnel.png"
LOWER_LIMB_JOINTS = {
    "l_hip": 2, "l_knee": 3, "l_ankle": 4, "l_toe": 8,
    "r_hip": 18, "r_knee": 19, "r_ankle": 20, "r_toe": 24,
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_pose(json_path, time_s):
    data = direct.load_mhr_json(json_path)
    frame, index, actual_time = direct.select_mhr_json_frame(data, time_s=time_s)
    values = np.asarray(frame["pred_body_params"], dtype=np.float32).reshape(-1)
    require(values.size >= 136 and np.isfinite(values).all(), "Invalid MHR JSON pose")
    parameters = torch.zeros(1, 204, dtype=torch.float32)
    parameters[0, 3:6] = torch.from_numpy(direct._mat_to_euler_zyx(direct._rot6d_to_matrix(values[:6])))
    parameters[0, 6:136] = torch.from_numpy(values[6:136])
    parameters[0, 136:204] = torch.from_numpy(direct._scale_params_68(frame.get("pred_scale_params")))
    return parameters, index, actual_time


def verify_compatibility(model, sculpt):
    character = model.character_torch
    basis_error = float(np.max(np.abs(character.blend_shape.base_shape.numpy() - sculpt["basis_cm"])))
    require(basis_error < 1e-4, "Sculpt bind coordinates or vertex order differ from MHR")
    require(np.array_equal(character.mesh.faces.numpy(), sculpt["faces"]), "Sculpt face order differs from MHR")
    require(sculpt["basis_cm"].shape == (18439, 3) and sculpt["faces"].shape == (36874, 3),
            "Expected full-resolution MHR LOD1 topology")
    skin = character.linear_blend_skinning
    names = list(character.skeleton.joint_names)
    expected = np.zeros((18439, len(names)), dtype=np.float32)
    np.add.at(expected, (skin.vert_indices_flattened.numpy(), skin.skin_indices_flattened.numpy()),
              skin.skin_weights_flattened.numpy())
    actual = np.zeros_like(expected)
    groups = sculpt["group_names"].tolist()
    actual[:, [names.index(name) for name in groups]] = sculpt["skin_weights"]
    weight_error = float(np.max(np.abs(expected - actual)))
    require(weight_error < 1e-6, "Sculpt armature weights differ from native MHR")
    delta = sculpt["sculpt_delta_cm"]
    require(delta.shape == (18439, 3) and np.isfinite(delta).all(), "Invalid sculpt delta")
    return {"basis_max_error_cm": basis_error, "faces_exact": True, "skin_weight_max_error": weight_error}


def skin_sculpt(model, parameters, sculpt_delta, strength):
    character = model.character_torch
    identity = torch.from_numpy(direct.LEAN_SHAPE).unsqueeze(0)
    expression = torch.zeros(1, 72, dtype=torch.float32)
    joint_parameters = character.model_parameters_to_joint_parameters(
        torch.cat([parameters, torch.zeros_like(identity)], dim=1))
    state = character.joint_parameters_to_skeleton_state(joint_parameters)
    rest = character.blend_shape(identity) + model.face_expressions_model(expression)
    rest = rest + model.pose_correctives_model(joint_parameters)
    delta = torch.from_numpy(sculpt_delta).unsqueeze(0)
    baseline = character.skin_points(state, rest)
    posed = character.skin_points(state, rest + strength * delta)
    return posed, baseline, state, rest, delta


def triangle_normals(vertices, faces):
    corners = vertices[faces]
    return np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])


def validate_transfer(model, parameters, sculpt, strength, posed, baseline, state, rest, delta):
    character = model.character_torch
    expected, expected_state = model(torch.from_numpy(direct.LEAN_SHAPE).unsqueeze(0), parameters,
                                     torch.zeros(1, 72))
    baseline_error = float(torch.max(torch.abs(baseline - expected)))
    require(baseline_error < 1e-4, "Zero-sculpt path differs from native model.forward")
    require(torch.equal(state, expected_state), "Sculpt changed JSON-driven joints")
    magnitude = np.linalg.norm(sculpt["sculpt_delta_cm"], axis=1)
    untouched = magnitude == 0
    support = magnitude > 1e-5
    vertices = posed[0].numpy()
    plain = baseline[0].numpy()
    displacement = vertices - plain
    untouched_error = float(np.max(np.abs(displacement[untouched])))
    require(untouched_error < 1e-5, "Sculpt changed vertices outside its support")
    original_basis = sculpt["basis_cm"]
    extremities = ((original_basis[:, 1] > 149) | (np.abs(original_basis[:, 0]) > 48)
                   | (original_basis[:, 1] < 65))
    require(np.max(np.abs(sculpt["sculpt_delta_cm"][extremities])) < 1e-5,
            "Sculpt would alter head/fingers or lower legs")
    transformed = state.clone()
    rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    transformed[:, :, :3] = state[:, :, :3] @ rotation.T + torch.tensor([20., -15., 10.])
    factor = 2 ** -0.5
    quat = state[:, :, 3:7]
    transformed[:, :, 3:7] = factor * torch.stack((quat[:, :, 0] - quat[:, :, 1],
                                                  quat[:, :, 0] + quat[:, :, 1],
                                                  quat[:, :, 2] + quat[:, :, 3],
                                                  quat[:, :, 3] - quat[:, :, 2]), dim=-1)
    moved = character.skin_points(transformed, rest + strength * delta)
    moved_plain = character.skin_points(transformed, rest)
    equivariance_error = float(torch.max(torch.abs((moved - moved_plain) - (posed - baseline) @ rotation.T)))
    require(equivariance_error < 1e-3, "Sculpt did not rotate with the armature")
    neutral_parameters = torch.zeros_like(parameters)
    neutral_posed, neutral_plain, _, _, _ = skin_sculpt(model, neutral_parameters,
                                                       sculpt["sculpt_delta_cm"], strength)
    require(torch.isfinite(posed).all().item(), "Nonfinite posed vertices")
    faces = sculpt["faces"]
    dot = np.einsum("ij,ij->i", triangle_normals(plain, faces), triangle_normals(vertices, faces))
    flipped = int(np.count_nonzero(dot <= 0))
    require(flipped == 0, "Sculpt introduced reversed triangles in selected pose")
    neutral_dot = np.einsum("ij,ij->i", triangle_normals(neutral_plain[0].numpy(), faces),
                            triangle_normals(neutral_posed[0].numpy(), faces))
    require(np.all(neutral_dot > 0), "Sculpt introduced reversed neutral triangles")
    naive_error = float(np.linalg.norm(displacement - strength * sculpt["sculpt_delta_cm"], axis=1).max())
    if strength > 0:
        require(naive_error > 0.01, "Selected-pose check did not distinguish skinning from neutral-delta addition")
        require(np.count_nonzero(np.linalg.norm(displacement, axis=1) > 0.01) > 100,
                "Sculpt is not visible in posed geometry")
    return {"native_forward_parity_max_cm": baseline_error, "joints_unchanged": True,
            "untouched_vertices": int(np.count_nonzero(untouched)), "untouched_max_error_cm": untouched_error,
            "head_fingers_lower_legs_unchanged": True,
            "sculpt_support_vertices": int(np.count_nonzero(support)),
            "max_posed_sculpt_mm": float(np.linalg.norm(displacement, axis=1).max() * 10),
            "rotated_skeleton_equivariance_error_cm": equivariance_error,
            "difference_from_wrong_world_delta_max_cm": naive_error,
            "new_reversed_triangles": flipped, "neutral_new_reversed_triangles": 0}


def skeleton_panels(vertices, joints, faces, result, body_frame, appearance, overlay, width, height):
    panels = []
    key_joints = [presentation.MHR_JOINTS[name] for name in
                  ("root", "c_spine3", "c_neck", "r_shoulder", "r_elbow", "r_wrist",
                   "l_shoulder", "l_elbow", "l_wrist")]
    key_joints.extend(LOWER_LIMB_JOINTS.values())
    for horizontal, depth in ((-body_frame.right, body_frame.forward), (body_frame.forward, body_frame.right)):
        spec = presentation._make_view_spec(vertices, joints, result, width, height,
                                            horizontal, body_frame.up, depth)
        panel = presentation._render_mesh(vertices, faces, spec, skin=appearance)
        presentation.draw_pose_overlay(panel, spec, joints, style=overlay,
                                       line_thickness=2, joint_radius=5, joint_indices=key_joints)
        panels.append(panel)
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--time", type=float, default=10.076)
    parser.add_argument("--side", choices=("right", "left"), default="right")
    parser.add_argument("--joint", choices=("shoulder",), default="shoulder")
    parser.add_argument("--movement", choices=("flexion",), default="flexion")
    parser.add_argument("--appearance", choices=("dark_fresnel",), default="dark_fresnel")
    parser.add_argument("--pose-overlay", choices=("unreal",), default="unreal")
    parser.add_argument("--annotations", choices=("skeleton", "rom"), default="skeleton")
    parser.add_argument("--definition", choices=("standard", "enhanced"), default="standard")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sculpt-scene", type=Path)
    parser.add_argument(
        "--blender",
        type=Path,
        default=Path(os.environ.get("MHR_BLENDER", "/home/haziq/blender-3.6.17-linux-x64/blender")),
    )
    parser.add_argument("--sculpt-strength", type=float, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--panel-width", type=int, default=800)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.sculpt_scene is None:
        args.sculpt_scene = (DEFAULT_SCENE if args.definition == "standard" else
                             ROOT / "lean_abs_enhanced" / "stylized_mhr_lean_abs_enhanced.blend")
    sculpt_keys = ["Lean_Abs_Definition"]
    if args.definition == "enhanced":
        sculpt_keys.append("Abs_Extra_Definition")
    require(0 <= args.sculpt_strength <= 1, "Sculpt strength must be in [0, 1]")
    for path in (args.json, args.video, args.model):
        require(path.is_file(), f"Missing input: {path}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    protected_paths = [direct.ROM_DIR / "visualize_mhr_json_rom.py", ROOT / "render_stylized_mhr_json_rom.py",
                       ROOT / "render_torchscript_lean_rom.py", *ROOT.rglob("*.blend")]
    protected = {str(path): digest(path) for path in protected_paths}
    cache_name = "sculpt_bind.npz" if args.definition == "standard" else "sculpt_bind_enhanced.npz"
    cache = args.work_dir / cache_name
    if not cache.is_file():
        for path in (args.sculpt_scene, args.blender):
            require(path.is_file(), f"Missing input needed to regenerate sculpt cache: {path}")
        command = [str(args.blender), "-b", "--threads", "8", "--python-exit-code", "1", "--python",
                   str(ROOT / "extract_mhr_sculpt.py"), "--", "--scene", str(args.sculpt_scene.resolve()),
                   "--output", str(cache.resolve()), "--sculpt-keys", *sculpt_keys]
        process = subprocess.run(command, capture_output=True, text=True)
        (args.work_dir / "extraction.log").write_text(process.stdout + process.stderr)
        require(process.returncode == 0, f"Geometry extraction failed; see {args.work_dir / 'extraction.log'}")
    with np.load(cache, allow_pickle=False) as data:
        sculpt = {name: data[name] for name in data.files}
    torch.set_num_threads(8)
    model = torch.jit.load(str(args.model), map_location="cpu").eval()
    compatibility = verify_compatibility(model, sculpt)
    parameters, frame_index, actual_time = load_pose(args.json.resolve(), args.time)
    with torch.no_grad():
        posed, baseline, state, rest, delta = skin_sculpt(model, parameters, sculpt["sculpt_delta_cm"], args.sculpt_strength)
        transfer = validate_transfer(model, parameters, sculpt, args.sculpt_strength, posed, baseline, state, rest, delta)
    vertices = (posed[0] / 100).numpy().astype(np.float64)
    joints = (state[0, :, :3] / 100).numpy().astype(np.float64)
    faces = sculpt["faces"]
    body_frame = direct.build_mhr_body_frame(joints)
    side = direct.select_shoulder_side(joints, args.side, args.movement)
    result = direct.compute_shoulder_flexion(joints, side, body_frame)
    if args.side == "right" and abs(args.time - 10.076) < 1e-6 and args.json.name == "s01_cam0_brett_adhesive_capsulitis.json":
        require(f"{result.angle_deg:+.1f}" == "+100.6", "Reference ROM measurement changed")
    report = {"renderer": "Native TorchScript MHR skinning + existing PyRender dark_fresnel",
              "blender_usage": "Read-only bind-space geometry extraction; no Blender rendering",
              "sculpt_scene": str(args.sculpt_scene.resolve()),
              "sculpt_keys": sculpt_keys, "definition": args.definition, "annotations": args.annotations,
              "sculpt_strength": args.sculpt_strength, "lean_shape": direct.LEAN_SHAPE.tolist(),
              "vertices": len(vertices), "triangles": len(faces), "frame_index": frame_index,
              "requested_time": args.time, "actual_time": actual_time, "angle_degrees": result.angle_deg,
              "compatibility": compatibility, "transfer": transfer,
              "geometry_method": "Add extracted sculpt to exact lean identity + facial expression + learned pose correctives BEFORE native skin_points",
              "protected_files": protected}
    np.savez_compressed(args.work_dir / "posed_geometry.npz", vertices=vertices,
                        lean_vertices=(baseline[0] / 100).numpy(), joints=joints, faces=faces,
                        skeleton_state=state[0].numpy(), neutral_delta_cm=sculpt["sculpt_delta_cm"])
    if not args.validate_only:
        if args.annotations == "skeleton":
            panels = skeleton_panels(vertices, joints, faces, result, body_frame, args.appearance,
                                      args.pose_overlay, args.panel_width, args.height)
        else:
            panels = direct._planar_panel_pair(vertices, joints, faces, result, body_frame, args.joint,
                                               args.movement, args.appearance, args.pose_overlay,
                                               args.panel_width, args.height)
        source = direct._source_panel(args.video.resolve(), frame_index, actual_time, args.time,
                                      args.panel_width, args.height)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        require(cv2.imwrite(str(args.output), cv2.hconcat([source, *panels])), "Failed to write final ROM PNG")
        report["output"] = str(args.output.resolve())
        if args.annotations == "skeleton":
            for label, image in (("front", panels[0]), ("side", panels[1]), ("mesh_only", cv2.hconcat(panels))):
                path = args.output.with_name(f"{args.output.stem}_{label}.png")
                require(cv2.imwrite(str(path), image), f"Failed to write {path}")
                report[f"output_{label}"] = str(path.resolve())
    require(all(digest(Path(path)) == expected for path, expected in protected.items()), "Protected input changed")
    (args.work_dir / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in {"protected_files", "lean_shape"}}, indent=2))


if __name__ == "__main__":
    main()
