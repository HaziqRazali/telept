"""Create a separate, reversible abs-only variant of the approved lean MHR."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
REPO_ROOT = ROOT.parents[3]
BODY_MODEL_ROOT = REPO_ROOT / "data" / "body_models" / "mhr"
SOURCE = BODY_MODEL_ROOT / "blender" / "stylized_mhr_lean_abs_outline.blend"
OUTPUT = BODY_MODEL_ROOT / "blender" / "stylized_mhr_lean_abs_enhanced.blend"
KEY_NAME = "Abs_Extra_Definition"
sys.dont_write_bytecode = True
sys.path.insert(0, str(PARENT))
from build_lean_abs import camera_view, coordinates, digest, evaluated, gaussian, signature, smoothstep, weights


def key_signatures(body):
    return {key.name: {"relative": key.relative_key.name, "value": key.value,
                       "coordinates_sha256": hashlib.sha256(coordinates(key.data).tobytes()).hexdigest()}
            for key in body.data.shape_keys.key_blocks}


def action_signature(rig):
    return [(curve.data_path, curve.array_index,
             [(point.co[:], point.handle_left[:], point.handle_right[:], point.interpolation)
              for point in curve.keyframe_points]) for curve in rig.animation_data.action.fcurves]


def scene_signature(scene, body, rig):
    return {
        "objects": sorted((obj.name, obj.type) for obj in scene.objects),
        "action": action_signature(rig),
        "body_matrix": [list(row) for row in body.matrix_world],
        "rig_matrix": [list(row) for row in rig.matrix_world],
        "modifiers": [(modifier.name, modifier.type, modifier.show_viewport, modifier.show_render,
                       getattr(modifier, "factor", None), getattr(modifier, "iterations", None),
                       getattr(modifier, "ratio", None), getattr(getattr(modifier, "object", None), "name", None))
                      for modifier in body.modifiers],
        "material_names": [material.name for material in body.data.materials],
        "material_inputs": [(socket.name, list(socket.default_value) if hasattr(socket.default_value, "__len__")
                             else socket.default_value)
                            for socket in body.data.materials[0].node_tree.nodes["Principled BSDF"].inputs
                            if hasattr(socket, "default_value")],
        "lights": [(obj.name, obj.data.energy, obj.data.size, [list(row) for row in obj.matrix_world])
                   for obj in scene.objects if obj.type == "LIGHT"],
        "color_management": [scene.view_settings.view_transform, scene.view_settings.look,
                             scene.view_settings.exposure, scene.view_settings.gamma],
        "auto_smooth": body.data.use_auto_smooth,
    }


def normals(vertices, faces):
    triangles = vertices[faces]
    return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    arguments = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    preserved = {str(path): digest(path) for path in PARENT.iterdir() if path.is_file()}
    preserved[str(SOURCE)] = digest(SOURCE)
    rom = PARENT.parent / "visualize_mhr_json_rom.py"
    if rom.exists():
        preserved[str(rom)] = digest(rom)
    bpy.ops.wm.open_mainfile(filepath=str(SOURCE))
    scene = bpy.context.scene
    body = bpy.data.objects["body_mesh"]
    rig = bpy.data.objects["body_world"]
    source_geometry = signature(body, rig)
    source_keys = key_signatures(body)
    source_scene = scene_signature(scene, body, rig)
    source_samples = {}
    for frame in (1, 40):
        scene.frame_set(frame)
        source_samples[frame] = evaluated(body)
    scene.frame_set(1)
    keys = body.data.shape_keys.key_blocks
    base = coordinates(keys[0].data)
    lean = base.copy()
    for key in keys[1:]:
        if key.name != "Lean_Abs_Definition":
            lean += key.value * (coordinates(key.data) - coordinates(key.relative_key.data))
    lateral, height, depth = lean.T
    abdominal = smoothstep(height, 101, 108) * (1 - smoothstep(height, 129, 137))
    anterior = smoothstep(depth, 2, 6) * (1 - smoothstep(np.abs(lateral), 10, 15))
    center = -0.65 * gaussian(lateral, 0, 1.6)
    paired = 0.85 * (gaussian(lateral, 3.8, 2.7) + gaussian(lateral, -3.8, 2.7))
    separations = -0.32 * (gaussian(height, 116.5, 1.5) + 0.75 * gaussian(height, 124, 1.5)) * gaussian(lateral, 0, 6)
    obliques = -0.28 * gaussian(np.abs(lateral), 8.2 + 0.10 * (height - 114), 2.2)
    extra = np.zeros_like(base)
    extra[:, 2] = 0.65 * (center + paired + separations + obliques) * abdominal * anterior
    extra_key = body.shape_key_add(name=KEY_NAME)
    extra_key.data.foreach_set("co", (base + extra).ravel())
    extra_key.value = 1
    body["Enhanced abs"] = "Abs_Extra_Definition: 0 restores approved variant, 1 adds 65% abdominal relief only."
    scene.frame_set(1)
    camera_view(scene, 0)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT))

    bpy.ops.wm.open_mainfile(filepath=str(OUTPUT))
    scene = bpy.context.scene
    body = bpy.data.objects["body_mesh"]
    rig = bpy.data.objects["body_world"]
    extra_key = body.data.shape_keys.key_blocks[KEY_NAME]
    assert signature(body, rig) == source_geometry
    current_keys = key_signatures(body)
    assert len(current_keys) == len(source_keys) + 1
    assert all(current_keys[name] == values for name, values in source_keys.items())
    assert scene_signature(scene, body, rig) == source_scene
    assert extra_key.value == 1
    assert not scene.render.use_freestyle
    assert not any(obj.type in {"CURVE", "GPENCIL"} for obj in scene.objects)
    arms = (weights(body, "l_uparm") + weights(body, "r_uparm")) > 0.5
    assert np.max(np.abs(extra[arms])) == 0
    frames = {}
    for frame in (1, 40):
        scene.frame_set(frame)
        enhanced, enhanced_faces = evaluated(body)
        extra_key.value = 0
        approved, approved_faces = evaluated(body)
        extra_key.value = 1
        source_vertices, source_faces = source_samples[frame]
        assert np.array_equal(approved_faces, source_faces)
        recovery = float(np.max(np.abs(approved - source_vertices)))
        assert recovery < 1e-5
        assert np.array_equal(enhanced_faces, approved_faces)
        displacement = np.linalg.norm(enhanced - approved, axis=1)
        arm_error = float(displacement[arms].max())
        assert arm_error < 1e-5
        assert 0.3 < displacement.max() < 0.6
        assert np.count_nonzero(displacement > 0.1) > 100
        assert np.isfinite(enhanced).all()
        flipped = int(np.count_nonzero(np.einsum("ij,ij->i", normals(approved, approved_faces),
                                               normals(enhanced, enhanced_faces)) <= 0))
        assert flipped == 0
        frames[str(frame)] = {"extra_max_mm": float(displacement.max() * 10),
                              "vertices_changed_over_1mm": int(np.count_nonzero(displacement > 0.1)),
                              "approved_recovery_error_cm": recovery, "biceps_error_cm": arm_error,
                              "flipped_triangles": flipped}
    report = {"source": str(SOURCE), "output": str(OUTPUT), "source_sha256": preserved[str(SOURCE)],
              "mesh_weights_bones_unchanged": True, "original_keys_unchanged": len(source_keys),
              "material_lighting_modifiers_action_unchanged": True,
              "additional_shape_key": KEY_NAME, "extra_abs_fraction": 0.65,
              "extra_max_raw_mm": float(np.linalg.norm(extra, axis=1).max() * 10),
              "frame_checks": frames, "vertices": len(body.data.vertices),
              "polygons": len(body.data.polygons), "preserved_files": preserved}
    (ROOT / "validation.json").write_text(json.dumps(report, indent=2))
    views = [("front_t_pose", 1, 0, False), ("three_quarter_t_pose", 1, 45, False),
             ("side_t_pose", 1, 90, False), ("back_t_pose", 1, 180, False),
             ("shoulder_flexion", 40, 35, False), ("torso_front", 1, 0, True),
             ("torso_three_quarter", 1, 35, True), ("torso_flexion", 40, 35, True)]
    if arguments.preview:
        views = [("preview_torso_front", 1, 0, True), ("preview_torso_three_quarter", 1, 35, True)]
    for label, frame, angle, close in views:
        scene.frame_set(frame)
        camera_view(scene, angle, close)
        for state, value in [("approved", 0), ("enhanced", 1)]:
            extra_key.value = value
            scene.render.filepath = str(ROOT / f"{label}_{state}.png")
            bpy.ops.render.render(write_still=True)
    assert all(digest(Path(path)) == value for path, value in preserved.items())
    print("ENHANCED_CHECK=" + json.dumps({key: value for key, value in report.items() if key != "preserved_files"}))
    print("PRESERVED_FILES=" + str(len(preserved)))


if __name__ == "__main__":
    main()
