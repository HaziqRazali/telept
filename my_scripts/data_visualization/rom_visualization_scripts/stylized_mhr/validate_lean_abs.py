"""Independently reopen source and deliverable and verify mesh/rig preservation."""
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
BODY_MODEL_ROOT = REPO_ROOT / "data" / "body_models" / "mhr"
sys.path.insert(0, str(ROOT))
from build_lean_abs import coordinates, evaluated, reset_pose, rotate_world, signature, weights


def key_snapshot(body):
    return {key.name: {"relative": key.relative_key.name, "value": key.value,
                       "coordinates": coordinates(key.data)}
            for key in body.data.shape_keys.key_blocks}


def action_snapshot(action):
    return [(curve.data_path, curve.array_index,
             [(point.co[:], point.handle_left[:], point.handle_right[:], point.interpolation)
              for point in curve.keyframe_points]) for curve in action.fcurves]


def edge_stats(faces):
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    unique, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
    return {"edges": len(unique), "boundary_edges": int(np.count_nonzero(counts == 1)),
            "nonmanifold_edges": int(np.count_nonzero(counts > 2))}


def normal_vectors(vertices, faces):
    triangles = vertices[faces]
    return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])


bpy.ops.wm.open_mainfile(filepath=str(BODY_MODEL_ROOT / "blender" / "stylized_mhr.blend"))
source_body = bpy.data.objects["body_mesh"]
source_rig = bpy.data.objects["body_world"]
source_signature = signature(source_body, source_rig)
source_keys = key_snapshot(source_body)
source_action_name = source_rig.animation_data.action.name
source_action = action_snapshot(source_rig.animation_data.action)
source_objects = {obj.name: obj.type for obj in bpy.context.scene.objects}
source_materials = [material.name for material in source_body.data.materials]
source_matrices = {obj.name: np.array(obj.matrix_world) for obj in (source_body, source_rig)}
source_edges = edge_stats(np.array(source_signature["faces"]))

bpy.ops.wm.open_mainfile(filepath=str(BODY_MODEL_ROOT / "blender" / "stylized_mhr_lean_abs_outline.blend"))
scene = bpy.context.scene
body = bpy.data.objects["body_mesh"]
rig = bpy.data.objects["body_world"]
before = bpy.data.objects["Original_Body_Comparison"]
assert signature(body, rig) == source_signature
assert signature(before, rig) == source_signature
for obj in (body, rig):
    assert np.allclose(np.array(obj.matrix_world), source_matrices[obj.name], atol=1e-8, rtol=0)
for obj in (body, before):
    current_keys = key_snapshot(obj)
    for name, values in source_keys.items():
        assert current_keys[name]["relative"] == values["relative"]
        assert current_keys[name]["value"] == values["value"]
        assert np.array_equal(current_keys[name]["coordinates"], values["coordinates"])
assert action_snapshot(bpy.data.actions[source_action_name]) == source_action
assert [material.name for material in before.data.materials] == source_materials
assert before.hide_render and before.hide_get()
assert before.data.use_auto_smooth and not body.data.use_auto_smooth
assert {obj.name: obj.type for obj in scene.objects if obj != before} == source_objects
assert not scene.render.use_freestyle
assert not any(obj.type in {"CURVE", "GPENCIL"} for obj in scene.objects)
assert all(obj.hide_render for obj in scene.objects if obj.type == "MESH" and obj != body)
assert edge_stats(np.array([face.vertices[:] for face in body.data.polygons])) == source_edges
assert all(sum(group.weight for group in vertex.groups) > 0.999 for vertex in body.data.vertices)
keys = body.data.shape_keys.key_blocks
assert keys["Lean_Proportions"].value == 1 and keys["Lean_Abs_Definition"].value == 1
assert scene.frame_current == 1

attachment = {}
base_coordinates = coordinates(body.data.vertices)
regions = {
    "abdomen": ((np.abs(base_coordinates[:, 0]) < 12) & (base_coordinates[:, 1] > 101)
                & (base_coordinates[:, 1] < 137) & (base_coordinates[:, 2] > 5)),
    "left_biceps": weights(body, "l_uparm") > 0.5,
    "right_biceps": weights(body, "r_uparm") > 0.5,
}
for frame in (1, 40):
    scene.frame_set(frame)
    defined, faces = evaluated(body)
    keys["Lean_Abs_Definition"].value = 0
    plain, plain_faces = evaluated(body)
    keys["Lean_Abs_Definition"].value = 1
    assert np.array_equal(faces, plain_faces)
    assert np.isfinite(defined).all()
    delta = np.linalg.norm(defined - plain, axis=1)
    normals_plain = normal_vectors(plain, faces)
    normals_defined = normal_vectors(defined, faces)
    flipped = int(np.count_nonzero(np.einsum("ij,ij->i", normals_plain, normals_defined) <= 0))
    assert flipped == 0
    assert delta.max() < 1.0 and np.count_nonzero(delta > 0.001) > 100
    region_details = {}
    for name, mask in regions.items():
        maximum = float(delta[mask].max())
        assert 0.3 < maximum < 1.0, (frame, name, maximum)
        region_details[name] = {"max_evaluated_detail_mm": maximum * 10,
                                "vertices_over_1mm": int(np.count_nonzero(delta[mask] > 0.1))}
    attachment[str(frame)] = {"max_evaluated_detail_mm": float(delta.max() * 10),
                             "detail_vertices": int(np.count_nonzero(delta > 0.001)),
                             "flipped_detail_triangles": flipped,
                             "regions": region_details,
                             "vertices": len(defined), "polygons": len(faces)}
scene.frame_set(1)
t_directions = {}
for side, sign in [("l_", 1), ("r_", -1)]:
    for segment, endpoint in [("uparm", "lowarm"), ("lowarm", "wrist")]:
        direction = (rig.pose.bones[side + endpoint].head - rig.pose.bones[side + segment].head).normalized()
        assert abs(direction.x - sign) < 1e-5 and abs(direction.y) < 1e-5 and abs(direction.z) < 1e-5
        t_directions[side + segment] = list(direction)
scene.frame_set(40)
direction = (rig.pose.bones["r_lowarm"].head - rig.pose.bones["r_uparm"].head).normalized()
flexion_angle = float(np.degrees(np.arccos(np.clip(-direction.y, -1, 1))))
assert abs(flexion_angle - 135) < 0.01 and direction.z > 0

rig.animation_data.action = None
reset_pose(rig)
neutral, neutral_faces = evaluated(body)
joint_checks = {}
for side in ("l_", "r_"):
    for segment in ("uparm", "lowarm", "upleg", "lowleg"):
        for degrees in (-30, 30):
            reset_pose(rig)
            rotate_world(rig, side + segment, (1, 0, 0), degrees)
            moved, moved_faces = evaluated(body)
            assert np.array_equal(neutral_faces, moved_faces)
            delta = np.linalg.norm(moved - neutral, axis=1)
            assert delta.max() > 1 and np.count_nonzero(delta > 0.01) > 100
            joint_checks[f"{side}{segment}_{degrees:+}"] = {
                "moved_vertices": int(np.count_nonzero(delta > 0.01)),
                "max_movement_cm": float(delta.max())}
reset_pose(rig)
assert signature(body, rig) == source_signature
protected = json.loads((ROOT / "lean_abs_build_validation.json").read_text())["protected_files"]
for filename, expected in protected.items():
    assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == expected, filename
report = {"passed": True, "original_vertex_positions_indices_topology_weights_bones_exact": True,
          "original_shape_keys_exact": len(source_keys), "original_action_retained_exact": True,
          "original_materials_retained_on_hidden_body": True,
          "new_objects": [before.name], "no_accent_geometry": True,
          "topology": source_edges, "attachment_checks": attachment,
          "t_pose_segment_directions": t_directions, "right_shoulder_flexion_degrees": flexion_angle,
          "isolated_joint_checks": joint_checks, "protected_files_unchanged": len(protected)}
(ROOT / "lean_abs_saved_validation.json").write_text(json.dumps(report, indent=2))
print("SAVED_VALIDATION=" + json.dumps({key: value for key, value in report.items()
                                      if key not in {"isolated_joint_checks", "t_pose_segment_directions"}}))
