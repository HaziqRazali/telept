"""Reversible, mesh-only lean refinement of the original stylized MHR scene."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[3]
BODY_MODEL_ROOT = REPO_ROOT / "data" / "body_models" / "mhr"
SOURCE = BODY_MODEL_ROOT / "blender" / "stylized_mhr.blend"
OUTPUT = BODY_MODEL_ROOT / "blender" / "stylized_mhr_lean_abs_outline.blend"


def coordinates(points):
    values = np.empty(len(points) * 3, dtype=np.float64)
    points.foreach_get("co", values)
    return values.reshape(-1, 3)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def smoothstep(values, lower, upper):
    fraction = np.clip((values - lower) / (upper - lower), 0, 1)
    return fraction * fraction * (3 - 2 * fraction)


def gaussian(values, center, width):
    return np.exp(-0.5 * ((values - center) / width) ** 2)


def weights(body, prefix):
    indices = {group.index for group in body.vertex_groups
               if group.name == prefix or group.name.startswith(prefix + "_twist")}
    return np.array([sum(group.weight for group in vertex.groups if group.group in indices)
                     for vertex in body.data.vertices])


def signature(body, rig):
    return {
        "vertices": coordinates(body.data.vertices).tolist(),
        "faces": [list(face.vertices) for face in body.data.polygons],
        "groups": [group.name for group in body.vertex_groups],
        "weights": [[(group.group, group.weight) for group in vertex.groups]
                    for vertex in body.data.vertices],
        "bones": {bone.name: {"parent": bone.parent.name if bone.parent else None,
                              "matrix": [list(row) for row in bone.matrix_local]}
                  for bone in rig.data.bones},
    }


def evaluated(body):
    bpy.context.view_layer.update()
    result = body.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = result.to_mesh()
    values = coordinates(mesh.vertices)
    faces = np.array([face.vertices[:] for face in mesh.polygons])
    result.to_mesh_clear()
    return values, faces


def reset_pose(rig):
    for bone in rig.pose.bones:
        bone.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()


def rotate_world(rig, name, axis, degrees):
    bone = rig.pose.bones[name]
    local_axis = (rig.matrix_world.to_3x3() @ bone.bone.matrix_local.to_3x3()).inverted() @ Vector(axis)
    bone.rotation_mode = "QUATERNION"
    bone.rotation_quaternion = Quaternion(local_axis.normalized(), math.radians(degrees))
    bpy.context.view_layer.update()


def aim_segment(rig, name, endpoint, direction):
    bone = rig.pose.bones[name]
    current = rig.pose.bones[endpoint].head - bone.head
    rotation = current.normalized().rotation_difference(Vector(direction).normalized())
    pivot = Matrix.Translation(bone.head)
    bone.matrix = pivot @ rotation.to_matrix().to_4x4() @ pivot.inverted() @ bone.matrix
    bpy.context.view_layer.update()


def t_pose(rig):
    reset_pose(rig)
    for side, sign in [("l_", 1), ("r_", -1)]:
        aim_segment(rig, side + "uparm", side + "lowarm", (sign, 0, 0))
        aim_segment(rig, side + "lowarm", side + "wrist", (sign, 0, 0))


def flexion(rig):
    t_pose(rig)
    aim_segment(rig, "r_uparm", "r_lowarm", (0, 1, 1))
    aim_segment(rig, "r_lowarm", "r_wrist", (0, 1, 1))


def anatomy_material():
    material = bpy.data.materials.new("Lean anatomy - soft blue gray")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (0.105, 0.155, 0.205, 1)
    shader.inputs["Roughness"].default_value = 0.78
    shader.inputs["Specular"].default_value = 0.22
    material.diffuse_color = (0.105, 0.155, 0.205, 1)
    return material


def camera_view(scene, angle, close=False):
    radians = math.radians(angle)
    target = Vector((0, 0, 1.20 if close else (1.00 if scene.frame_current == 40 else 0.86)))
    if close == "arm":
        rig = bpy.data.objects["body_world"]
        target = rig.matrix_world @ ((rig.pose.bones["r_uparm"].head + rig.pose.bones["r_lowarm"].head) * 0.5)
    camera = scene.camera
    camera.location = target + Vector((4 * math.sin(radians), -4 * math.cos(radians), 0))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.ortho_scale = 0.86 if close else (2.25 if scene.frame_current == 40 else 2.02)
    if close == "arm":
        camera.data.ortho_scale = 0.65


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true")
    arguments = parser.parse_args(__import__("sys").argv[__import__("sys").argv.index("--") + 1:]
                                  if "--" in __import__("sys").argv else [])
    protected = {str(path): digest(path) for path in ROOT.iterdir()
                 if path.is_file() and not path.name.startswith(("lean_abs_", "build_lean_abs", "validate_lean_abs", "stylized_mhr_lean_abs_outline.blend"))}
    for path in [ROOT.parent / "visualize_mhr_json_rom.py", ROOT / "render_stylized_mhr_json_rom.py"]:
        if path.exists():
            protected[str(path)] = digest(path)
    bpy.ops.wm.open_mainfile(filepath=str(SOURCE))
    scene = bpy.context.scene
    body = bpy.data.objects["body_mesh"]
    rig = bpy.data.objects["body_world"]
    original = signature(body, rig)
    keys = body.data.shape_keys.key_blocks
    base = coordinates(keys[0].data)
    active = base.copy()
    for key in keys[1:]:
        active += key.value * (coordinates(key.data) - coordinates(key.relative_key.data))
    inspection = {
        "source": str(SOURCE), "source_sha256": digest(SOURCE),
        "body": body.name, "armature": rig.name, "frame": scene.frame_current,
        "vertices": len(body.data.vertices), "polygons": len(body.data.polygons),
        "bones": len(rig.data.bones), "vertex_groups": [group.name for group in body.vertex_groups],
        "weight_sum_range": [min(sum(group.weight for group in vertex.groups) for vertex in body.data.vertices),
                             max(sum(group.weight for group in vertex.groups) for vertex in body.data.vertices)],
        "materials": [material.name for material in body.data.materials],
        "modifiers": [{"name": modifier.name, "type": modifier.type,
                       "ratio": getattr(modifier, "ratio", None),
                       "factor": getattr(modifier, "factor", None),
                       "iterations": getattr(modifier, "iterations", None)} for modifier in body.modifiers],
        "shape_keys": [{"name": key.name, "value": key.value, "relative_key": key.relative_key.name,
                        "max_delta_cm": float(np.linalg.norm(coordinates(key.data) - base, axis=1).max())}
                       for key in keys],
        "nonidentity_pose_bones": [bone.name for bone in rig.pose.bones
                                   if not np.allclose(np.array(bone.matrix_basis), np.eye(4), atol=1e-6)],
        "action": rig.animation_data.action.name if rig.animation_data and rig.animation_data.action else None,
        "lean_parameter_mapping": "Not verified. Imported shape_* names are not a documented MHR identity interface. Reference is visual only.",
        "reference_coefficients_not_applied": [-1.5, -0.75, 0.75],
    }
    (ROOT / "lean_abs_inspection.json").write_text(json.dumps(inspection, indent=2))
    before = body.copy()
    before.data = body.data.copy()
    before.name = "Original_Body_Comparison"
    scene.collection.objects.link(before)
    before.hide_render = True
    before.hide_set(True)
    for material in before.data.materials:
        material.use_fake_user = True
    male = next(key for key in keys if key.name.startswith("Male proportions"))
    lateral, height, depth = active.T
    torso = (1 - smoothstep(np.abs(lateral), 22, 35)) * smoothstep(height, 68, 87) * (1 - smoothstep(height, 146, 156))
    lean = active - (coordinates(male.data) - base) * torso[:, None]
    width_reduction = (0.08 * gaussian(height, 110, 11) + 0.065 * gaussian(height, 94, 9)
                       + 0.025 * gaussian(height, 134, 10)) * torso
    lean[:, 0] *= 1 - width_reduction
    lean[:, 2] *= 1 - (0.095 * gaussian(height, 108, 15) + 0.035 * gaussian(height, 91, 10)) * torso
    front = smoothstep(depth, 0, 6)
    lean[:, 2] -= 0.65 * gaussian(height, 110, 10) * gaussian(lateral, 0, 10) * front * torso
    transform = body.matrix_world.inverted() @ rig.matrix_world
    for side in ("l_", "r_"):
        for segment, endpoint, amount in [("uparm", "lowarm", -0.07), ("lowarm", "wrist", -0.045),
                                           ("upleg", "lowleg", -0.035), ("lowleg", "talocrural", -0.035)]:
            start = np.array(transform @ rig.data.bones[side + segment].head_local)
            end = np.array(transform @ rig.data.bones[side + endpoint].head_local)
            direction = (end - start) / np.linalg.norm(end - start)
            radial = lean - start - np.outer((lean - start) @ direction, direction)
            lean += radial * (amount * weights(body, side + segment))[:, None]
    edges = np.array([edge.vertices[:] for edge in body.data.edges])
    sources = np.concatenate((edges[:, 0], edges[:, 1]))
    targets = np.concatenate((edges[:, 1], edges[:, 0]))
    degree = np.bincount(sources, minlength=len(lean)).clip(1)[:, None]
    smoothing = 0.35 * torso
    for side in ("l_", "r_"):
        smoothing = np.maximum(smoothing, 0.2 * weights(body, side + "uparm"))
    for iteration in range(6):
        neighbors = np.zeros_like(lean)
        np.add.at(neighbors, sources, lean[targets])
        lean += (neighbors / degree - lean) * smoothing[:, None]
    lean_key = body.shape_key_add(name="Lean_Proportions")
    lean_key.data.foreach_set("co", (base + lean - active).ravel())
    lean_key.value = 1
    detail = np.zeros_like(lean)
    lateral, height, depth = lean.T
    abdominal = smoothstep(height, 101, 108) * (1 - smoothstep(height, 129, 137))
    anterior = smoothstep(depth, 2, 6) * (1 - smoothstep(np.abs(lateral), 10, 15))
    center = -0.65 * gaussian(lateral, 0, 1.6)
    paired = 0.85 * (gaussian(lateral, 3.8, 2.7) + gaussian(lateral, -3.8, 2.7))
    separations = -0.32 * (gaussian(height, 116.5, 1.5) + 0.75 * gaussian(height, 124, 1.5)) * gaussian(lateral, 0, 6)
    obliques = -0.28 * gaussian(np.abs(lateral), 8.2 + 0.10 * (height - 114), 2.2)
    detail[:, 2] = (center + paired + separations + obliques) * abdominal * anterior
    for side in ("l_", "r_"):
        start = np.array(transform @ rig.data.bones[side + "uparm"].head_local)
        end = np.array(transform @ rig.data.bones[side + "lowarm"].head_local)
        direction = (end - start) / np.linalg.norm(end - start)
        along = (lean - start) @ direction
        radial = lean - start - np.outer(along, direction)
        radial_length = np.linalg.norm(radial, axis=1).clip(1e-8)
        influence = weights(body, side + "uparm")
        belly = gaussian(along, np.linalg.norm(end - start) * 0.56, 5.5)
        anterior_arm = smoothstep(radial[:, 2] / radial_length, -0.35, 0.7)
        detail += radial / radial_length[:, None] * (0.70 * belly * influence * anterior_arm)[:, None]
    definition = body.shape_key_add(name="Lean_Abs_Definition")
    definition.data.foreach_set("co", (base + detail).ravel())
    definition.value = 1
    body["Lean reference"] = "Visual approximation; MHR coefficients [-1.5, -0.75, +0.75] NOT mapped or applied."
    body["Detail method"] = "Additive body shape keys, before original weighted armature; no accent geometry."
    body["Original appearance"] = "Hidden Original_Body_Comparison retains original mesh, keys, materials, and modifiers."
    for modifier in body.modifiers:
        if modifier.type == "DECIMATE":
            modifier.show_viewport = False
            modifier.show_render = False
        elif modifier.type == "SMOOTH":
            modifier.factor = 0.25
            modifier.iterations = 2
    material = anatomy_material()
    body.data.use_auto_smooth = False
    body.data.materials.clear()
    body.data.materials.append(material)
    for polygon in body.data.polygons:
        polygon.material_index = 0
    hidden = []
    for obj in scene.objects:
        if obj.type == "MESH" and obj != body:
            obj.hide_render = True
            obj.hide_set(True)
            hidden.append(obj.name)
    source_action = rig.animation_data.action
    source_action.use_fake_user = True
    rig.animation_data.action = bpy.data.actions.new("Lean anatomy - T pose and right shoulder flexion")
    t_pose(rig)
    for bone in rig.pose.bones:
        bone.rotation_mode = "QUATERNION"
        bone.keyframe_insert("rotation_quaternion", frame=1)
        bone.keyframe_insert("location", frame=1)
        bone.keyframe_insert("scale", frame=1)
    flexion(rig)
    for bone in rig.pose.bones:
        bone.keyframe_insert("rotation_quaternion", frame=40)
        bone.keyframe_insert("location", frame=40)
        bone.keyframe_insert("scale", frame=40)
    for marker in list(scene.timeline_markers):
        scene.timeline_markers.remove(marker)
    scene.timeline_markers.new("T pose", frame=1)
    scene.timeline_markers.new("Right shoulder flexion - manual armature test", frame=40)
    scene.frame_set(1)
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.use_gtao = False
    scene.eevee.taa_render_samples = 96
    scene.render.use_freestyle = False
    scene.world = scene.world.copy()
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.16, 0.18, 0.21, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.35
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0
    for name, energy, size in [("Broad key", 650, 3.0), ("Soft fill", 210, 4.0), ("Rear fill", 400, 3.0)]:
        light = bpy.data.objects[name]
        light.data.energy = energy
        light.data.size = size
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 65 if arguments.preview else 100
    camera_view(scene, 0)
    assert signature(body, rig) == original, "Source mesh, vertex order, weights, or bones changed"
    source_faces = np.array([face.vertices[:] for face in body.data.polygons])
    def normals(values):
        corners = values[source_faces]
        return np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    old_normals = normals(active)
    new_normals = normals(lean + detail)
    flipped = int(np.sum(np.einsum("ij,ij->i", old_normals, new_normals) <= 0))
    assert flipped == 0, f"New flipped triangles: {flipped}"
    neutral, neutral_faces = evaluated(body)
    scene.frame_set(40)
    posed, posed_faces = evaluated(body)
    assert np.array_equal(neutral_faces, posed_faces)
    movement = np.linalg.norm(posed - neutral, axis=1)
    assert movement.max() > 20 and np.count_nonzero(movement > 0.1) > 300
    scene.frame_set(1)
    report = {"base_mesh_weights_bones_unchanged": True, "flipped_rest_triangles": flipped,
              "new_shape_keys": [lean_key.name, definition.name],
              "max_proportion_delta_cm": float(np.linalg.norm(lean - active, axis=1).max()),
              "max_definition_delta_mm": float(np.linalg.norm(detail, axis=1).max() * 10),
              "evaluated_vertices": len(neutral), "evaluated_polygons": len(neutral_faces),
              "shoulder_test_moved_vertices": int(np.count_nonzero(movement > 0.1)),
              "hidden_objects": hidden, "protected_files": protected,
              "original_action_retained": source_action.name,
              "render_notes": "Before and after use identical blue-gray material, lighting, cameras, and modifier settings. Before disables only the two new shape keys. Clothing is a material region, bypassed on working body, retained on hidden original."}
    (ROOT / "lean_abs_build_validation.json").write_text(json.dumps(report, indent=2))
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.select_all(action="DESELECT")
    body.hide_set(False)
    body.select_set(True)
    scene.render.resolution_percentage = 100
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT))
    views = [("front_t_pose", 1, 0, False), ("three_quarter_t_pose", 1, 45, False),
             ("side_t_pose", 1, 90, False), ("back_t_pose", 1, 180, False),
             ("shoulder_flexion", 40, 35, False), ("torso_front", 1, 0, True),
             ("torso_three_quarter", 1, 35, True), ("biceps_flexion", 40, -35, "arm")]
    if arguments.preview:
        views = [("preview_front", 1, 0, False), ("preview_torso", 1, 30, True),
                 ("preview_torso_front", 1, 0, True),
                 ("preview_flexion", 40, 35, False), ("preview_biceps", 40, -35, "arm")]
    for label, frame, angle, close in views:
        scene.frame_set(frame)
        camera_view(scene, angle, close)
        scene.render.resolution_percentage = 100
        states = [("before", 0, 0), ("after", 1, 1)]
        if close:
            states.append(("lean_only", 1, 0))
        for state, lean_value, definition_value in states:
            lean_key.value = lean_value
            definition.value = definition_value
            scene.render.filepath = str(ROOT / f"lean_abs_{label}_{state}.png")
            bpy.ops.render.render(write_still=True)
    assert all(digest(Path(path)) == value for path, value in protected.items())
    print("LEAN_ABS_CHECK=" + json.dumps({key: value for key, value in report.items() if key != "protected_files"}))


if __name__ == "__main__":
    main()
