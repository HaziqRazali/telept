"""Export attached sculpt layers in MHR bind coordinates; never render or save a blend."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np


SCULPT_KEYS = ("Lean_Abs_Definition", "Abs_Extra_Definition")


def coordinates(points):
    values = np.empty(len(points) * 3, dtype=np.float32)
    points.foreach_get("co", values)
    return values.reshape(-1, 3)


def extract(scene_path, output, sculpt_keys=SCULPT_KEYS):
    source_hash = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    body = bpy.data.objects["body_mesh"]
    rig = bpy.data.objects["body_world"]
    keys = body.data.shape_keys.key_blocks
    if (len(body.data.vertices), len(body.data.polygons)) != (18439, 36874):
        raise ValueError("Sculpt must retain MHR LOD1 base topology")
    if any(len(face.vertices) != 3 for face in body.data.polygons):
        raise ValueError("Sculpt faces must all be triangles")
    armatures = [modifier for modifier in body.modifiers if modifier.type == "ARMATURE"]
    if len(armatures) != 1 or armatures[0].object != rig:
        raise ValueError("Expected the existing body_world armature binding")
    if any(obj.type in {"CURVE", "GPENCIL"} for obj in bpy.context.scene.objects):
        raise ValueError("Unexpected curve or outline objects in sculpt scene")
    extra_visible = [obj.name for obj in bpy.context.scene.objects
                     if obj.type == "MESH" and obj != body and not obj.hide_render]
    if extra_visible:
        raise ValueError(f"Unexpected visible separate geometry: {extra_visible}")
    offsets = []
    for name in sculpt_keys:
        key = keys.get(name)
        if key is None or key.value != 1 or key.mute or key.vertex_group:
            raise ValueError(f"Expected active, unmasked sculpt key: {name}")
        offsets.append(coordinates(key.data) - coordinates(key.relative_key.data))
    skin_weights = np.zeros((len(body.data.vertices), len(body.vertex_groups)), dtype=np.float32)
    for vertex in body.data.vertices:
        for group in vertex.groups:
            skin_weights[vertex.index, group.group] = group.weight
    weight_sums = skin_weights.sum(axis=1)
    if not np.allclose(weight_sums, 1, atol=1e-5):
        raise ValueError("Sculpt has unweighted or nonnormalized vertices")
    report = {
        "scene": str(scene_path), "scene_sha256": source_hash,
        "body": body.name, "armature": rig.name, "bones": len(rig.data.bones),
        "vertices": len(body.data.vertices), "triangles": len(body.data.polygons),
        "vertex_groups": [group.name for group in body.vertex_groups],
        "weight_sum_range": [float(weight_sums.min()), float(weight_sums.max())],
        "shape_keys": [{"name": key.name, "value": key.value, "relative": key.relative_key.name}
                       for key in keys],
        "sculpt_keys_extracted": list(sculpt_keys), "mesh_attached": True,
        "detached_curves": [], "extra_visible_meshes": extra_visible,
        "other_meshes": [{"name": obj.name, "hidden_render": obj.hide_render}
                         for obj in bpy.context.scene.objects if obj.type == "MESH" and obj != body],
        "modifiers_not_applied": [{"name": modifier.name, "type": modifier.type}
                                  for modifier in body.modifiers],
        "coordinate_space": "Unmodified mesh-local MHR bind coordinates, centimeters",
        "excluded": "Artist proportions, smoothing, decimation, materials, Blender pose and renderer",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, basis_cm=coordinates(keys[0].data),
                        faces=np.array([face.vertices[:] for face in body.data.polygons], dtype=np.int32),
                        sculpt_delta_cm=np.sum(offsets, axis=0),
                        sculpt_layers_cm=np.stack(offsets), skin_weights=skin_weights,
                        group_names=np.array(report["vertex_groups"]),
                        metadata=np.array(json.dumps(report)))
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    assert hashlib.sha256(scene_path.read_bytes()).hexdigest() == source_hash
    print("SCULPT_EXPORTED=" + json.dumps({key: report[key] for key in
          ("scene", "body", "armature", "vertices", "triangles", "bones", "mesh_attached")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sculpt-keys", nargs="+", choices=SCULPT_KEYS, default=SCULPT_KEYS)
    arguments = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    extract(arguments.scene.resolve(), arguments.output.resolve(), arguments.sculpt_keys)


if __name__ == "__main__":
    main()