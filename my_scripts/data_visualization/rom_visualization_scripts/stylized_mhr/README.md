# Pastel MHR Exercise Character

The canonical editable scene is
`/home/haziq/datasets/telept/data/body_models/mhr/blender/stylized_mhr.blend`.
Open it in Blender 3.6.17. Frame 1 is the original reference pose with male
proportions; frame 40 is the original asymmetric march/reach test. The body is
the articulated Momentum Human Rig mesh, not a replacement mannequin or baked
pose. The former duplicate beside these ROM scripts is archived under
`/home/haziq/datasets/telept/data/body_models/mhr/archive/`.

Select `body_world`, enter Pose Mode, and rotate its bones. Both stored poses use the same weighted body. Edit or unlink the existing action when authoring poses so timeline changes do not restore its keys. Use Rendered viewport shading with scene lights/world to see the final EEVEE materials; Solid view does not show the procedural clothing.

## Visual Changes

- Dusty coral top `#D98A8C`, warm peach skin `#F1C7AC`, blue-charcoal pants `#515967`, dark brown hair `#3D3433`, and warm off-white background `#F5F4F1`. Hex values are converted from sRGB to linear shader inputs.
- Smooth, bounded diffuse tonal shading, three broad area lights, no specular or metallic shader, and the existing orthographic framing. Individual renders contain only the actual 3D character and background, with no cards or text overlays.
- Rounded front neckline, higher curved rear neckline, curved armholes, and clean waist/ankle boundaries. Rest-position coordinates are captured after smoothing/decimation and before the armature, so clothing stays attached without the previous noisy hem. The shirt reaches 98 cm and pants reach 102 cm in rest space, keeping a 4 cm overlap.
- The original male proportion key is unchanged. A new editable `Soft tailoring and rounded anatomy` key softens torso/face/hand/foot surface detail and gently fills out arms and calves using the existing limb twist influences. Its maximum source-space adjustment is about 3.98 cm. Disable both artist keys to inspect the original Basis.
- A scalp-derived, smoothly clipped hair cap retains full `c_head` weighting and shares the MHR armature. Its simplification happens before skinning, keeping topology stable during head rotations.

## What Is Generated

The inspected original saved scene matched the existing builder: MHR FBX body
and rig, imported shape keys, male adjustment, weighted scalp proxy, materials,
two keyed poses, camera, and light. No separate hand-authored character asset
was found only in the blend. The reusable FBX is now at
`/home/haziq/datasets/telept/data/body_models/mhr/native/lod1.fbx`.

[build_character.py](build_character.py) continues to import the same local `lod1.fbx`, preserve its topology and weights, and rebuild the editable scene. [render_angles.py](render_angles.py) opens the saved blend and renders it without changing it. [contact_sheet.py](contact_sheet.py) assembles the inspection sheets, not the character itself.

## Validation

[saved_scene_validation.json](saved_scene_validation.json) records a direct comparison with the untouched backup and tests through the final evaluated modifier stack:

- All 126 bones, their parents/bind matrices, body/rig transforms, original mesh topology, vertex groups, weights, 237 pre-existing shape keys, and stored pose keyframes are unchanged.
- Original body: 18,439 vertices / 36,874 triangles. Evaluated body: 4,426 vertices / 8,848 triangles. Evaluated hair: 3,019 vertices / 5,678 triangles. Body and hair remain skinned; no disconnected replacement body parts were introduced.
- Sixteen isolated rotations: left/right shoulder, elbow, hip, and knee, each at -30 and +30 degrees. Each moves the appropriate region; unaffected vertices remain stationary. Evaluated topology, skin weights, and clothing coordinates remain identical between poses. No unweighted body vertices.
- A 30-degree head rotation carries the hair with a maximum rigid-attachment error below 0.000001 m.
- [validation.json](validation.json) retains the original source-topology joint checks and comparison with the previously cached zero-input TorchScript mesh/hierarchy. The maximum Basis discrepancy remains approximately 0.000057 cm. TorchScript inference was not rerun for this styling update.

The twelve-view contact sheet, before/after sheet, and representative positive/negative side views were visually inspected. Both poses stay framed, with a covered abdomen and coherent front/rear silhouettes. [output_audit.json](output_audit.json) records file hashes, render framing/nonblank checks, and the unchanged ROM-script checksum.

## Renders And Backups

All outputs are in this directory. The current [contact_sheet.png](contact_sheet.png) contains the twelve requested 720 x 840 views. [comparison_sheet.png](comparison_sheet.png) shows matching-camera before/after renders from the backup and final character.

For both `neutral_` and `test_`, the requested suffixes are:

| Azimuth | Suffix |
| --- | --- |
| 0 | `front_000.png` |
| +45 | `azimuth_plus045.png` |
| -45 | `azimuth_minus045.png` |
| +135 | `azimuth_plus135.png` |
| -135 | `azimuth_minus135.png` |
| 180 | `rear_180.png` |

Additional side checks use `side_plus090.png` and `side_minus090.png` for each pose. The existing `neutral_front.png`, `neutral_three_quarter.png`, `test_front.png`, and `test_three_quarter.png` were also refreshed; their three-quarter angle remains +40 degrees. `preview_*.png` are intermediate 60%-resolution styling previews, not the final delivery. [render_inventory.json](render_inventory.json) lists all requested and side renders with camera metadata.

Every pre-existing working file was copied to `backup_before_polish/` before edits. No original working file was deleted. Blender's `stylized_mhr.blend1` is only the latest automatic backup; use the explicit backup directory for the untouched starting point. [file_changes.json](file_changes.json) lists every modified, added, and unchanged top-level file by comparison with that backup.

## Commands

Final build and verification commands, using only existing local installations:

```bash
cd /home/haziq/datasets/telept/my_scripts/data_visualization/rom_visualization_scripts/stylized_mhr
/home/haziq/blender-3.6.17-linux-x64/blender -b --python-exit-code 1 --python build_character.py > build.log 2>&1
/home/haziq/blender-3.6.17-linux-x64/blender -b --python-exit-code 1 --python validate_saved_character.py > saved_scene_validation.log 2>&1
/home/haziq/blender-3.6.17-linux-x64/blender -b --python-exit-code 1 --python render_angles.py > render_angles.log 2>&1
python contact_sheet.py
python audit_outputs.py
```

[COMMANDS.md](COMMANDS.md) records the inspection, backup, iteration, build, and verification command history. Rebuilding resets the scene from the local FBX and overwrites generated outputs; back up any future manual changes before rebuilding. Camera-only refreshes need just the render and contact-sheet commands.

## Limits And Scope

The Blender armature now uses volume-preserving deformation with the original MHR skin weights. This is not a full port of MHR's learned pose correctives, parametric runtime, or procedural twist automation. Imported shape keys remain present, but a complete MHR identity/expression UI and IK controls are not supplied. Large arbitrary rotations can still need corrective work.

Clothing is an attached procedural material region, not a separate simulated garment. Fingers and facial topology remain original, with reversible visual simplification. The tonal shader uses EEVEE Shader to RGB and is not Cycles-compatible without material changes.

No ROM JSON or MP4 was used. The separate ROM visualization script was not edited; only its checksum was read for preservation verification. No dependencies or unrelated assets were installed or downloaded.
