# Lean MHR Surface Refinement

## Deliverables

- Scene: `/home/haziq/datasets/telept/data/body_models/mhr/blender/stylized_mhr_lean_abs_outline.blend`
- Previous result versus updated: [lean_abs_update_comparison.png](lean_abs_update_comparison.png)
- Same lean body, definition off/on: [lean_abs_definition_comparison.png](lean_abs_definition_comparison.png)
- Raised-arm biceps off/on: [lean_abs_biceps_flexion_definition_comparison.png](lean_abs_biceps_flexion_definition_comparison.png)
- Five-view overview: [lean_abs_comparison_sheet.png](lean_abs_comparison_sheet.png)
- Front comparison: [lean_abs_front_t_pose_comparison.png](lean_abs_front_t_pose_comparison.png)
- 3/4 comparison: [lean_abs_three_quarter_t_pose_comparison.png](lean_abs_three_quarter_t_pose_comparison.png)
- Side comparison: [lean_abs_side_t_pose_comparison.png](lean_abs_side_t_pose_comparison.png)
- Back comparison: [lean_abs_back_t_pose_comparison.png](lean_abs_back_t_pose_comparison.png)
- Raised-arm comparison: [lean_abs_shoulder_flexion_comparison.png](lean_abs_shoulder_flexion_comparison.png)
- Torso close-ups: [lean_abs_torso_comparison.png](lean_abs_torso_comparison.png)

Reports and comparison renders remain in
`/home/haziq/datasets/telept/my_scripts/data_visualization/rom_visualization_scripts/stylized_mhr/`.
The reusable Blender scene is kept in
`/home/haziq/datasets/telept/data/body_models/mhr/blender/`.
The requested filename contains "outline", but this scene has no outlines or accent geometry.

## Source Inspection

Built only from the canonical original scene at
`/home/haziq/datasets/telept/data/body_models/mhr/blender/stylized_mhr.blend`,
not any trial scene.
Actual inspection is recorded in [lean_abs_inspection.json](lean_abs_inspection.json).

- Body: `body_mesh`; armature: `body_world`, 126 bones.
- 18,439 base vertices, 36,874 triangles; vertex indices, base coordinates, topology, vertex groups and every skin weight retained exactly. No unweighted vertices.
- 238 existing shape keys, including imported `shape_*` keys and artistic proportion/tailoring keys. The original broad-torso and soft-tailoring keys were active at 1; imported keys were inactive.
- Original scene frame 1 was the bind/default A-pose with identity bone-local pose transforms, not a T-pose. The original `body_worldAction` is retained unchanged as an unused action.
- Materials were `Warm peach` (with procedural clothing regions) and `Charcoal navy exercise wear`; hair had its own dark-brown material.
- Body stack: Smooth, Decimate (0.24), rest-position Geometry Nodes, and volume-preserving Armature.
- Imported custom normals and Auto Smooth were enabled. Those stale normals produced patchy shading after reshaping.

The exact lean MHR shape was **not verified to be present**. Imported shape-key names are not a documented identity-parameter mapping, and this file does not provide a verified interface for setting the requested coefficients. Values `[-1.5, -0.75, +0.75]` were recorded as reference metadata, **not assigned to guessed shape keys**. The supplied lean render and comparison sheet were used as visual guides. This is an artistic approximation, not an exact parameter reconstruction.

## Editable Changes

`Lean_Proportions` narrows the original broad torso, waist and hips, reduces abdominal depth and limb fullness, and gently smooths local transitions. `Lean_Abs_Definition` adds a shallow center depression, broad paired abdominal planes, gentle upper/lower separations, mild oblique shaping, and upper-arm volume. Definition was strengthened on 2026-09-30 after the previous result proved visually insufficient. Broad planes replace the nearly invisible first treatment; there are no individually exaggerated six-pack blocks.

Both are additive shape keys on the original body, evaluated before its existing armature. Each is saved at 1; set either to 0 to remove that layer. Set both to 0 for original proportions under the working anatomy shading. Maximum new definition displacement is about 7.79 mm before modifiers. Evaluated regional maxima are 7.45 mm at the abdomen and 6.80 mm at each biceps, in both saved poses. No separate muscle objects, curves, tubes, decals, painted lines, normal-map lines, or outline effects exist.

The existing decimator is retained but disabled to avoid discarding shallow anatomy. No subdivision, multiresolution or remeshing was added: the evaluated body remains 18,439 vertices / 36,874 triangles. The retained Smooth modifier now uses factor 0.25 and 2 iterations instead of 0.7 and 5, so it no longer flattens the intended relief. Auto Smooth is disabled on the working body so normals follow its actual geometry; imported custom-normal data remains on the hidden original copy.

The body uses one rough, nonmetallic blue-gray Principled material. Soft studio lighting supplies the dark blue/gray surface shading; no artificial cavity or stripe mask is used.

## Preserved Objects And Poses

`Original_Body_Comparison` is a hidden, independently copied original body with its original mesh, all keys, materials, custom normals and modifier settings. It shares the preserved armature. Hide the working `body_mesh` before unhiding this copy to avoid coincident surfaces. Select the retained `body_worldAction` to inspect the original source poses.

The existing hair object, `Hair — head skinned proxy`, is hidden in the working scene and in all anatomy renders, not deleted. There are no separate clothing meshes: the source shirt and trousers are procedural material regions. The anatomy material bypasses those regions on the working body; they remain intact on the hidden original and in the original materials.

The new action has a true joint-aligned T-pose at frame 1 and a manually authored 135-degree right shoulder-flexion pose at frame 40. The other arm stays in T-pose for comparison. Upper-arm and forearm directions are controlled through the original armature, not mesh pose baking. Edit/unlink this new action when authoring other poses. No ROM JSON, video or pipeline is involved.

## Comparison And Validation

Each before/after pair uses the same pose, camera, lighting, material, normals and modifier settings. "Before" means original body proportions with the two new shape keys disabled, not the original clothed pastel render. This isolates the shape change. Both front and 3/4 torso close-ups are also supplied.

The new previous-result/update sheet uses the preserved close-up images from the first delivery on the left and the updated surface on the right. Material, light settings and cameras are unchanged; the definition key and smoothing settings differ. The separate definition-off/on sheets keep the lean proportion key and all render/modifier settings identical, toggling only `Lean_Abs_Definition`. They include a dedicated raised-arm biceps close-up. These comparisons are more useful for judging anatomy than the small five-view overview.

[lean_abs_saved_validation.json](lean_abs_saved_validation.json) comes from independently reopening both scenes:

- Exact preservation of base geometry, indices, topology, weights, bone hierarchy/rest matrices and all 238 pre-existing shape keys.
- Exact preservation of the original action and original materials on the hidden copy.
- Zero boundary edges, zero nonmanifold edges; no new accent objects.
- No flipped triangles introduced by the lean edit at rest or by definition in either saved pose.
- Definition remains on the same weighted mesh with bounded displacement in both poses. Abdomen and left/right biceps are checked separately for nontrivial relief below 10 mm.
- True horizontal upper-arm and forearm directions in T-pose; measured right flexion 135.000004 degrees.
- Sixteen isolated +/-30-degree shoulder, elbow, hip and knee tests pass on unchanged evaluated topology.
- 69 pre-existing files, including the original scene, prior trials and ROM scripts, remain unchanged by SHA-256.

[lean_abs_render_audit.json](lean_abs_render_audit.json) checks eight before/after pairs plus three definition-off/on pairs, covering nineteen 1000x1000 renders. Checks cover nonblank content, pair differences, and unclipped full-body framing. Close-ups deliberately crop the rest of the body. Visual review covers torso/limb transitions, side abdomen, back and raised-arm deformation; the source mesh's coarse shoulder/armpit contour remains visible at close range. These checks validate the requested poses, not every possible articulation or an exact anatomical fit to the image. Pixel differences alone do not establish artistic quality. Native learned MHR pose correctives/procedural automation have not been added.

## Reproduce

Run from this directory using the existing Blender and Python installations:

```bash
/home/haziq/blender-3.6.17-linux-x64/blender -b --threads 8 --python-exit-code 1 --python build_lean_abs.py
/home/haziq/blender-3.6.17-linux-x64/blender -b --threads 8 --python-exit-code 1 --python validate_lean_abs.py
/home/haziq/anaconda3/envs/mhr_new/bin/python lean_abs_make_sheets.py
```

The builder overwrites only its new deliverable and `lean_abs_*` outputs; preserve manual edits to that deliverable before rerunning. It never saves over the original or a previous trial. Existing ROM visualization scripts were not modified.
