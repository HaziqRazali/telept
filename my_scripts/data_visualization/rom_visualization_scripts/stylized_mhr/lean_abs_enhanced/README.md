# Two Lean MHR Creations

## Creation 1: Approved Definition

Approved source scene:
`/home/haziq/datasets/telept/data/body_models/mhr/blender/stylized_mhr_lean_abs_outline.blend`

```text
/home/haziq/datasets/telept/data/body_models/mhr/blender/stylized_mhr_lean_abs_outline.blend
```

Its existing images remain in the parent `stylized_mhr` directory. This is the stronger surface revision the user approved on 2026-09-30, not the earlier nearly invisible treatment. The scene, scripts, and all existing top-level image files are unchanged.

## Creation 2: Stronger Abs

Scene: [stylized_mhr_lean_abs_enhanced.blend](stylized_mhr_lean_abs_enhanced.blend)

```text
/home/haziq/datasets/telept/my_scripts/data_visualization/rom_visualization_scripts/stylized_mhr/lean_abs_enhanced/stylized_mhr_lean_abs_enhanced.blend
```

All new images, scripts, and validation reports are in this `lean_abs_enhanced` folder.

- Abs close-up comparison: [abs_comparison.png](abs_comparison.png)
- Five-view comparison: [five_view_comparison.png](five_view_comparison.png)
- Shoulder-flexion abdomen comparison: [torso_flexion_comparison.png](torso_flexion_comparison.png)
- Standalone front: [front_t_pose_enhanced.png](front_t_pose_enhanced.png)
- Standalone abdominal close-up: [torso_front_enhanced.png](torso_front_enhanced.png)

All comparisons put creation 1 on the left (or top) and creation 2 on the right (or bottom), with identical cameras, lighting, material, modifiers, and poses.

## What Changed

One additional body shape key, `Abs_Extra_Definition`, adds 65% more of the approved abdominal surface relief. The central separation, upper/lower transitions, paired planes and obliques are more pronounced. The stronger highlights and shadows come from real surface shaping under the same lighting, not painted strips, detached geometry, normal-map lines or outline effects.

The new layer is abdomen-only. Lean proportions, biceps shaping, head, limbs, mesh topology, vertex indices, skin weights, armature, original keys and pose action remain unchanged. The body remains `body_mesh`, driven by the original 126-bone `body_world`. Base/evaluated topology remains 18,439 vertices / 36,874 triangles; no remeshing, subdivision or multiresolution is added.

Set `Abs_Extra_Definition` to 0 to recover the approved creation, or 1 for the saved stronger version. Intermediate values blend the extra strength. Keep the two existing lean/definition keys at 1 for this comparison. The saved scene uses frame 1 for T-pose and frame 40 for the existing manual right shoulder-flexion test.

The hidden original body and hidden hair are retained exactly as in creation 1. The same anatomy material bypasses the procedural clothing regions. No new material masks or accent objects are used. The original source mesh's low-resolution shoulder/armpit contour remains visible at close range.

This variant builds from the user-approved saved scene, not an earlier trial. The original MHR shape-parameter mapping remains unverified; neither creation claims an exact reconstruction of reference coefficients [-1.5, -0.75, +0.75].

## Validation

[validation.json](validation.json) records checks after reopening the new saved scene:

- Exact original base geometry, weights, vertex groups, bone hierarchy/rest matrices, and all 240 existing shape keys retained.
- Same scene objects, material inputs, lighting, modifier settings, transforms, and pose action.
- New key at zero recovers the approved evaluated body in both stored poses.
- Biceps vertices remain unchanged; new displacement is confined to the abdominal layer.
- Added relief is bounded below 6 mm, with no new flipped triangles in either pose. This is additional relief, on top of the approved anatomy, not total relief.
- 126 pre-existing top-level files, including both original/approved scenes, previous trials, images and ROM scripts, preserved by SHA-256 checks.

[render_audit.json](render_audit.json) checks sixteen 1000x1000 images across eight matched view pairs. The five full-body views are unclipped; close-ups intentionally crop the rest of the character. Close-up image changes are checked separately, with visual review used to judge the result rather than treating pixel differences as artistic approval.

To reproduce, run in this folder:

```bash
/home/haziq/blender-3.6.17-linux-x64/blender -b --threads 8 --python-exit-code 1 --python build_enhanced.py
/home/haziq/anaconda3/envs/mhr_new/bin/python make_sheets.py
```

These commands overwrite only this variant's outputs; they never save over creation 1. Preserve manual edits to creation 2 before rebuilding.
