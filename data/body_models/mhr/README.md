# MHR body-model assets

This directory contains the reusable MHR body-model assets only.  The ROM
visualization and rendering scripts remain in:

```text
/home/haziq/datasets/telept/my_scripts/data_visualization/rom_visualization_scripts/
```

The current model has the native lean MHR identity (`[-1.5, -0.75, +0.75]` in
the first three shape coefficients) plus the standard `Lean_Abs_Definition`
surface layer.  The blue and translucent outputs use the same geometry and
skeleton; they differ only in rendering appearance.

## Contents

```text
mhr/
├── native/
│   ├── mhr_model.pt                 # Native TorchScript MHR runtime
│   └── lod1.fbx                     # Blender-compatible MHR body/rig asset
├── blender/
│   ├── stylized_mhr.blend           # Original editable stylized scene
│   └── stylized_mhr_lean_abs_outline.blend
├── derived/
│   └── lean_abs_standard/
│       └── sculpt_bind.npz          # Regenerable extracted sculpt delta
├── archive/
│   └── legacy_rom_visualization_sources/
│       └── ...                       # Verified redundant ROM-side copies
└── README.md
```

The renderers intentionally stay in the ROM visualization directory.  Their
defaults now resolve the reusable assets from this directory, so normal ROM
renders do not need explicit model, scene, or cache-path overrides.  The
Blender scene is used as the source of the editable abs shape layer; the
final direct render uses native TorchScript MHR skinning and PyRender.

## Blue opaque/dark-Fresnel appearance

Run the existing standard-abs renderer.  Its model, sculpt scene, and derived
cache defaults point to this `mhr/` directory:

```bash
TELEPT_ROOT=$(git rev-parse --show-toplevel)
ROM_ROOT="$TELEPT_ROOT/my_scripts/data_visualization/rom_visualization_scripts"
PYTHON="${MHR_PYTHON:-python}"
JSON=$ROM_ROOT/s01_cam0_brett_adhesive_capsulitis.json
VIDEO=/home/haziq/datasets/mocap/data/brett/val/s01/videos/cam0/adhesive_capsulitis.mp4

$PYTHON "$ROM_ROOT/stylized_mhr/render_torchscript_lean_abs_rom.py" \
  --json "$JSON" \
  --video "$VIDEO" \
  --time 10.076 \
  --side right \
  --definition standard \
  --annotations skeleton
```

The mesh-only result is written to:

```text
$ROM_ROOT/results/adhesive_capsulitis_right_shoulder_flexion_standard_abs_skeleton_dark_fresnel_mesh_only.png
```

This command also writes the front, side, and source-video composite outputs.

## Translucent reskinned dark-gray appearance

This version does not need the MP4 because it produces only the front and
sagittal mesh panels.  Its model, sculpt scene, and derived cache also come
from this `mhr/` directory by default:

```bash
TELEPT_ROOT=$(git rev-parse --show-toplevel)
ROM_ROOT="$TELEPT_ROOT/my_scripts/data_visualization/rom_visualization_scripts"
PYTHON="${MHR_PYTHON:-python}"
JSON=$ROM_ROOT/s01_cam0_brett_adhesive_capsulitis.json

$PYTHON "$ROM_ROOT/stylized_mhr/render_reskinned_abs_skeleton_mesh_only.py" \
  --json "$JSON" \
  --time 10.076 \
  --side right \
  --background 0.16 0.18 0.21 \
  --output "$ROM_ROOT/results/adhesive_capsulitis_right_shoulder_flexion_reskinned_dark_gray.png"
```

The output is:

```text
$ROM_ROOT/results/adhesive_capsulitis_right_shoulder_flexion_reskinned_dark_gray.png
```

Separate `_front.png` and `_side.png` files are also written beside it.

## Notes

- `mhr_model.pt` contains the native MHR topology, weights, pose correctives,
  and skinning runtime.
- `lod1.fbx` is retained for Blender-based inspection and editing; the two
  direct TorchScript render commands above do not need it at render time.
- `sculpt_bind.npz` is a derived cache.  It can be regenerated from the
  copied `.blend` with Blender if removed.  The direct renderers reuse this
  cache when it is present, so Blender is not required for a render on a host
  that already has the cache.
- JSON/NPZ motion outputs, MP4 files, and rendered results belong in their
  dataset or results directories, not in this body-model directory.
- The legacy duplicate scene/cache files are archived under
  `archive/legacy_rom_visualization_sources/`.  They were byte-for-byte
  verified before the redundant ROM-side originals were removed.  The shared
  package cache under `/home/haziq/.cache/` is intentionally left intact.

## Default-pose examples

The [example_scripts](example_scripts/) directory renders this copied body in
the native MHR zero-parameter default pose without a JSON file or MP4:

```bash
bash /home/haziq/datasets/telept/data/body_models/mhr/example_scripts/render_default_blue.sh
bash /home/haziq/datasets/telept/data/body_models/mhr/example_scripts/render_default_reskinned.sh
```

The examples use the same lean-plus-standard-abs geometry as the ROM renders.
They produce the opaque blue/dark-Fresnel and translucent reskinned previews
under `previews/`, with front and side panels plus the colored skeleton and
white joint dots.
