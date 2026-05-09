# SMPL → OpenSim Pipeline

Converts a SMPL motion sequence into an OpenSim biomechanical model fit, then
renders a superimposed visualization video.

---

## Overview

```
SMPL .npz  →  [Step 1] smpl2addbio.py  →  .trc markers + _subject.json
                                              ↓
                         [Step 2] engine.py  →  .osim model + .mot IK result
                                              ↓
                     [Step 3] show_ab_results.py  →  superimp_res.mp4
```

---

## Prerequisites

One conda environment is required:

| Env name           | Python | Used for       |
|--------------------|--------|----------------|
| `addbiomechanics`  | 3.10   | All 3 steps    |

Key repos (assumed cloned under `~/code/`):
- `~/code/SMPL2AddBiomechanics/` — main pipeline scripts
- `~/code/aitviewer-skel/` — visualization (installed as editable pkg)
- `~/code/AddBiomechanics/` — engine

Required model files under `~/code/SMPL2AddBiomechanics/models/`:
```
models/
  smpl/
    SMPL_MALE.pkl
    SMPL_FEMALE.pkl
    SMPL_NEUTRAL.pkl
  bsm/
    bsm.osim
    Geometry/   (VTP mesh files)
    sample_motion/
```

BSM/skel models at `~/datasets/mocap/data/skel_models_v1.1/`.

aitviewer config (`~/code/aitviewer-skel/aitviewer/aitvconfig.yaml`) must have:
```yaml
window_type: "headless"
smplx_models: "/home/haziq/code/SMPL2AddBiomechanics/models"
skel_models:  "/home/haziq/datasets/mocap/data/skel_models_v1.1"
osim_geometry: "/home/haziq/datasets/mocap/data/skel_models_v1.1/Geometry"
```

---

## Step 1 — Convert SMPL motion to marker trajectories

**Environment:** `addbiomechanics`  
**Script:** `~/code/SMPL2AddBiomechanics/smpl2ab/smpl2addbio.py`

```bash
conda activate addbiomechanics
cd ~/code/SMPL2AddBiomechanics

python smpl2ab/smpl2addbio.py \
  -i models/bsm/sample_motion/01 \
  -o output/01
```

**What it does:** Reads SMPL pose parameters (`.npz`), places virtual markers on
the body surface at anatomical landmark positions, and writes them out as a `.trc`
marker trajectory file that OpenSim can read.

**Outputs:**
```
output/01/
  _subject.json               ← subject metadata (height, mass, gender)
  unscaled_generic.osim       ← copy of bsm.osim (required by engine)
  Geometry/                   ← copy of bsm mesh files (required by engine)
  trials/
    01_01_poses/
      markers.trc             ← 3D marker trajectories
```

---

## Step 2 — Run the AddBiomechanics engine (IK + model scaling)

**Environment:** `addbiomechanics`  
**Script:** `~/code/AddBiomechanics/server/engine/src/engine.py`

```bash
conda activate addbiomechanics  # same env
cd ~/code/SMPL2AddBiomechanics

python ~/code/AddBiomechanics/server/engine/src/engine.py \
  output/01 \
  osim_results
```

**Arguments:**
- `output/01` — path to the subject folder produced by Step 1
- `osim_results` — name for the output subfolder (default: `osim_results`)

**What it does:** Runs IPOPT-based optimization to scale a generic OpenSim model
to the subject's proportions and fits it to the marker trajectories (inverse
kinematics).

**Outputs:**
```
output/01/osim_results/
  Models/
    match_markers_but_ignore_physics.osim   ← scaled subject model
    unscaled_generic.osim
    Geometry/                               ← mesh files for visualization
  IK/
    01_01_poses_segment_0_ik.mot            ← joint angle time series
    01_01_poses_segment_0_ik_setup.xml
    01_01_poses_segment_0_marker_errors.csv
```

> **Note:** Some non-fatal errors about `plotting.py` (pandas/numpy dtype conflict)
> and `Geometry already exists` can be ignored — all IK outputs are still written.

---

## Step 3 — Visualize and export video

**Environment:** `addbiomechanics`  
**Script:** `~/code/SMPL2AddBiomechanics/smpl2ab/show_ab_results.py`

```bash
conda activate addbiomechanics  # same env
cd ~/code/SMPL2AddBiomechanics

# Export video (headless, no display needed):
python smpl2ab/show_ab_results.py \
  --osim_path=output/01/osim_results/Models/match_markers_but_ignore_physics.osim \
  --mot_path=output/01/osim_results/IK/01_01_poses_segment_0_ik.mot \
  --smpl_motion_path=models/bsm/sample_motion/01/01_01_poses.npz

# Or launch the interactive GUI viewer:
python smpl2ab/show_ab_results.py \
  --osim_path=output/01/osim_results/Models/match_markers_but_ignore_physics.osim \
  --mot_path=output/01/osim_results/IK/01_01_poses_segment_0_ik.mot \
  --smpl_motion_path=models/bsm/sample_motion/01/01_01_poses.npz \
  --gui
```

**What it does:** Loads the SMPL mesh sequence and the OpenSim IK skeleton
sequence, superimposes them, renders all frames headlessly, and encodes them
into an MP4 via ffmpeg.

**Output:**
```
superimp_res.mp4    (in the current working directory)
```

> If the file already exists, it saves as `superimp_res_0.mp4`, `superimp_res_1.mp4`, etc.

---

## Quick one-liner (all three steps)

```bash
conda activate addbiomechanics
cd ~/code/SMPL2AddBiomechanics

python smpl2ab/smpl2addbio.py \
  -i models/bsm/sample_motion/01 -o output/01

python ~/code/AddBiomechanics/server/engine/src/engine.py output/01 osim_results

python smpl2ab/show_ab_results.py \
  --osim_path=output/01/osim_results/Models/match_markers_but_ignore_physics.osim \
  --mot_path=output/01/osim_results/IK/01_01_poses_segment_0_ik.mot \
  --smpl_motion_path=models/bsm/sample_motion/01/01_01_poses.npz
```

  Converts a SMPL motion sequence into an OpenSim biomechanical model fit,
  then renders a superimposed visualization video.

  Overview
  --------
  SMPL .npz  →  [Step 1] smpl2addbio.py  →  .trc markers + _subject.json
                                                 ↓
                            [Step 2] engine.py  →  .osim model + .mot IK
                                                 ↓
                        [Step 3] show_ab_results.py  →  superimp_res.mp4

  Environment
  -----------
  Conda env : addbiomechanics  (Python 3.10, used for all steps)

  Key repos (cloned under ~/code/):
    ~/code/SMPL2AddBiomechanics/   main pipeline scripts
    ~/code/aitviewer-skel/         visualization (installed as editable pkg)
    ~/code/AddBiomechanics/        engine

  Required model files under ~/code/SMPL2AddBiomechanics/models/:
    models/
      smpl/
        SMPL_MALE.pkl
        SMPL_FEMALE.pkl
        SMPL_NEUTRAL.pkl
      bsm/
        bsm.osim
        Geometry/       (VTP mesh files)
        sample_motion/

  BSM/skel models:  ~/datasets/mocap/data/skel_models_v1.1/

  aitviewer config  ~/code/aitviewer-skel/aitviewer/aitvconfig.yaml  must have:
    window_type: "headless"
    smplx_models: "/home/haziq/code/SMPL2AddBiomechanics/models"
    skel_models:  "/home/haziq/datasets/mocap/data/skel_models_v1.1"
    osim_geometry: "/home/haziq/datasets/mocap/data/skel_models_v1.1/Geometry"


================================================================================
  Step 1 — Convert SMPL motion to marker trajectories
================================================================================

  Script : ~/code/SMPL2AddBiomechanics/smpl2ab/smpl2addbio.py
  What   : Reads SMPL pose .npz, places virtual markers at anatomical landmarks,
           writes a .trc marker trajectory file that OpenSim can read.

    conda activate addbiomechanics
    cd ~/code/SMPL2AddBiomechanics
    python smpl2ab/smpl2addbio.py -i models/bsm/sample_motion/01 -o output/01

  Outputs
    output/01/
      _subject.json              subject metadata (height, mass, gender)
      unscaled_generic.osim      copy of bsm.osim (required by engine)
      Geometry/                  copy of bsm mesh files (required by engine)
      trials/01_01_poses/
        markers.trc              3D marker trajectories


================================================================================
  Step 2 — Run the AddBiomechanics engine (IK + model scaling)
================================================================================

  Script : ~/code/AddBiomechanics/server/engine/src/engine.py
  What   : IPOPT-based optimisation — scales the generic OpenSim model to the
           subject's proportions, then fits joint angles to marker trajectories
           (inverse kinematics).

    conda activate addbiomechanics
    cd ~/code/SMPL2AddBiomechanics
    python ~/code/AddBiomechanics/server/engine/src/engine.py output/01 osim_results

  Arguments
    output/01       subject folder from Step 1
    osim_results    name for the output subfolder

  Outputs
    output/01/osim_results/
      Models/
        match_markers_but_ignore_physics.osim   scaled subject model
        Geometry/                               mesh files for visualization
      IK/
        01_01_poses_segment_0_ik.mot            joint angle time series
        01_01_poses_segment_0_ik_setup.xml
        01_01_poses_segment_0_marker_errors.csv

  NOTE: non-fatal errors about plotting.py (pandas/numpy dtype) and
        "Geometry already exists" can be ignored — IK outputs are still written.


================================================================================
  Step 3 — Visualize and export video
================================================================================

  Script : ~/code/SMPL2AddBiomechanics/smpl2ab/show_ab_results.py
  What   : Loads the SMPL mesh sequence + OpenSim IK skeleton, superimposes
           them, renders all frames headlessly, encodes to MP4 via ffmpeg.

    conda activate addbiomechanics
    cd ~/code/SMPL2AddBiomechanics

    # Export video (headless)
    python smpl2ab/show_ab_results.py \
      --osim_path=output/01/osim_results/Models/match_markers_but_ignore_physics.osim \
      --mot_path=output/01/osim_results/IK/01_01_poses_segment_0_ik.mot \
      --smpl_motion_path=models/bsm/sample_motion/01/01_01_poses.npz

    # Interactive GUI
    python smpl2ab/show_ab_results.py \
      --osim_path=output/01/osim_results/Models/match_markers_but_ignore_physics.osim \
      --mot_path=output/01/osim_results/IK/01_01_poses_segment_0_ik.mot \
      --smpl_motion_path=models/bsm/sample_motion/01/01_01_poses.npz \
      --gui

  Output:  superimp_res.mp4  (cwd)
           If file exists, saves as superimp_res_0.mp4, superimp_res_1.mp4, ...


================================================================================
  Quick one-liner (all three steps)
================================================================================

    conda activate addbiomechanics
    cd ~/code/SMPL2AddBiomechanics

    python smpl2ab/smpl2addbio.py -i models/bsm/sample_motion/01 -o output/01

    python ~/code/AddBiomechanics/server/engine/src/engine.py output/01 osim_results

    python smpl2ab/show_ab_results.py \
      --osim_path=output/01/osim_results/Models/match_markers_but_ignore_physics.osim \
      --mot_path=output/01/osim_results/IK/01_01_poses_segment_0_ik.mot \
      --smpl_motion_path=models/bsm/sample_motion/01/01_01_poses.npz


================================================================================
  Standalone OpenSim viewers
  (all in ~/datasets/telept/my_scripts/opensim_scripts/)
  (all require: conda activate addbiomechanics)
================================================================================

  Model paths quick-reference
  ---------------------------
  BSM        ~/datasets/mocap/data/skel_models_v1.1/bsm.osim
               80 lower-body muscles   geometry: .ply files present
  Rajagopal  ~/code/AddBiomechanics/server/data/StandardizedModels/
               Rajagopal2015_passiveCal_hipAbdMoved.osim
               80 lower-body muscles   geometry: .vtp converted to .ply (done)
  MoBL-ARMS  ~/Downloads/MoBL_ARMS/Bimanual Upper Arm Model/
               MoBL_ARMS_bimanual_6_2_21.osim
               ~100 upper-body muscles  geometry: .vtp auto-converted on first run


--------------------------------------------------------------------------------
  view_opensim.py — Interactive 3D skeleton + muscles  [MAIN VIEWER]
--------------------------------------------------------------------------------

  Auto-detects Geometry/ next to .osim, converts .vtp→.ply on first run,
  then opens aitviewer with full bone meshes + colour-coded muscle path tubes.

  cd ~/datasets/telept/my_scripts/opensim_scripts

  BSM (default pose):
    python view_opensim.py --osim_path ~/datasets/mocap/data/skel_models_v1.1/bsm.osim

    Expected:
      Full skeleton — skull, ribcage, spine, pelvis, arms, legs as solid meshes.
      Legs slightly apart (anatomical default stance).
      80 lower-limb muscle tubes visible around hip and leg region.
      Blue via-point dots spread across the whole body (marker set).
      Upper body has NO muscle tubes — BSM uses torque actuators above the waist.
      Legend shows 7 groups: Hip adductors, Hamstrings/gracilis, Ankle/foot,
        Gastrocnemius/soleus, Gluteal/deep hip, Iliopsoas, Quadriceps.

  Rajagopal (default pose):
    python view_opensim.py \
      --osim_path ~/code/AddBiomechanics/server/data/StandardizedModels/Rajagopal2015_passiveCal_hipAbdMoved.osim

    Expected:
      Full skeleton in narrow neutral stance (legs close together).
      No skull mesh — Rajagopal's Geometry/ omits the head bone.
      80 lower-limb muscle tubes concentrated around hip/upper leg.
      No blue via-point dots (Rajagopal has no marker set).
      Same 7 muscle groups as BSM.

  With animated IK motion:
    python view_opensim.py \
      --osim_path /path/to/model.osim \
      --mot_path  /path/to/ik.mot \
      --downsample 2            # optional: use every 2nd frame

    Expected:
      Same skeleton + muscles, but animated.
      Frame scrubber visible in Playback bar at the bottom.
      Space to play/pause.

  GUI layout
    Top-left panel   "Muscle groups" colour legend
    Left sidebar     Scene tree — toggle Skeleton or any individual muscle
    Bottom bar       Playback scrubber (active only when --mot_path given)

  Controls
    Left-drag    rotate       Right-drag   pan
    Scroll       zoom         Space        play / pause
    , / .        prev/next frame           Q / Esc   quit

  Muscle colour groups
    Hip adductors           orange
    Hamstrings / gracilis   dark green
    Ankle / foot            purple
    Gastrocnemius / soleus  dark red
    Gluteal / deep hip      red
    Iliopsoas               bright orange
    Quadriceps              steel blue
    Biceps / brachialis     blue          (upper-body models only)
    Triceps                 red           (upper-body models only)
    Shoulder                purple        (upper-body models only)


--------------------------------------------------------------------------------
  view_muscles_3d.py — Interactive 3D muscles (stick skeleton)  [OLDER]
--------------------------------------------------------------------------------

  Shows muscle tubes + skeleton as a stick figure (joint centres connected by
  lines, no bone meshes). Useful when no Geometry/ folder is available.

    python view_muscles_3d.py \
      --osim_path ~/datasets/mocap/data/skel_models_v1.1/bsm.osim

    python view_muscles_3d.py \
      --osim_path ~/datasets/mocap/data/skel_models_v1.1/bsm.osim \
      --mot_path  /path/to/ik.mot

  Expected:
    Same colour-coded muscle tubes and legend as view_opensim.py, but skeleton
    rendered as thin lines between joint centres instead of solid bone meshes.


--------------------------------------------------------------------------------
  render_muscles_static.py — Static 2D anatomical diagram (PNG)
--------------------------------------------------------------------------------

  Renders front + side views as a clean 2D image. No window opened.
  Suitable for slides or reports.

    python render_muscles_static.py \
      --osim_path ~/datasets/mocap/data/skel_models_v1.1/bsm.osim \
      --output    bsm_muscles_default.png

    python render_muscles_static.py \
      --osim_path ~/code/AddBiomechanics/server/data/StandardizedModels/Rajagopal2015_passiveCal_hipAbdMoved.osim \
      --output    rajagopal_muscles.png

  Expected:
    Side-by-side PNG — front (coronal) view on the left, side (sagittal) view
    on the right. Skeleton outline + all muscle paths in group colours.

<!--
Rajagopal: full skeleton + 80 lower-body muscles. Apache 2.0.
Use for: gait, squats, lower-limb joint torques/muscle forces.
NOT suitable for: upper-body *muscle forces* (no Hill-type muscles above waist).
IS suitable for: upper-body *joint torques* (has CoordinateActuators on shoulder/elbow/wrist).
-->

python view_opensim.py --osim_path ~/code/AddBiomechanics/server/data/StandardizedModels/Rajagopal2015_passiveCal_hipAbdMoved.osim

<!--
# BSM — full skeleton (skull included) + 80 lower-body muscles, anatomical default pose.
# Virtual markers shown as blue dots. Upper body torque-actuated (net joint torques only, no individual muscle forces above waist).
# Non-commercial research only (MPI-IS licence).
# Use for: gait, SMPL2AddBiomechanics pipeline. NOT suitable for: upper-body muscle force analysis.
-->
python view_opensim.py --osim_path ~/datasets/mocap/data/skel_models_v1.1/bsm.osim

# MoBL-ARMS — upper-body model, ~100 shoulder/elbow/wrist muscles.
# Use for: bicep curls, resistance band exercises, shoulder rehabilitation, any upper-limb task.
# First run auto-converts .vtp → .ply. Non-commercial research only (SimTK licence).
python view_opensim.py --osim_path "~/Downloads/MoBL_ARMS/Bimanual Upper Arm Model/MoBL_ARMS_bimanual_6_2_21.osim"


# Rajagopal 

python visualize_smpl_bsm_markers.py