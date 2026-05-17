
# OpenSim Scripts

## Order of steps

1. *(Optional)* `prepare_smpl2ab.py` — convert source data (fit3d JSON, etc.) → smpl2ab-compatible `.npz`
2. `run_smpl2bsm.sh` — fit BSM skeleton to SMPL motion (markers → IK → video)

---

## Fitting BSM to SMPL motion

conda activate addbiomechanics
cd ~/code/SMPL2AddBiomechanics

<!--
Full pipeline shortcut — runs Steps 1+2+3 in one go.
  Input:  fit3d JSON (auto-converted to .npz via prepare_smpl2ab.py internally)
  Output: output/dumbbell_biceps_curls/<trial>/osim_results/
            IK/<trial>_segment_0_ik.mot   ← joint angles
            Models/match_markers_but_ignore_physics.osim ← scaled model
  --markers=full   uses all 105 virtual markers (vs --markers=bony for 57 anatomical-only)
  --gui            opens interactive aitviewer instead of exporting video
  --vis-only       skip Steps 1+2 if outputs already exist (re-use cached IK)
-->
cd ~/code/SMPL2AddBiomechanics
./run_smpl2bsm.sh \
  /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/dumbbell_biceps_curls.json \
  output/dumbbell_biceps_curls \
  --markers=full --gui

cd ~/code/SMPL2AddBiomechanics && ~/datasets/telept/my_scripts/opensim_scripts/run_smpl2bsm.sh \
  /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/dumbbell_biceps_curls.json \
  output/dumbbell_biceps_curls --markers=full --vis --vis-only

<!--
Step 1: SMPL → markers.trc
  reads SMPL .npz → places virtual markers at anatomical landmarks → writes .trc
  inp:  models/bsm/sample_motion/01/  ← dir of SMPL .npz files (one per trial, e.g. 01_01_poses.npz)
  out:  output/01/
          _subject.json               ← subject metadata (height, mass, gender)
          unscaled_generic.osim       ← copy of bsm.osim (required by engine)
          Geometry/                   ← copy of bsm mesh files (required by engine)
          trials/01_01_poses/
            markers.trc               ← 3D marker trajectories
-->
cd ~/code/SMPL2AddBiomechanics
python smpl2ab/smpl2addbio.py -i models/bsm/sample_motion/01 -o output

<!--
Step 2: IK + model scaling (AddBiomechanics engine)
  IPOPT-based: scales generic OpenSim model to subject proportions, fits joint angles to marker trajectories (inverse kinematics)
  inp:  output/01/                                ← subject folder from Step 1 (contains markers.trc + _subject.json)
  out:  output/01/osim_results/
          Models/
          match_markers_but_ignore_physics.osim   ← scaled subject model
          Geometry/                               ← mesh files
        IK/
          01_01_poses_segment_0_ik.mot            ← joint angles
          01_01_poses_segment_0_marker_errors.csv
  NOTE: non-fatal errors about plotting.py and "Geometry already exists" are safe to ignore
  python ~/code/AddBiomechanics/server/engine/src/engine.py output/01 osim_results
-->
cd ~/code/SMPL2AddBiomechanics
python ~/code/AddBiomechanics/server/engine/src/engine.py output/01 osim_results

<!--
Step 3: visualize — superimpose SMPL mesh + OpenSim IK skeleton, export video
  inp:  output/01/osim_results/Models/match_markers_but_ignore_physics.osim  ← scaled subject model
        output/01/osim_results/IK/01_01_poses_segment_0_ik.mot               ← joint angles from Step 2
        models/bsm/sample_motion/01/01_01_poses.npz                          ← original SMPL motion
  out:  superimp_res.mp4  (cwd) — add --gui for interactive viewer instead
-->
cd ~/code/SMPL2AddBiomechanics
python smpl2ab/show_ab_results.py \
  --osim_path=output/01/osim_results/Models/match_markers_but_ignore_physics.osim \
  --mot_path=output/01/osim_results/IK/01_01_poses_segment_0_ik.mot \
  --smpl_motion_path=models/bsm/sample_motion/01/01_01_poses.npz \
  --gui

---

## Standalone OpenSim viewers

<!--
Rajagopal: Full skeleton + 80 lower-body muscles. Apache 2.0. Rigid torso (no lumbar torque), simplified shoulder (no scapula).
-->
python view_opensim.py --osim_path ~/code/AddBiomechanics/server/data/StandardizedModels/Rajagopal2015_passiveCal_hipAbdMoved.osim

<!--
# BSM: Full skeleton (skull included) + 80 lower-body muscles. Non-commercial research only (MPI-IS licence). Virtual markers shown as blue dots. Upper body torque-actuated (net joint torques only, no individual muscle forces above waist).
-->
python view_opensim.py --osim_path ~/datasets/mocap/data/skel_models_v1.1/bsm.osim

# MoBL-ARMS — upper-body model, ~100 shoulder/elbow/wrist muscles.
# Use for: bicep curls, resistance band exercises, shoulder rehabilitation, any upper-limb task.
# First run auto-converts .vtp → .ply. Non-commercial research only (SimTK licence).
python view_opensim.py --osim_path "~/Downloads/MoBL_ARMS/Bimanual Upper Arm Model/MoBL_ARMS_bimanual_6_2_21.osim"


# Rajagopal 

python visualize_smpl_bsm_markers.py
python find_mhr_rajagopal_markers.py --visualize