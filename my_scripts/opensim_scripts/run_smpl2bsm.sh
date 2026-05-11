# ./run_smpl2bsm.sh ~/code/SMPL2AddBiomechanics/models/bsm/sample_motion/01 ~/code/SMPL2AddBiomechanics/output/01
#!/bin/bash
set -e

INPUT_DIR=$1   # e.g. models/bsm/sample_motion/01
OUTPUT_DIR=$2  # e.g. output/01

cd ~/code/SMPL2AddBiomechanics

# derive trial name from the first .npz in the input dir (e.g. 01_01_poses)
NPZ=$(ls "$INPUT_DIR"/*.npz | head -1)
TRIAL=$(basename "$NPZ" .npz)

# Step 1: SMPL → markers.trc
#   reads SMPL .npz → places virtual markers at anatomical landmarks → writes .trc
#   inp:  models/bsm/sample_motion/01/  ← dir of SMPL .npz files (one per trial, e.g. 01_01_poses.npz)
#   out:  output/01/
#           _subject.json               ← subject metadata (height, mass, gender)
#           unscaled_generic.osim       ← copy of bsm.osim (required by engine)
#           Geometry/                   ← copy of bsm mesh files (required by engine)
#           trials/01_01_poses/
#             markers.trc               ← 3D marker trajectories
python smpl2ab/smpl2addbio.py -i "$INPUT_DIR" -o "$OUTPUT_DIR"

# Step 2: IK + model scaling (AddBiomechanics engine)
#   IPOPT-based: scales generic OpenSim model to subject proportions, fits joint angles to marker trajectories (inverse kinematics)
#   inp:  output/01/                                ← subject folder from Step 1 (contains markers.trc + _subject.json)
#   out:  output/01/osim_results/
#           Models/
#           match_markers_but_ignore_physics.osim   ← scaled subject model
#           Geometry/                               ← mesh files
#         IK/
#           01_01_poses_segment_0_ik.mot            ← joint angles
#           01_01_poses_segment_0_marker_errors.csv
#   NOTE: non-fatal errors about plotting.py and "Geometry already exists" are safe to ignore
python ~/code/AddBiomechanics/server/engine/src/engine.py "$OUTPUT_DIR" osim_results

# Step 3: visualize — superimpose SMPL mesh + OpenSim IK skeleton, export video
#   inp:  output/01/osim_results/Models/match_markers_but_ignore_physics.osim  ← scaled subject model
#         output/01/osim_results/IK/01_01_poses_segment_0_ik.mot               ← joint angles from Step 2
#         models/bsm/sample_motion/01/01_01_poses.npz                          ← original SMPL motion
#   out:  superimp_res.mp4  (cwd) — add --gui for interactive viewer instead
python smpl2ab/show_ab_results.py \
  --osim_path="$OUTPUT_DIR/osim_results/Models/match_markers_but_ignore_physics.osim" \
  --mot_path="$OUTPUT_DIR/osim_results/IK/${TRIAL}_segment_0_ik.mot" \
  --smpl_motion_path="$NPZ"