# Usage:
#   ./run_smpl2bsm.sh <INPUT> <OUTPUT_DIR> [options]
#
#   INPUT            dir with SMPL .npz file(s)  e.g. models/bsm/sample_motion/01
#                    OR a fit3d .json file        e.g. .../fit3d/train/s03/smplx/band_pull_apart.json
#   OUTPUT_DIR       e.g. output/band_pull_apart
#
#   --markers X      full = all 105 markers; bony = 57 anatomical-only markers  [REQUIRED]
#   --vis-only       skip Steps 1 & 2 if their outputs already exist (use cached IK)
#   --force          force re-run Steps 1 & 2 even if outputs exist (overrides --vis-only)
#   --gui / --vis    open interactive aitviewer instead of exporting video
#   --z_up           treat input data as Z-up (needed for raw BSM sample motions; not for fit3d JSON)
#   --fps N          frame rate for JSON inputs (default: 50)
#   --gender X       neutral|male|female (default: neutral)
#   --body_model X   smpl or smplx (default: smpl)
#   --offset N       shift SMPL mesh N metres along X to separate it from skeleton in viewer
#   --load_camera_settings  restore saved camera position from previous session

# Full pipeline — all 105 markers vs 57 bony markers
#   ./run_smpl2bsm.sh /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/barbell_row.json output/barbell_row_full --markers=full --vis-only --gui --offset=1
#   ./run_smpl2bsm.sh /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/barbell_row.json output/barbell_row_bony --markers=bony --vis-only --gui --offset=1
#   ./run_smpl2bsm.sh /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json output/band_pull_apart_full --markers=full --vis-only --gui --offset=1 --load_camera_settings
#   ./run_smpl2bsm.sh /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json output/band_pull_apart_bony --markers=bony --vis-only --gui --offset=1 --load_camera_settings
# ./run_compare.sh /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/barbell_row.json output/barbell_row --offset=1 --load_camera_settings
# Compare the two runs
#   python ~/datasets/telept/my_scripts/opensim_scripts/compare_bony_vs_full.py \
#       ~/code/SMPL2AddBiomechanics/output/barbell_row_full/barbell_row \
#       ~/code/SMPL2AddBiomechanics/output/barbell_row_bony/barbell_row
#
#   python ~/datasets/telept/my_scripts/opensim_scripts/visualization_scripts/visualize_bony_vs_full.py \
#   /home/haziq/code/SMPL2AddBiomechanics/output/barbell_row_full/barbell_row \
#   /home/haziq/code/SMPL2AddBiomechanics/output/barbell_row_bony/barbell_row \
#   --npz /tmp/smpl2ab_barbell_row/barbell_row/barbell_row.npz \
#   --offset 0

# Full pipeline (NPZ dir)
#./run_smpl2bsm.sh models/bsm/sample_motion/01 output/01 --vis-only --gui

# Visualization only (skip Steps 1 & 2 if outputs exist)
#./run_smpl2bsm.sh .../band_pull_apart.json output/band_pull_apart_full --vis-only

# Visualization only + interactive GUI
#./run_smpl2bsm.sh .../band_pull_apart.json output/band_pull_apart_full --vis-only --gui

#!/bin/bash
set -e

# ensure the addbiomechanics conda env is active
if [[ "$CONDA_DEFAULT_ENV" != "addbiomechanics" ]]; then
  eval "$(conda shell.bash hook)"
  conda activate addbiomechanics
fi

INPUT=$1
OUTPUT_DIR=$2
VIZ_ONLY=0
FORCE=0
GUI_FLAG=""
FPS=50
GENDER=neutral
MARKERS=""
OFFSET_FLAG=""
LOAD_CAM_FLAG=""
ZUP_FLAG=""
BODY_MODEL="smpl"

# parse optional flags
for arg in "$@"; do
  case $arg in
    --vis-only)        VIZ_ONLY=1 ;;
    --force)           FORCE=1 ;;
    --gui|--vis)       GUI_FLAG="--gui" ;;
    --fps=*)           FPS="${arg#*=}" ;;
    --gender=*)        GENDER="${arg#*=}" ;;
    --markers=*)       MARKERS="${arg#*=}" ;;
    --offset=*)        OFFSET_FLAG="--offset=${arg#*=}" ;;
    --load_camera_settings) LOAD_CAM_FLAG="--load_camera_settings" ;;
    --z_up)            ZUPFLAG="--z_up" ;;
    --body_model=*)    BODY_MODEL="${arg#*=}" ;;
    --id_path=*)       ID_PATH_FLAG="--id_path=${arg#*=}" ;;
  esac
done

# --force overrides --vis-only
if [[ $FORCE -eq 1 ]]; then
  VIZ_ONLY=0
fi

if [[ "$MARKERS" != "full" && "$MARKERS" != "bony" ]]; then
  echo "ERROR: --markers=full or --markers=bony is required." >&2
  exit 1
fi

# select osim model + marker dict based on --markers
if [[ "$MARKERS" == "bony" ]]; then
  OSIM_PATH=~/code/SMPL2AddBiomechanics/models/bsm/bsm_bony.osim
  MARKER_DICT=~/code/SMPL2AddBiomechanics/smpl2ab/data/bsm_markers_bony.yaml
else
  OSIM_PATH=~/code/SMPL2AddBiomechanics/models/bsm/bsm.osim
  MARKER_DICT=~/code/SMPL2AddBiomechanics/smpl2ab/data/bsm_markers.yaml
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd ~/code/SMPL2AddBiomechanics

# --- if input is a JSON, convert to a temp NPZ dir on the fly ---
if [[ "$INPUT" == *.json ]]; then
  STEM=$(basename "$INPUT" .json)
  TMPDIR="/tmp/smpl2ab_${STEM}"
  python "$SCRIPT_DIR/prepare_smpl2ab.py" \
    --input "$INPUT" --output "$TMPDIR" --fps "$FPS" --gender "$GENDER"
  INPUT_DIR="$TMPDIR/$STEM"
else
  INPUT_DIR="$INPUT"
fi

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
# smpl2addbio may create a subject subdir inside OUTPUT_DIR (e.g. output/band_pull_apart/band_pull_apart/)
# detect it via _subject.json so downstream steps use the correct path
SUBJECT_JSON=$(find "$OUTPUT_DIR" -maxdepth 2 -name '_subject.json' 2>/dev/null | head -1)
if [[ $VIZ_ONLY -eq 0 ]] || [[ -z "$SUBJECT_JSON" ]]; then
  python smpl2ab/smpl2addbio.py -i "$INPUT_DIR" -o "$OUTPUT_DIR" \
    --osim "$OSIM_PATH" --marker_dict "$MARKER_DICT" --no_confirm
  SUBJECT_JSON=$(find "$OUTPUT_DIR" -maxdepth 2 -name '_subject.json' | head -1)
fi
SUBJECT_DIR=$(dirname "$SUBJECT_JSON")

# smpl2addbio writes trials/<name>.trc (flat file)
# engine.py expects  trials/<name>/markers.trc (subdir + fixed filename)
for trc in "$SUBJECT_DIR/trials/"*.trc; do
  [[ -f "$trc" ]] || continue
  name=$(basename "$trc" .trc)
  mkdir -p "$SUBJECT_DIR/trials/$name"
  mv "$trc" "$SUBJECT_DIR/trials/$name/markers.trc"
done

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
MOT_FILE="$SUBJECT_DIR/osim_results/IK/${TRIAL}_segment_0_ik.mot"
if [[ $VIZ_ONLY -eq 0 ]] || [[ ! -f "$MOT_FILE" ]]; then
  python ~/code/AddBiomechanics/server/engine/src/engine.py "$SUBJECT_DIR" osim_results
fi

# Repair geometry: engine truncates some PLY files during VTP→PLY conversion.
# Overwrite any output PLY that is smaller than the source with the correct file.
export BSM_GEO=~/code/SMPL2AddBiomechanics/models/bsm/Geometry
export OUT_GEO="$SUBJECT_DIR/osim_results/Models/Geometry"
python3 - <<'PYEOF'
import os, shutil
src = os.path.expanduser(os.environ.get('BSM_GEO', ''))
dst = os.environ.get('OUT_GEO', '')
if os.path.isdir(src) and os.path.isdir(dst):
    for f in os.listdir(src):
        if not f.endswith('.ply'): continue
        sp, dp = os.path.join(src, f), os.path.join(dst, f)
        if not os.path.exists(dp) or os.path.getsize(dp) < os.path.getsize(sp):
            shutil.copy2(sp, dp)
PYEOF

# Step 3: visualize — superimpose SMPL mesh + OpenSim IK skeleton, export video
#   inp:  output/01/osim_results/Models/match_markers_but_ignore_physics.osim  ← scaled subject model
#         output/01/osim_results/IK/01_01_poses_segment_0_ik.mot               ← joint angles from Step 2
#         models/bsm/sample_motion/01/01_01_poses.npz                          ← original SMPL motion
#   out:  superimp_res.mp4  (cwd) — add --gui for interactive viewer instead
VIDEO_OUT="$SUBJECT_DIR/osim_results/superimp_res.mp4"
python smpl2ab/show_ab_results.py \
  --osim_path="$SUBJECT_DIR/osim_results/Models/match_markers_but_ignore_physics.osim" \
  --mot_path="$SUBJECT_DIR/osim_results/IK/${TRIAL}_segment_0_ik.mot" \
  --smpl_motion_path="$NPZ" \
  --smpl_markers_path="$MARKER_DICT" \
  --body_model="$BODY_MODEL" \
  --output="$VIDEO_OUT" \
  $GUI_FLAG $OFFSET_FLAG $LOAD_CAM_FLAG $ZUPFLAG $ID_PATH_FLAG
echo "Video saved to: $VIDEO_OUT"