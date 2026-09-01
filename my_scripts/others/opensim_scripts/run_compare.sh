#!/bin/bash
# Run full vs bony pipeline for a single trial and produce a side-by-side comparison video.
#
# Usage:
#   ./run_compare.sh <INPUT_JSON> <OUTPUT_BASE> [--fps N] [--gender X] [--offset N] [--load_camera_settings]
#
# Example:
#   ./run_compare.sh /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json \
#       output/band_pull_apart --offset=1 --load_camera_settings
#
# Outputs:
#   <OUTPUT_BASE>_full/<stem>/osim_results/superimp_res.mp4   (full 105-marker run)
#   <OUTPUT_BASE>_bony/<stem>/osim_results/superimp_res.mp4   (bony 57-marker run)
#   <OUTPUT_BASE>_compare.mp4                                 (side-by-side)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_JSON=$1
OUTPUT_BASE=$2
shift 2

# Forward all remaining flags to run_smpl2bsm.sh (e.g. --offset=1 --load_camera_settings)
# but strip --gui if present (we want video, not interactive)
EXTRA_FLAGS=""
for arg in "$@"; do
  [[ "$arg" == "--gui" ]] && continue
  EXTRA_FLAGS="$EXTRA_FLAGS $arg"
done

STEM=$(basename "$INPUT_JSON" .json)

SMPL2AB_DIR="$HOME/code/SMPL2AddBiomechanics"

OUT_FULL="$SMPL2AB_DIR/${OUTPUT_BASE}_full"
OUT_BONY="$SMPL2AB_DIR/${OUTPUT_BASE}_bony"

echo "=== Running FULL (105 markers) ==="
bash "$SCRIPT_DIR/run_smpl2bsm.sh" "$INPUT_JSON" "${OUTPUT_BASE}_full" \
  --markers=full --vis-only $EXTRA_FLAGS

echo "=== Running BONY (57 markers) ==="
bash "$SCRIPT_DIR/run_smpl2bsm.sh" "$INPUT_JSON" "${OUTPUT_BASE}_bony" \
  --markers=bony --vis-only $EXTRA_FLAGS

# Locate the two MP4 outputs — grab the newest file matching superimp_res*.mp4
VID_FULL=$(find "$OUT_FULL" -name 'superimp_res*.mp4' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
VID_BONY=$(find "$OUT_BONY" -name 'superimp_res*.mp4' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

if [[ -z "$VID_FULL" || -z "$VID_BONY" ]]; then
  echo "ERROR: could not find one or both output videos." >&2
  echo "  full: $VID_FULL" >&2
  echo "  bony: $VID_BONY" >&2
  exit 1
fi

OUT_COMPARE="$SMPL2AB_DIR/${OUTPUT_BASE}_compare.mp4"

echo "=== Stitching side-by-side → $OUT_COMPARE ==="
ffmpeg -y \
  -i "$VID_FULL" \
  -i "$VID_BONY" \
  -filter_complex "\
    [0:v]drawtext=text='full (105 markers)':fontcolor=white:fontsize=28:x=10:y=10[a];\
    [1:v]drawtext=text='bony (57 markers)':fontcolor=white:fontsize=28:x=10:y=10[b];\
    [a][b]hstack=inputs=2[out]" \
  -map "[out]" \
  -c:v libx264 -crf 18 -preset fast \
  "$OUT_COMPARE"

echo "Side-by-side saved to: $OUT_COMPARE"
