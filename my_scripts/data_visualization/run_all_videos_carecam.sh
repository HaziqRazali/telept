#!/usr/bin/env bash

VIDEOS_DIR="/home/haziq/datasets/telept/data/Special Tests/videos"
CARECAM_DIR="/home/haziq/datasets/telept/data/Special Tests/carecam"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for video in "$VIDEOS_DIR"/*.MP4; do
    stem="$(basename "$video" .MP4)"
    carecam_folder="$(find "$CARECAM_DIR" -maxdepth 1 -type d -name "*_${stem}" | head -1)"

    if [[ -n "$carecam_folder" && -f "$carecam_folder/out.pkl" ]]; then
        echo "Processing: $stem"
        python "$SCRIPT_DIR/visualize_carecam.py" "$video" "$carecam_folder/out.pkl"
    else
        echo "SKIP: no out.pkl found for $stem"
    fi
done
