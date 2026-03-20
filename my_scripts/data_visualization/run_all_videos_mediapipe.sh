#!/usr/bin/env bash

VIDEOS_DIR="/home/haziq/datasets/telept/data/Special Tests/videos"
MEDIAPIPE_DIR="/home/haziq/datasets/telept/data/Special Tests/mediapipe"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for video in "$VIDEOS_DIR"/*.MP4; do
    stem="$(basename "$video" .MP4)"
    json="$MEDIAPIPE_DIR/$stem.json"

    if [[ -f "$json" ]]; then
        echo "Processing: $stem"
        python "$SCRIPT_DIR/visualize_mediapipe.py" "$video" "$json"
    else
        echo "SKIP: no JSON found for $stem"
    fi
done
