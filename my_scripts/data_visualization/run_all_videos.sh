#!/usr/bin/env bash

VIDEOS_DIR="/home/haziq/datasets/telept/data/Special Tests/videos"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for video in "$VIDEOS_DIR"/*.MP4; do
    echo "Processing: $video"
    python "$SCRIPT_DIR/visualize_sam3d.py" "$video"
done
