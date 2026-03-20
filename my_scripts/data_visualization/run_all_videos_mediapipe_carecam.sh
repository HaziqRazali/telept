#!/usr/bin/env bash

VIDEOS_DIR="/home/haziq/datasets/telept/data/Special Tests/videos"
MEDIAPIPE_DIR="/home/haziq/datasets/telept/data/Special Tests/mediapipe"
CARECAM_DIR="/home/haziq/datasets/telept/data/Special Tests/carecam"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for video in "$VIDEOS_DIR"/*.MP4; do
    stem="$(basename "$video" .MP4)"
    echo "===== $stem ====="

    # ── MediaPipe ──────────────────────────────────────────────────────────────
    json="$MEDIAPIPE_DIR/$stem.json"
    if [[ -f "$json" ]]; then
        echo "  [mediapipe] Running..."
        python "$SCRIPT_DIR/visualize_mediapipe.py" "$video" "$json"
    else
        echo "  [mediapipe] SKIP — no JSON found: $json"
    fi

    # ── CareCam ────────────────────────────────────────────────────────────────
    # Carecam folders are named like "20260302_<stem>"; use glob to find it
    carecam_folder="$(find "$CARECAM_DIR" -maxdepth 1 -type d -name "*_${stem}" | head -1)"
    if [[ -n "$carecam_folder" && -f "$carecam_folder/out.pkl" ]]; then
        echo "  [carecam]  Running..."
        python "$SCRIPT_DIR/visualize_carecam.py" "$video" "$carecam_folder/out.pkl"
    else
        echo "  [carecam]  SKIP — no out.pkl found for stem: $stem"
    fi

    echo ""
done
