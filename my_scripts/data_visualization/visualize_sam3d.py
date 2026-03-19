#!/usr/bin/env python3
"""
Run SAM-3D-Body on a video file, saving NPZ results and a rendered video,
then automatically convert the MHR NPZ to SMPL-X JSON.

Usage:
    python visualize_sam3d.py <video_path> [--output_folder <dir>]

Example:
    python visualize_sam3d.py "/home/haziq/datasets/telept/data/Special Tests/videos/sd m neg 01.MP4" \
    --biomarker "left knee flexion"
"""

import argparse
import subprocess
import sys
from pathlib import Path

SAM3D_DIR = Path("/home/haziq/sam-3d-body")
CHECKPOINT_PATH = SAM3D_DIR / "checkpoints/sam-3d-body-dinov3/model.ckpt"
MHR_PATH = SAM3D_DIR / "checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
DEMO_SCRIPT = SAM3D_DIR / "demo.py"

MHR_CONVERSION_DIR = Path("/home/haziq/MHR/tools/mhr_smpl_conversion")
MHR_TO_SMPL_SCRIPT = MHR_CONVERSION_DIR / "mhr_to_smpl.py"

RENDER_SCRIPT = Path(__file__).parent / "render_smplx_video.py"


def main():
    parser = argparse.ArgumentParser(
        description="Run SAM-3D-Body on a video and save results + rendered video."
    )
    parser.add_argument("video_path", help="Path to the input video file.")
    parser.add_argument(
        "--output_folder",
        default=None,
        help=(
            "Directory to write outputs. "
            "Defaults to a folder named after the video stem, "
            "placed next to the video file."
        ),
    )
    parser.add_argument(
        "--biomarker",
        type=str,
        nargs="+",
        default=[],
        help="Biomarker(s) to plot on the rendered video, e.g. 'left knee flexion'",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path).resolve()
    if not video_path.exists():
        print(f"Error: video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    if args.output_folder is None:
        output_folder = video_path.parent.parent / "sam3d" / video_path.stem
    else:
        output_folder = Path(args.output_folder).resolve()

    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Input : {video_path}")
    print(f"Output: {output_folder}")

    npz_path    = output_folder / f"{video_path.stem}_mhr_outputs.npz"
    json_path   = npz_path.with_suffix(".json")
    render_path = output_folder / f"{video_path.stem}_smplx_rendered.mp4"

    if npz_path.exists():
        print(f"\nSkipping SAM-3D-Body (NPZ already exists): {npz_path}")
    else:
        cmd = [
            "conda", "run", "-n", "sam_3d_body",
            "python", str(DEMO_SCRIPT),
            "--video_path", str(video_path),
            "--output_folder", str(output_folder),
            "--checkpoint_path", str(CHECKPOINT_PATH),
            "--mhr_path", str(MHR_PATH),
            "--detector_name", "sam3",
            "--save_npz",
            "--center_person_only",
        ]

        print("\nRunning:", " ".join(f'"{c}"' if " " in c else c for c in cmd), "\n")
        result = subprocess.run(cmd, cwd=str(SAM3D_DIR))

        if result.returncode != 0:
            print(f"\nERROR: SAM-3D-Body exited with code {result.returncode}. Skipping conversion.")
            sys.exit(result.returncode)

    # ── MHR → SMPL-X conversion ───────────────────────────────────────────────
    if not npz_path.exists():
        print(f"\nWARNING: Expected NPZ not found at {npz_path}. Skipping conversion.")
        sys.exit(0)

    if json_path.exists():
        print(f"\nSkipping conversion (JSON already exists): {json_path}")
    else:
        convert_cmd = [
            "conda", "run", "-n", "mhr",
            "python", str(MHR_TO_SMPL_SCRIPT),
            "--mhr_path", str(npz_path),
            "--out_json", str(json_path),
        ]

        print("\nConverting MHR → SMPL-X:")
        print(" ".join(f'"{c}"' if " " in c else c for c in convert_cmd), "\n")
        convert_result = subprocess.run(convert_cmd, cwd=str(MHR_CONVERSION_DIR))

        if convert_result.returncode != 0:
            print(f"\nERROR: mhr_to_smpl.py exited with code {convert_result.returncode}.")
            sys.exit(convert_result.returncode)

        print(f"\nConversion done → {json_path}")

    # ── SMPL-X render (always re-render) ────────────────────────────────────────
    render_cmd = [
        "conda", "run", "-n", "sam_3d_body",
        "python", str(RENDER_SCRIPT),
        str(video_path),
        str(json_path),
        str(npz_path),
        str(render_path),
    ]
    if args.biomarker:
        render_cmd += ["--biomarker"] + args.biomarker

    print("\nRendering SMPL-X video:")
    print(" ".join(f'"{c}"' if " " in c else c for c in render_cmd), "\n")
    render_result = subprocess.run(render_cmd)

    if render_result.returncode != 0:
        print(f"\nERROR: render_smplx_video.py exited with code {render_result.returncode}.")
        sys.exit(render_result.returncode)

    print(f"\nRender done → {render_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
