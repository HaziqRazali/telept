#!/usr/bin/env python3
"""
gen_mhr_fixture.py
==================
Generates a ground-truth fixture for the Dart MHR FK unit test.

What it does
------------
1.  Reads model_params[204] from a real video frame via SAM3DBody inference.
2.  Runs pymomentum MHR forward pass → skinned vertices in cm.
3.  Saves:
      out_dir/model_params.bin   – raw float32[204]  (input to Dart FK)
      out_dir/expected_verts.bin – raw float32[V*3]  (expected output, cm)
      out_dir/manifest.json      – n_verts, max_err_threshold_cm, etc.

The Dart test loads these files, runs the same FK, and asserts that
max(|dart_verts - expected_verts|) < threshold_cm (default 0.05 cm = 0.5 mm).

Usage
-----
  conda activate telept_server
  cd /home/haziq/telept/my_scripts
  python gen_mhr_fixture.py \\
      /home/haziq/telept/telept/pushup_trimmed.mp4 \\
      --frame 20 \\
      --out /home/haziq/telept/app/mobile/test/fixtures/mhr
"""

import argparse
import json
import os
import struct
import sys

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAM3D_ROOT    = os.path.expanduser("~/Fast-SAM-3D-Body")
CHECKPOINT    = os.path.expanduser("~/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt")
MHR_PATH      = os.path.expanduser("~/MHR/assets/mhr_model.pt")
DETECTOR_NAME = "yolo_pose"
BBOX_THRESH   = 0.3

# Use the same env-vars as mesh_gen.py (needed before importing sam_3d_body)
os.environ.setdefault("GPU_HAND_PREP",              "1")
os.environ.setdefault("SKIP_KEYPOINT_PROMPT",       "1")
os.environ.setdefault("LAYER_DTYPE",                "fp32")
os.environ.setdefault("USE_COMPILE",                "0")
os.environ.setdefault("USE_COMPILE_BACKBONE",       "0")
os.environ.setdefault("DECODER_COMPILE",            "0")
os.environ.setdefault("MHR_NO_CORRECTIVES",         "1")
os.environ.setdefault("IMG_SIZE",                   "512")
os.environ.setdefault("FOV_FAST",                   "1")
os.environ.setdefault("FOV_MODEL",                  "s")
os.environ.setdefault("FOV_LEVEL",                  "0")

# ---------------------------------------------------------------------------
# Imports (same pattern as verify_dart_projection.py)
# ---------------------------------------------------------------------------
sys.path.insert(0, SAM3D_ROOT)

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.build_detector import HumanDetector


def load_frame(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame  # BGR uint8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--frame", type=int, default=20)
    parser.add_argument("--out", default="/home/haziq/telept/app/mobile/test/fixtures/mhr",
                        help="Output directory for fixture files")
    parser.add_argument("--threshold-cm", type=float, default=0.05,
                        help="Max allowed Dart FK error in cm (default 0.05 = 0.5 mm)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── Load frame ──────────────────────────────────────────────────────────
    print(f"[fixture] Reading frame {args.frame} from {args.video}")
    bgr = load_frame(args.video, args.frame)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── Load model ──────────────────────────────────────────────────────────
    print("[fixture] Loading SAM3DBody model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_sam_3d_body(CHECKPOINT, device=device, mhr_path=MHR_PATH)

    yolo_engine = os.path.expanduser("~/Fast-SAM-3D-Body/checkpoints/yolo/yolo11m-pose.engine")
    yolo_model  = yolo_engine if os.path.exists(yolo_engine) else \
                  os.path.expanduser("~/Fast-SAM-3D-Body/checkpoints/yolo/yolo11m-pose.pt")
    human_detector = HumanDetector(name=DETECTOR_NAME, device=device, model=yolo_model)

    from tools.build_fov_estimator import FOVEstimator
    fov_estimator = FOVEstimator(name="moge2", device=device)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=None,
        fov_estimator=fov_estimator,
    )

    # ── Run inference ───────────────────────────────────────────────────────
    print(f"[fixture] Running inference on frame {args.frame}...")
    outputs = estimator.process_one_image(rgb, bbox_thr=BBOX_THRESH, use_mask=False)
    assert len(outputs) > 0, "No person detected — try a different --frame"
    out = outputs[0]

    # ── Extract model_params[204] the same way mesh_gen.py does ─────────────
    print("[fixture] Extracting model_params[204]...")
    head   = estimator.model.head_pose
    dev    = estimator.device

    shape_params = out["shape_params"]
    expr_params  = out["expr_params"]

    mean_shape_t = torch.from_numpy(shape_params[None]).float().to(dev)
    global_trans = torch.zeros(1, 3, device=dev)
    global_rots  = torch.from_numpy(out["global_rot"][None]).float().to(dev)
    body_poses   = torch.from_numpy(out["body_pose_params"][None]).float().to(dev)
    scales       = torch.from_numpy(out["scale_params"][None]).float().to(dev)
    exprs        = torch.from_numpy(expr_params[None]).float().to(dev)
    hand_poses   = None
    if out.get("hand_pose_params") is not None:
        hand_poses = torch.from_numpy(out["hand_pose_params"][None]).float().to(dev)

    with torch.no_grad():
        _, _, _, mhr_model_params, _ = head._mhr_forward_core(
            global_trans, global_rots, body_poses, hand_poses,
            scales, mean_shape_t, exprs,
            return_keypoints=False,
        )
    model_params_204 = mhr_model_params[0].cpu().numpy().astype(np.float32)  # (204,)

    # ── Re-run pymomentum LOD3 FK to get expected vertices ───────────────────
    # Use the mhr_new env's MHR.from_files(lod=3) — the same pipeline that
    # generated the app's assets/mhr/ files.
    # We do this via a subprocess (different conda env).
    print("[fixture] Running LOD3 pymomentum FK via mhr_new env...")
    params_path = os.path.join(args.out, "model_params.bin")
    model_params_204.tofile(params_path)   # write params first so subprocess can read

    import subprocess, tempfile
    lod3_script = os.path.join(os.path.dirname(__file__), "_gen_fixture_lod3_worker.py")
    verts_path  = os.path.join(args.out, "expected_verts.bin")

    result = subprocess.run(
        ["conda", "run", "-n", "mhr_new", "python3", lod3_script,
         "--params", params_path, "--out", verts_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("[fixture] lod3 worker stderr:", result.stderr[-2000:])
        raise RuntimeError(f"LOD3 worker failed with code {result.returncode}")
    print(result.stdout.strip())

    expected_verts_cm = np.fromfile(verts_path, dtype=np.float32).reshape(-1, 3)
    n_verts = expected_verts_cm.shape[0]
    print(f"\n[fixture] n_verts = {n_verts}")
    print(f"[fixture] expected_verts Y range: [{expected_verts_cm[:,1].min():.3f}, {expected_verts_cm[:,1].max():.3f}] cm")

    manifest_path = os.path.join(args.out, "fixture_manifest.json")
    manifest = {
        "n_verts": n_verts,
        "n_params": 204,
        "source_video": args.video,
        "source_frame": args.frame,
        "threshold_cm": args.threshold_cm,
        "description": (
            "model_params.bin: float32[204] input to Dart FK. "
            "expected_verts.bin: float32[n_verts*3] ground-truth pymomentum output in cm "
            "(Y-UP, before the [1,2]*=-1 flip in sam3d_body.py)."
        ),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[fixture] Saved:")
    print(f"  {params_path}  ({os.path.getsize(params_path)} bytes)")
    print(f"  {verts_path}   ({os.path.getsize(verts_path)} bytes)")
    print(f"  {manifest_path}")
    print(f"\n[fixture] Dart test should assert: max|dart - expected| < {args.threshold_cm} cm")
    print("[fixture] Done.")


if __name__ == "__main__":
    main()
