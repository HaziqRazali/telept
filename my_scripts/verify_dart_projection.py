#!/usr/bin/env python3
"""
verify_dart_projection.py
=========================
Runs FastSAM3DBody on a single video frame and verifies that the Dart app's
pipeline would produce an identical projected mesh.

Three paths are compared side-by-side:

  (A) REFERENCE  — FastSAM3DBody's own renderer.py renders the mesh.
                   Ground truth: this is what the .mp4 video shows.

  (B) RERUN_FK   — We take the raw model_params[204] that the server would
                   write to the binary, feed them back into pymomentum MHR,
                   and re-project using our formula.  This isolates whether
                   the formula is correct independent of any Dart issues.

  (C) DART_PROJ  — We take (B)'s vertices and apply the exact Dart projection
                   math (Python translation) to produce projected 2D coords.
                   Diff from (B) reveals any remaining formula errors.

Output: verify_output.png  — a 4-panel image:
  [frame + renderer.py overlay]  [frame + RERUN_FK overlay]
  [frame + DART_PROJ overlay]    [numerical diff table]

Usage
-----
  conda activate telept_server
  cd /home/haziq/telept/my_scripts
  python verify_dart_projection.py /home/haziq/telept/telept/pushup_trimmed.mp4 \
      --frame 20 --out verify_output.png

Requirements: telept_server conda env (FastSAM3DBody, pymomentum, cv2, etc.)
"""

import argparse
import os
import sys

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAM3D_ROOT      = os.path.expanduser("~/Fast-SAM-3D-Body")
CHECKPOINT      = os.path.expanduser("~/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt")
MHR_PATH        = os.path.expanduser("~/MHR/assets/mhr_model.pt")
DETECTOR_NAME   = "yolo_pose"
BBOX_THRESH     = 0.5

# Use the same env-vars as mesh_gen.py (needed before importing sam_3d_body)
os.environ.setdefault("GPU_HAND_PREP",              "1")
os.environ.setdefault("SKIP_KEYPOINT_PROMPT",       "1")
os.environ.setdefault("LAYER_DTYPE",                "fp32")
os.environ.setdefault("USE_COMPILE",                "0")   # off for this debug script
os.environ.setdefault("USE_COMPILE_BACKBONE",       "0")
os.environ.setdefault("DECODER_COMPILE",            "0")
os.environ.setdefault("MHR_NO_CORRECTIVES",         "1")
os.environ.setdefault("IMG_SIZE",                   "512")
os.environ.setdefault("FOV_FAST",                   "1")
os.environ.setdefault("FOV_MODEL",                  "s")
os.environ.setdefault("FOV_LEVEL",                  "0")

sys.path.insert(0, SAM3D_ROOT)

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from sam_3d_body.visualization.renderer import Renderer
from tools.build_detector import HumanDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_frame(video_path: str, frame_idx: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, bgr = cap.read()
    cap.release()
    assert ret, f"Could not read frame {frame_idx} from {video_path}"
    return bgr  # BGR uint8


def project_vertices(verts_m: np.ndarray, cam_t: np.ndarray,
                     focal: float, cx: float, cy: float) -> np.ndarray:
    """
    Projects MHR vertices (metres) onto image pixels using the same math as
    renderer.py, but applied directly to pred_vertices without re-applying R_x(180°).

    pred_vertices from SAM3DBody already have global_rot applied, which folds in
    R_x(180°): head is at Y≈-1.7m, feet at Y≈0.

    renderer.py applies R_x(180°) again to the mesh (flipping Y and Z back), then
    places the camera at (-tx, ty, tz) and looks along -Z (pyrender/OpenGL).

    Working through the combined transform:
      V_world = R_x(180°) @ V_orig  →  y_world = -y_orig,  z_world = -z_orig
      camera at (-tx, ty, tz),  looks along -Z  (OpenGL convention)
      V_cam.x = V_world.x - (-tx) = V_world.x + tx = V_orig.x + tx
      V_cam.y = V_world.y - ty     = -V_orig.y - ty
      V_cam.z = V_world.z - tz     = -V_orig.z - tz   (negative = in front of camera)
      depth   = -V_cam.z           = V_orig.z + tz     (positive for visible verts)
      u = focal * V_cam.x / depth  + cx = focal * (X + tx) / (Z + tz) + cx
      v = cy - focal * V_cam.y / depth  = cy - focal * (-Y - ty) / (Z + tz)
                                        = cy + focal * (Y + ty) / (Z + tz)

    Returns (N, 2) float32 array of (u, v) pixel coordinates.
    """
    X, Y, Z = verts_m[:, 0], verts_m[:, 1], verts_m[:, 2]
    tx, ty, tz = cam_t[0], cam_t[1], cam_t[2]

    depth = tz - Z          # tz - Z  (positive for visible Y-UP vertices)
    valid = depth > 1e-6

    u = np.where(valid, focal * (X + tx) / depth + cx, cx)
    v = np.where(valid, cy - focal * (Y - ty) / depth, cy)   # Y-UP: cy - fy*(Y-ty)/depth

    return np.stack([u, v], axis=1).astype(np.float32)


def project_vertices_ydown(verts_m: np.ndarray, cam_t: np.ndarray,
                           focal: float, cx: float, cy: float) -> np.ndarray:
    """
    Projects Y,Z-flipped pred_vertices (from sam3d_body.py verts[..,[1,2]]*=-1)
    onto image pixels.  This matches the 2D projection done inside sam3d_body.py
    and by extension renderer.py (ground truth).

    depth = Z_down + tz   (positive for visible objects)
    u     = focal*(X+tx)/depth + cx
    v     = focal*(Y_down+ty)/depth + cy   (Y_down is already negative for head)
    """
    X, Y, Z = verts_m[:, 0], verts_m[:, 1], verts_m[:, 2]
    tx, ty, tz = cam_t[0], cam_t[1], cam_t[2]
    depth = Z + tz
    valid = depth > 1e-6
    u = np.where(valid, focal * (X + tx) / depth + cx, cx)
    v = np.where(valid, focal * (Y + ty) / depth + cy, cy)
    return np.stack([u, v], axis=1).astype(np.float32)


def draw_mesh_overlay(bgr: np.ndarray, uv: np.ndarray,
                      faces: np.ndarray, color=(100, 200, 255),
                      alpha=0.55) -> np.ndarray:
    """Draw a semi-transparent mesh overlay onto a BGR image."""
    out = bgr.astype(np.float32).copy()
    h, w = bgr.shape[:2]
    overlay = np.zeros_like(out)

    for f in faces:
        pts = uv[f].astype(np.int32)
        if np.all((pts[:, 0] >= -w) & (pts[:, 0] < 2*w) &
                  (pts[:, 1] >= -h) & (pts[:, 1] < 2*h)):
            cv2.fillConvexPoly(overlay, pts, color)

    out = cv2.addWeighted(out, 1.0 - alpha, overlay, alpha, 0)
    return out.clip(0, 255).astype(np.uint8)


def mhr_params_to_verts(estimator, model_params_204: np.ndarray,
                         shape_params: np.ndarray,
                         expr_params: np.ndarray) -> np.ndarray:
    """
    Re-run the MHR forward pass from raw model_params[204].
    Mirrors exactly what _fk_and_get_mhr_params_single stores and what the
    Dart code would do (param_transform → FK → LBS → * 0.01).

    Returns (V, 3) float32 vertices in metres.
    """
    head   = estimator.model.head_pose
    device = estimator.device

    mp   = torch.from_numpy(model_params_204[None]).float().to(device)     # (1, 204)
    sh   = torch.from_numpy(shape_params[None]).float().to(device)         # (1, S)
    expr = torch.from_numpy(expr_params[None]).float().to(device)          # (1, E)

    with torch.no_grad():
        skinned_verts, _ = estimator.model.head_pose.mhr(sh, mp, expr,
                                                          head.apply_correctives)
    verts_cm = skinned_verts[0].cpu().numpy()   # (V, 3) in cm (pymomentum native)
    return verts_cm * 0.01                       # → metres


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--frame", type=int, default=20, help="Frame index to inspect")
    parser.add_argument("--out", default="verify_output.png")
    args = parser.parse_args()

    print(f"[verify] Loading frame {args.frame} from {args.video}")
    bgr = load_frame(args.video, args.frame)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # ── Load model ──────────────────────────────────────────────────────────
    print("[verify] Loading SAM3DBody model...")
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
    faces = estimator.faces   # (F, 3)

    # ── Run inference ───────────────────────────────────────────────────────
    print(f"[verify] Running inference on frame {args.frame}...")
    outputs = estimator.process_one_image(rgb, bbox_thr=BBOX_THRESH, use_mask=False)

    assert len(outputs) > 0, "No person detected in this frame — try a different --frame"
    out = outputs[0]

    # Focal length the model used
    focal_length = float(np.atleast_1d(out["focal_length"]).ravel()[0])
    print(f"[verify] focal_length = {focal_length:.1f} px")

    # ── (A) renderer.py — ground truth ──────────────────────────────────────
    print("[verify] Rendering (A) reference via renderer.py ...")
    renderer = Renderer(focal_length=focal_length, faces=faces)
    ref_verts = out["pred_vertices"]          # (V, 3) metres
    cam_t     = out["pred_cam_t"]             # (3,) metres
    print(f"[verify] pred_vertices range:  Y=[{ref_verts[:,1].min():.3f}, {ref_verts[:,1].max():.3f}] m")
    print(f"[verify] cam_t = {cam_t}")

    # Renderer expects uint8 HxWx3 RGB image when full_frame=False
    rendered_A_img = renderer(ref_verts, cam_t, rgb.copy())
    rendered_A_bgr = (rendered_A_img * 255).clip(0, 255).astype(np.uint8)
    rendered_A_bgr = cv2.cvtColor(rendered_A_bgr, cv2.COLOR_RGB2BGR)

    # ── Extract model_params[204] the way the server does ───────────────────
    print("[verify] Extracting model_params[204] via _fk_and_get_mhr_params_single logic...")
    head   = estimator.model.head_pose
    dev    = estimator.device

    # Use frame's own shape (no averaging needed for a single-frame check)
    shape_params = out["shape_params"]          # (S,)
    expr_params  = out["expr_params"]           # (E,)

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
    print(f"[verify] model_params[204] sample (first 10): {model_params_204[:10]}")

    # ── (B) Re-run FK from model_params[204] ────────────────────────────────
    print("[verify] Re-running FK from model_params[204] (path B)...")
    rerun_verts = mhr_params_to_verts(estimator, model_params_204, shape_params, expr_params)
    print(f"[verify] rerun_verts range:    Y=[{rerun_verts[:,1].min():.3f}, {rerun_verts[:,1].max():.3f}] m")

    # FK roundtrip: rerun_verts (Y-UP) vs ref_verts (Y-DOWN pred_verts).
    # sam3d_body.py applies [1,2]*=-1 AFTER _mhr_forward_core, so flip rerun_verts
    # before comparing.
    rerun_flipped = rerun_verts.copy()
    rerun_flipped[:, [1, 2]] *= -1
    diff_verts = np.abs(ref_verts - rerun_flipped)
    print(f"[verify] FK roundtrip vertex error: max={diff_verts.max():.6f}m  mean={diff_verts.mean():.6f}m")

    # ── (C) Dart projection math (Python translation) ────────────────────────
    # Exact same formula as mesh_viewer.dart _MeshPainterProjected (Y-UP formula)
    cam_t_arr = np.asarray(cam_t, dtype=np.float32).ravel()[:3]

    # Project rerun_verts (Y-UP Dart FK) with Y-UP formula → what the app shows (B)
    uv_dart = project_vertices(rerun_verts, cam_t_arr, focal_length, cx, cy)
    # Project ref_verts (Y-DOWN pred_verts) with Y-DOWN formula → sanity check (C)
    uv_ref  = project_vertices_ydown(ref_verts, cam_t_arr, focal_length, cx, cy)

    # Projection diff: ground-truth pixel positions (uv_ref) vs Dart formula (uv_dart)
    uv_diff = np.abs(uv_dart - uv_ref)
    print(f"[verify] (B dart vs C ref) projection diff: max={uv_diff.max():.2f}px  mean={uv_diff.mean():.2f}px")

    # ── Draw overlays B and C ────────────────────────────────────────────────
    rendered_B_bgr = draw_mesh_overlay(bgr, uv_dart, faces, color=(100, 200, 255), alpha=0.55)
    rendered_C_bgr = draw_mesh_overlay(bgr, uv_ref,  faces, color=(80, 255, 140), alpha=0.55)

    # ── Panel layout ─────────────────────────────────────────────────────────
    # Resize panels to same height
    panel_h = 480
    def resize_h(img, h=panel_h):
        scale = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), h))

    pA = resize_h(rendered_A_bgr)
    pB = resize_h(rendered_B_bgr)
    pC = resize_h(rendered_C_bgr)

    # Stats panel
    stats_w = pA.shape[1]
    stats   = np.zeros((panel_h, stats_w, 3), dtype=np.uint8)
    lines = [
        "NUMERICAL VERIFICATION",
        "",
        f"Frame:        {args.frame}",
        f"focal_length: {focal_length:.1f} px",
        f"cam_t:        [{cam_t_arr[0]:.3f}, {cam_t_arr[1]:.3f}, {cam_t_arr[2]:.3f}]",
        "",
        "FK roundtrip (after Y,Z flip):",
        f"  max:  {diff_verts.max()*100:.3f} cm",
        f"  mean: {diff_verts.mean()*100:.4f} cm",
        "",
        "Projection diff (B dart vs C ref):",
        f"  max:  {uv_diff.max():.2f} px",
        f"  mean: {uv_diff.mean():.2f} px",
        "",
        "(A) renderer.py (ground truth)",
        "(B) Y-UP Dart FK + Dart formula",
        "(C) Y-DOWN pred_verts + ref formula",
        "",
        "B==C==A means pipeline is correct",
    ]
    for i, line in enumerate(lines):
        cv2.putText(stats, line, (10, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    row1 = np.hstack([pA, pB])
    row2 = np.hstack([pC, stats])
    min_w = min(row1.shape[1], row2.shape[1])
    out_img = np.vstack([row1[:, :min_w], row2[:, :min_w]])

    # Labels
    for label, x in [("(A) renderer.py reference", 10),
                      ("(B) Y-UP Dart FK + Dart formula", pA.shape[1] + 10)]:
        cv2.putText(out_img, label, (x, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2, cv2.LINE_AA)
    for label, x in [("(C) Y-DOWN pred_verts + ref formula", 10),
                      ("Numerical stats", pA.shape[1] + 10)]:
        cv2.putText(out_img, label, (x, panel_h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2, cv2.LINE_AA)

    cv2.imwrite(args.out, out_img)
    print(f"\n[verify] Saved comparison image → {args.out}")
    print("\n[verify] Summary:")
    print(f"  FK roundtrip error (flip→compare):       max {diff_verts.max()*100:.3f} cm  mean {diff_verts.mean()*100:.4f} cm")
    print(f"  Projection diff (B_dart vs C_ref):       max {uv_diff.max():.2f} px  mean {uv_diff.mean():.2f} px")
    print("  → If FK error < 1mm and projection diff < 1px, the Dart pipeline is correct.")


if __name__ == "__main__":
    main()
