"""
Mesh generation utilities.

Provides:
- A rest-pose stub mesh (icosphere) for development without a GPU.
- The real SAM3DBody processing pipeline (commented out by default).
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from io import BytesIO
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# REST-POSE STUB  (used when USE_SAM3D=0)
# ---------------------------------------------------------------------------

def _make_icosphere(subdivisions: int = 3, radius: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Create a UV-sphere (icosphere approximation) as a rest-pose placeholder."""
    try:
        import trimesh
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
        return np.array(sphere.vertices, dtype=np.float32), np.array(sphere.faces, dtype=np.int32)
    except ImportError:
        # Fallback: simple cube
        verts = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
        ], dtype=np.float32) * radius
        faces = np.array([
            [0,1,2],[0,2,3],[4,6,5],[4,7,6],
            [0,4,5],[0,5,1],[2,6,7],[2,7,3],
            [0,3,7],[0,7,4],[1,5,6],[1,6,2],
        ], dtype=np.int32)
        return verts, faces


def _write_obj(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a simple Wavefront OBJ file (1-indexed faces)."""
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            # OBJ faces are 1-indexed
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def generate_stub_meshes(
    video_path: str,
    work_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[str, int, float]:
    """
    Generate a rest-pose icosphere for every frame of the input video.

    Returns (zip_path, frame_count, fps).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if n_frames <= 0:
        n_frames = 1  # safety

    verts, faces = _make_icosphere()

    obj_dir = os.path.join(work_dir, "objs")
    os.makedirs(obj_dir, exist_ok=True)

    if progress_callback:
        progress_callback(0, n_frames)
    for i in range(n_frames):
        _write_obj(os.path.join(obj_dir, f"frame_{i:04d}.obj"), verts, faces)
        if progress_callback:
            progress_callback(i + 1, n_frames)

    # meta.json
    meta = {"frame_count": n_frames, "fps": fps}
    meta_path = os.path.join(obj_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # zip everything
    zip_path = os.path.join(work_dir, "result.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(obj_dir)):
            zf.write(os.path.join(obj_dir, fname), fname)

    return zip_path, n_frames, fps


# ---------------------------------------------------------------------------
# SAM3DBody REAL PROCESSING  (Fast-SAM-3D-Body accelerated path)
# ---------------------------------------------------------------------------

import sys
import time
import torch

# Enable TensorFloat32 on Ampere+ GPUs (free ~10% speedup on matrix ops)
torch.set_float32_matmul_precision("high")

# ── Performance env vars (must be set before importing sam_3d_body) ──────────
os.environ.setdefault("GPU_HAND_PREP", "1")                  # GPU hand preprocessing
os.environ.setdefault("SKIP_KEYPOINT_PROMPT", "1")          # Skip slow keypoint prompt
os.environ.setdefault("LAYER_DTYPE", "fp32")                 # fp32 (required for multi-person)
os.environ.setdefault("USE_COMPILE", "1")                    # torch.compile
os.environ.setdefault("USE_COMPILE_BACKBONE", "1")
os.environ.setdefault("DECODER_COMPILE", "1")
os.environ.setdefault("COMPILE_MODE", "reduce-overhead")
os.environ.setdefault("COMPILE_WARMUP_BATCH_SIZES", "1")
os.environ.setdefault("BODY_INTERM_PRED_LAYERS", "0,1,2")    # fewer layers = faster decoder
os.environ.setdefault("HAND_INTERM_PRED_LAYERS", "0,1")
os.environ.setdefault("KEYPOINT_PROMPT_INTERM_INTERVAL", "999")  # disable keypoint prompt
os.environ.setdefault("MHR_NO_CORRECTIVES", "1")             # skip slow corrective blendshapes
os.environ.setdefault("IMG_SIZE", "512")                     # backbone image size
os.environ.setdefault("FOV_FAST", "1")                      # MoGe2 fast mode
os.environ.setdefault("FOV_MODEL", "s")                     # MoGe2-s (35M, fastest)
os.environ.setdefault("FOV_LEVEL", "0")                     # 1200 tokens (fewest)
# Auto-enable TRT engines when built (run build_trt_engines.sh first)
_FOV_TRT_ENGINE = os.path.expanduser("~/Fast-SAM-3D-Body/checkpoints/moge_trt/moge_dinov2_encoder_fp16.engine")
if os.path.exists(_FOV_TRT_ENGINE):
    os.environ.setdefault("FOV_TRT", "1")
    print(f"[mesh_gen] FOV TRT engine found, enabling FOV_TRT=1")
_YOLO_ENGINE = os.path.expanduser("~/Fast-SAM-3D-Body/checkpoints/yolo/yolo11m-pose.engine")
_BACKBONE_TRT_ENGINE = os.path.expanduser(
    "~/Fast-SAM-3D-Body/checkpoints/sam-3d-body-dinov3/backbone_trt/backbone_dinov3_fp16.engine"
)
if os.path.exists(_BACKBONE_TRT_ENGINE):
    os.environ.setdefault("USE_TRT_BACKBONE", "1")
    os.environ.setdefault("TRT_BACKBONE_PATH", _BACKBONE_TRT_ENGINE)
    print(f"[mesh_gen] Backbone TRT engine found, enabling USE_TRT_BACKBONE=1")
# ─────────────────────────────────────────────────────────────────────────────

# Switch between Fast-SAM-3D-Body (default) and original SAM3D via env var:
#   USE_FAST_SAM3D=1  (default) -> ~/Fast-SAM-3D-Body
#   USE_FAST_SAM3D=0            -> ~/sam-3d-body
_use_fast = os.getenv("USE_FAST_SAM3D", "1") == "1"
_sam3d_code_root = os.path.expanduser(
    "~/Fast-SAM-3D-Body" if _use_fast else "~/sam-3d-body"
)
print(f"[mesh_gen] SAM3D backend: {'Fast-SAM-3D-Body' if _use_fast else 'sam-3d-body (original)'}")
sys.path.insert(0, _sam3d_code_root)

from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from config import (
    SAM3D_CHECKPOINT, SAM3D_MHR_PATH, SAM3D_DETECTOR_NAME,
    SAM3D_DETECTOR_PATH, SAM3D_SEGMENTOR_NAME, SAM3D_SEGMENTOR_PATH,
    SAM3D_FOV_NAME, SAM3D_FOV_PATH, SAM3D_BBOX_THRESH, SAM3D_YOLO_MODEL,
)

_estimator: SAM3DBodyEstimator | None = None


def _get_estimator() -> SAM3DBodyEstimator:
    global _estimator
    if _estimator is not None:
        return _estimator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg = load_sam_3d_body(SAM3D_CHECKPOINT, device=device, mhr_path=SAM3D_MHR_PATH)

    human_detector, human_segmentor, fov_estimator = None, None, None

    if SAM3D_DETECTOR_NAME:
        from tools.build_detector import HumanDetector
        detector_kwargs = {}
        if SAM3D_DETECTOR_NAME == "yolo_pose":
            # Prefer compiled TRT engine for extra ~2x; falls back to .pt weights
            yolo_model = _YOLO_ENGINE if os.path.exists(_YOLO_ENGINE) else SAM3D_YOLO_MODEL
            print(f"[mesh_gen] YOLO model: {yolo_model}")
            detector_kwargs["model"] = yolo_model
        elif SAM3D_DETECTOR_PATH:
            detector_kwargs["path"] = SAM3D_DETECTOR_PATH
        human_detector = HumanDetector(
            name=SAM3D_DETECTOR_NAME,
            device=device,
            **detector_kwargs,
        )

    if SAM3D_SEGMENTOR_NAME:
        from tools.build_sam import HumanSegmentor
        human_segmentor = HumanSegmentor(name=SAM3D_SEGMENTOR_NAME, device=device, path=SAM3D_SEGMENTOR_PATH)

    if SAM3D_FOV_NAME:
        from tools.build_fov_estimator import FOVEstimator
        fov_estimator = FOVEstimator(name=SAM3D_FOV_NAME, device=device)

    _estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )
    return _estimator


def generate_sam3d_meshes(
    video_path: str,
    work_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[str, int, float]:
    """
    Run SAM3DBody on every frame and export per-frame OBJ meshes.

    Returns (zip_path, frame_count, fps).
    """
    estimator = _get_estimator()
    faces = estimator.faces  # (F, 3) int ndarray

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Some containers (e.g. iOS .mov) don't expose a reliable frame count via
    # CAP_PROP_FRAME_COUNT.  Leave n_frames=0 so the client shows an
    # indeterminate progress bar rather than a stuck 0%.

    obj_dir = os.path.join(work_dir, "objs")
    os.makedirs(obj_dir, exist_ok=True)

    if progress_callback:
        progress_callback(0, n_frames)

    _COMPILE_WARMUP_FRAMES = 3  # torch.compile finishes kernel compilation by frame 3

    # --- First pass: run inference on every frame, collect outputs --------
    # We store per-frame results so we can average shape_params across the
    # whole clip before writing any OBJ files.
    frame_outputs: List[Optional[dict]] = []
    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        outputs = estimator.process_one_image(frame_rgb, bbox_thr=SAM3D_BBOX_THRESH, use_mask=False)
        elapsed = time.perf_counter() - t0
        if idx < _COMPILE_WARMUP_FRAMES:
            print(f"[mesh_gen] frame {idx:04d}: {elapsed:.2f}s  (torch.compile warmup — will speed up)")
        elif idx == _COMPILE_WARMUP_FRAMES:
            print(f"[mesh_gen] frame {idx:04d}: {elapsed:.2f}s  ← torch.compile warmed up, full speed from here")
        else:
            print(f"[mesh_gen] frame {idx:04d}: {elapsed:.2f}s")

        best_output = None
        if len(outputs) > 0:
            # Pick the detection whose bounding-box center is closest to the
            # image center (there should always be exactly one person, but
            # just in case the detector fires on background figures).
            img_cy, img_cx = frame_rgb.shape[0] / 2.0, frame_rgb.shape[1] / 2.0
            best_output = min(
                outputs,
                key=lambda o: (
                    ((o["bbox"][0] + o["bbox"][2]) / 2.0 - img_cx) ** 2
                    + ((o["bbox"][1] + o["bbox"][3]) / 2.0 - img_cy) ** 2
                )
                if o.get("bbox") is not None
                else float("inf"),
            )

        frame_outputs.append(best_output)
        idx += 1
        if progress_callback:
            progress_callback(idx, n_frames)

    cap.release()
    actual_frames = idx

    # --- Average shape from first 10 valid frames -----------------------
    # Shape (betas) is person-specific and constant over a clip.  Using only
    # the first 10 frames is enough to get a stable estimate and means we
    # don't need to wait for the whole video.
    valid_shapes = [
        o["shape_params"]
        for o in frame_outputs
        if o is not None and o.get("shape_params") is not None
    ][:10]

    if valid_shapes:
        mean_shape = np.mean(valid_shapes, axis=0)  # (S,)
        print(f"[mesh_gen] Averaged shape over {len(valid_shapes)} frame(s) (first 10)")

        # Re-run FK for ALL frames with mean shape, batched in a single GPU
        # call — essentially free (~1-2 ms total regardless of clip length).
        valid_indices = [
            i for i, o in enumerate(frame_outputs)
            if o is not None and o.get("shape_params") is not None
        ]
        head = estimator.model.head_pose
        device = estimator.device
        B = len(valid_indices)

        mean_shape_t = torch.from_numpy(
            np.tile(mean_shape[None], (B, 1))
        ).float().to(device)
        global_trans = torch.zeros(B, 3, device=device)
        global_rots  = torch.from_numpy(np.stack([frame_outputs[i]["global_rot"]      for i in valid_indices])).float().to(device)
        body_poses   = torch.from_numpy(np.stack([frame_outputs[i]["body_pose_params"] for i in valid_indices])).float().to(device)
        scales       = torch.from_numpy(np.stack([frame_outputs[i]["scale_params"]     for i in valid_indices])).float().to(device)
        exprs        = torch.from_numpy(np.stack([frame_outputs[i]["expr_params"]      for i in valid_indices])).float().to(device)
        hand_poses   = None
        if frame_outputs[valid_indices[0]].get("hand_pose_params") is not None:
            hand_poses = torch.from_numpy(
                np.stack([frame_outputs[i]["hand_pose_params"] for i in valid_indices])
            ).float().to(device)

        with torch.no_grad():
            verts_batch, _, _, _, _ = head._mhr_forward_core(
                global_trans, global_rots, body_poses, hand_poses,
                scales, mean_shape_t, exprs,
                return_keypoints=False,
            )
            verts_np = verts_batch.cpu().numpy()  # (B, V, 3)  -- no flip, raw camera space

        for batch_i, frame_i in enumerate(valid_indices):
            frame_outputs[frame_i]["pred_vertices"] = verts_np[batch_i]
    else:
        print("[mesh_gen] No valid shape params found; skipping shape averaging")

    # --- Write OBJ files -------------------------------------------------
    for i, o in enumerate(frame_outputs):
        verts = o.get("pred_vertices") if o is not None else None
        if verts is not None:
            _write_obj(os.path.join(obj_dir, f"frame_{i:04d}.obj"), verts, faces)
        else:
            _write_obj(
                os.path.join(obj_dir, f"frame_{i:04d}.obj"),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32),
            )

    # meta.json
    meta = {"frame_count": actual_frames, "fps": fps}
    meta_path = os.path.join(obj_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    # zip everything
    zip_path = os.path.join(work_dir, "result.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(obj_dir)):
            zf.write(os.path.join(obj_dir, fname), fname)

    return zip_path, actual_frames, fps


# ---------------------------------------------------------------------------
# MHR params (compact binary output)
# ---------------------------------------------------------------------------


def _fk_and_get_mhr_params_single(
    estimator,
    output: dict,
    mean_shape: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Run MHR forward kinematics for a single frame and return the raw
    model_parameters[204] + cam_t[3] needed by the app's Dart FK engine.

    Returns (params_row_f32[207], valid_u8).
    Params layout: model_params(204) + cam_t(3).

    model_params[204] layout (set by _mhr_forward_core):
        [0:3]    global_trans * 10  (always zero; root translation is in cam_t)
        [3:6]    global_rot euler ZYX
        [6:136]  body_pose euler (130 joints)
        [136:204] scales (68 per-joint scale values, expanded from 28 PCA comps)
    """
    PARAMS_PER_FRAME = 207
    params_row = np.zeros(PARAMS_PER_FRAME, dtype=np.float32)

    if output.get("pred_cam_t") is None:
        return params_row, 0

    head   = estimator.model.head_pose
    device = estimator.device

    mean_shape_t = torch.from_numpy(mean_shape[None]).float().to(device)  # (1, S)
    global_trans = torch.zeros(1, 3, device=device)
    global_rots  = torch.from_numpy(output["global_rot"][None]).float().to(device)
    body_poses   = torch.from_numpy(output["body_pose_params"][None]).float().to(device)
    scales       = torch.from_numpy(output["scale_params"][None]).float().to(device)
    exprs        = torch.from_numpy(output["expr_params"][None]).float().to(device)
    hand_poses   = None
    if output.get("hand_pose_params") is not None:
        hand_poses = torch.from_numpy(output["hand_pose_params"][None]).float().to(device)

    with torch.no_grad():
        _, _, _, mhr_model_params, _ = head._mhr_forward_core(
            global_trans, global_rots, body_poses, hand_poses,
            scales, mean_shape_t, exprs,
            return_keypoints=False,
        )

    params_row[:204] = mhr_model_params[0].cpu().numpy().astype(np.float32)
    params_row[204:207] = np.asarray(output["pred_cam_t"], dtype=np.float32).ravel()[:3]
    return params_row, 1


def generate_sam3d_params(
    video_path: str,
    work_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    frame_ready_callback: Optional[Callable[[int, np.ndarray, int, float], None]] = None,
) -> Tuple[str, int, float]:
    """
    Run SAM3DBody on every frame, extract raw MHR model_parameters[204],
    and write a compact binary result for the app's Dart FK engine.

    As soon as mean_shape is established (after the first 10 valid frames),
    FK results are emitted immediately per frame via
    frame_ready_callback(frame_idx, params_row[207], valid_u8, fps).
    This allows callers to stream partial results without waiting for the
    full video to finish processing.

    Binary format (result.bin):
        Header  – 3 × int32  : [magic=0x4D485250, frame_count, param_count_per_frame=207]
        Header  – 2 × float32: [fps, focal_length]
        Frames  – frame_count × 207 × float32:
                    [model_params(204), cam_t(3)]
        Validity – frame_count × uint8 : 1 if valid, 0 if detection failed

    Returns (bin_path, frame_count, fps).
    """
    import struct

    estimator = _get_estimator()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1920
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    # Focal length fallback: SAM3DBody's own default when no FOV estimator is used
    # (see prepare_batch.py: focal = sqrt(h^2 + w^2)).
    # Overwritten with the actual model output values once valid frames arrive.
    import math as _math
    focal_length = _math.sqrt(float(img_w) ** 2 + float(img_h) ** 2)
    focal_lengths_collected: List[float] = []

    if progress_callback:
        progress_callback(0, n_frames)

    _COMPILE_WARMUP_FRAMES = 3
    _MEAN_SHAPE_FRAMES = 10  # collect this many valid frames before fixing mean_shape

    # Inference outputs buffered until mean_shape is established.
    frame_outputs: List[Optional[dict]] = []
    mean_shape: Optional[np.ndarray] = None
    valid_shape_count = 0

    # Final params arrays (filled either eagerly or in the post-loop pass).
    PARAMS_PER_FRAME = 207  # model_params(204) + cam_t(3)
    params_list: List[np.ndarray] = []  # one (76,) row per frame, in order
    valid_list:  List[int]        = []

    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()
        outputs = estimator.process_one_image(frame_rgb, bbox_thr=SAM3D_BBOX_THRESH, use_mask=False)
        elapsed = time.perf_counter() - t0
        if idx <= _COMPILE_WARMUP_FRAMES:
            print(f"[mesh_gen] frame {idx:04d}: {elapsed:.2f}s")
        else:
            print(f"[mesh_gen] frame {idx:04d}: {elapsed:.2f}s")

        best_output = None
        if outputs:
            img_cy, img_cx = frame_rgb.shape[0] / 2.0, frame_rgb.shape[1] / 2.0
            best_output = min(
                outputs,
                key=lambda o: (
                    ((o["bbox"][0] + o["bbox"][2]) / 2.0 - img_cx) ** 2
                    + ((o["bbox"][1] + o["bbox"][3]) / 2.0 - img_cy) ** 2
                )
                if o.get("bbox") is not None
                else float("inf"),
            )

        # Collect real focal length from the model output (set by FOV estimator or
        # SAM3DBody's default cam_int).  Use a running mean so streaming callbacks
        # always have the best available estimate.
        if best_output is not None and best_output.get("focal_length") is not None:
            fl = float(np.atleast_1d(best_output["focal_length"]).ravel()[0])
            focal_lengths_collected.append(fl)
            focal_length = float(np.mean(focal_lengths_collected))

        # Track raw output for potential retroactive processing.
        frame_outputs.append(best_output)

        # Accumulate valid shapes toward mean_shape.
        if best_output is not None and best_output.get("shape_params") is not None:
            valid_shape_count += 1

        # Check whether we just reached enough frames to fix mean_shape.
        if mean_shape is None and valid_shape_count >= _MEAN_SHAPE_FRAMES:
            collected = [
                o["shape_params"]
                for o in frame_outputs
                if o is not None and o.get("shape_params") is not None
            ][:_MEAN_SHAPE_FRAMES]
            mean_shape = np.mean(collected, axis=0)
            print(f"[mesh_gen] mean_shape established at frame {idx:04d}")

            # Retroactively process all buffered frames.
            for buf_idx, buf_out in enumerate(frame_outputs):
                if buf_out is not None and buf_out.get("shape_params") is not None:
                    row, v = _fk_and_get_mhr_params_single(estimator, buf_out, mean_shape)
                else:
                    row, v = np.zeros(PARAMS_PER_FRAME, dtype=np.float32), 0
                params_list.append(row)
                valid_list.append(v)
                if frame_ready_callback:
                    frame_ready_callback(buf_idx, row, v, fps, focal_length)
            # Note: frame_outputs already includes the current frame (appended
            # above), so the retroactive loop above has already processed it.
            # No need to process it again here.
        elif mean_shape is not None:
            # mean_shape is already established — process this frame immediately.
            if best_output is not None and best_output.get("shape_params") is not None:
                row, v = _fk_and_get_mhr_params_single(estimator, best_output, mean_shape)
            else:
                row, v = np.zeros(PARAMS_PER_FRAME, dtype=np.float32), 0
            params_list.append(row)
            valid_list.append(v)
            if frame_ready_callback:
                frame_ready_callback(idx, row, v, fps, focal_length)

        idx += 1
        if progress_callback:
            progress_callback(idx, n_frames)

    cap.release()
    actual_frames = idx

    # --- Post-loop: handle case where mean_shape was never established ----
    # (e.g. video with no detections, or fewer than _MEAN_SHAPE_FRAMES valid frames)
    if mean_shape is None:
        valid_shapes = [
            o["shape_params"]
            for o in frame_outputs
            if o is not None and o.get("shape_params") is not None
        ]
        if valid_shapes:
            mean_shape = np.mean(valid_shapes, axis=0)

        for buf_idx, buf_out in enumerate(frame_outputs):
            if mean_shape is not None and buf_out is not None and buf_out.get("shape_params") is not None:
                row, v = _fk_and_get_mhr_params_single(estimator, buf_out, mean_shape)
            else:
                row, v = np.zeros(PARAMS_PER_FRAME, dtype=np.float32), 0
            params_list.append(row)
            valid_list.append(v)
            if frame_ready_callback:
                frame_ready_callback(buf_idx, row, v, fps, focal_length)
    params_array = np.stack(params_list, axis=0) if params_list else np.zeros((0, PARAMS_PER_FRAME), dtype=np.float32)
    valid_array  = np.array(valid_list, dtype=np.uint8)

    print(f"[mesh_gen] Valid frames: {int(valid_array.sum())}/{actual_frames}")

    # --- Write compact binary --------------------------------------------
    MAGIC = 0x4D485250  # 'MHRP'
    bin_path = os.path.join(work_dir, "result.bin")
    with open(bin_path, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", actual_frames))
        f.write(struct.pack("<I", PARAMS_PER_FRAME))
        f.write(struct.pack("<f", fps))
        f.write(struct.pack("<f", focal_length))
        f.write(params_array.tobytes())
        f.write(valid_array.tobytes())

    size_kb = os.path.getsize(bin_path) / 1024
    print(f"[mesh_gen] result.bin: {size_kb:.1f} KB ({actual_frames} frames)")

    return bin_path, actual_frames, fps
