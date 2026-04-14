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
from typing import List, Tuple

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


def generate_stub_meshes(video_path: str, work_dir: str) -> Tuple[str, int, float]:
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

    for i in range(n_frames):
        _write_obj(os.path.join(obj_dir, f"frame_{i:04d}.obj"), verts, faces)

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
# SAM3DBody REAL PROCESSING  (uncomment when running on a GPU server)
# ---------------------------------------------------------------------------
#
# import sys, torch
# sys.path.insert(0, os.path.expanduser("~/sam-3d-body"))
#
# from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
# from config import (
#     SAM3D_CHECKPOINT, SAM3D_MHR_PATH, SAM3D_DETECTOR_NAME,
#     SAM3D_DETECTOR_PATH, SAM3D_SEGMENTOR_NAME, SAM3D_SEGMENTOR_PATH,
#     SAM3D_FOV_NAME, SAM3D_FOV_PATH, SAM3D_BBOX_THRESH,
# )
#
# _estimator: SAM3DBodyEstimator | None = None
#
#
# def _get_estimator() -> SAM3DBodyEstimator:
#     global _estimator
#     if _estimator is not None:
#         return _estimator
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model, model_cfg = load_sam_3d_body(SAM3D_CHECKPOINT, device=device, mhr_path=SAM3D_MHR_PATH)
#
#     human_detector, human_segmentor, fov_estimator = None, None, None
#
#     if SAM3D_DETECTOR_NAME:
#         from tools.build_detector import HumanDetector
#         human_detector = HumanDetector(name=SAM3D_DETECTOR_NAME, device=device, path=SAM3D_DETECTOR_PATH)
#
#     if SAM3D_SEGMENTOR_NAME:
#         from tools.build_sam import HumanSegmentor
#         human_segmentor = HumanSegmentor(name=SAM3D_SEGMENTOR_NAME, device=device, path=SAM3D_SEGMENTOR_PATH)
#
#     if SAM3D_FOV_NAME:
#         from tools.build_fov_estimator import FOVEstimator
#         fov_estimator = FOVEstimator(name=SAM3D_FOV_NAME, device=device, path=SAM3D_FOV_PATH)
#
#     _estimator = SAM3DBodyEstimator(
#         sam_3d_body_model=model,
#         model_cfg=model_cfg,
#         human_detector=human_detector,
#         human_segmentor=human_segmentor,
#         fov_estimator=fov_estimator,
#     )
#     return _estimator
#
#
# def generate_sam3d_meshes(video_path: str, work_dir: str) -> Tuple[str, int, float]:
#     """
#     Run SAM3DBody on every frame and export per-frame OBJ meshes.
#
#     Returns (zip_path, frame_count, fps).
#     """
#     estimator = _get_estimator()
#     faces = estimator.faces  # (F, 3) int ndarray
#
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {video_path}")
#
#     fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#     n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     obj_dir = os.path.join(work_dir, "objs")
#     os.makedirs(obj_dir, exist_ok=True)
#
#     idx = 0
#     while True:
#         ret, frame_bgr = cap.read()
#         if not ret:
#             break
#         frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
#
#         outputs = estimator.process_one_image(frame_rgb, bbox_thr=SAM3D_BBOX_THRESH, use_mask=False)
#
#         if len(outputs) > 0:
#             verts = outputs[0].get("pred_vertices", outputs[0].get("vertices"))
#             if verts is not None:
#                 _write_obj(os.path.join(obj_dir, f"frame_{idx:04d}.obj"), verts, faces)
#             else:
#                 # Write an empty OBJ as placeholder
#                 _write_obj(os.path.join(obj_dir, f"frame_{idx:04d}.obj"),
#                            np.zeros((0, 3), dtype=np.float32),
#                            np.zeros((0, 3), dtype=np.int32))
#         else:
#             _write_obj(os.path.join(obj_dir, f"frame_{idx:04d}.obj"),
#                        np.zeros((0, 3), dtype=np.float32),
#                        np.zeros((0, 3), dtype=np.int32))
#         idx += 1
#
#     cap.release()
#     actual_frames = idx
#
#     # meta.json
#     meta = {"frame_count": actual_frames, "fps": fps}
#     meta_path = os.path.join(obj_dir, "meta.json")
#     with open(meta_path, "w") as f:
#         json.dump(meta, f)
#
#     # zip everything
#     zip_path = os.path.join(work_dir, "result.zip")
#     with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
#         for fname in sorted(os.listdir(obj_dir)):
#             zf.write(os.path.join(obj_dir, fname), fname)
#
#     return zip_path, actual_frames, fps
