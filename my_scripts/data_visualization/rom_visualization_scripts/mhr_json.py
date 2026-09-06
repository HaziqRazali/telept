"""Load and reconstruct MHR poses stored by the Synthium video runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def load_mhr_json(path: str | Path) -> dict:
    """Load one ``video_inference.py`` JSON timeseries."""

    with Path(path).open() as handle:
        data = json.load(handle)
    if str(data.get("body_model", "")).lower().find("mhr") < 0:
        raise ValueError(f"JSON does not declare an MHR body model: {path}")
    if "frames" not in data or "fps" not in data:
        raise ValueError(f"MHR JSON is missing 'frames' or 'fps': {path}")
    return data


def select_mhr_json_frame(
    data: dict,
    time_s: float | None = None,
    frame_override: int | None = None,
) -> tuple[dict, int, float]:
    """Select a detected JSON frame by timestamp or zero-based frame index.

    JSON inference videos retain the original video timeline, including frames
    where no person was detected.  Timestamps therefore map directly through
    the JSON's stored FPS rather than through the shorter SAM-3D rendered
    timeline used by the NPZ visualizers.
    """

    fps = float(data["fps"])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Invalid JSON FPS: {data['fps']!r}")

    frames = data["frames"]
    by_index = {int(frame["frame_idx"]): frame for frame in frames}
    if not by_index:
        raise ValueError("MHR JSON contains no frames.")

    if frame_override is not None:
        frame_idx = int(frame_override)
    elif time_s is not None:
        if not np.isfinite(time_s):
            raise ValueError(f"Invalid timestamp: {time_s!r}")
        frame_idx = int(np.rint(float(time_s) * fps))
    else:
        raise ValueError("Provide either time_s or frame_override.")

    if frame_idx not in by_index:
        available = np.asarray(sorted(by_index), dtype=np.int64)
        nearest = int(available[np.argmin(np.abs(available - frame_idx))])
        raise ValueError(
            f"JSON frame {frame_idx} is unavailable; nearest stored frame is "
            f"{nearest}."
        )

    frame = by_index[frame_idx]
    if not frame.get("detected", False) or frame.get("pred_body_params") is None:
        raise ValueError(f"JSON frame {frame_idx} has no detected MHR pose.")
    return frame, frame_idx, frame_idx / fps


def _finite_vector(value, length: int) -> np.ndarray:
    if value is None:
        return np.zeros(length, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size < length or not np.isfinite(array[:length]).all():
        return np.zeros(length, dtype=np.float32)
    return array[:length]


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert one column-major 6-D rotation representation to a matrix."""

    value = np.asarray(rot6d, dtype=np.float32).reshape(6)
    a1 = value[:3]
    a2 = value[3:]
    b1 = a1 / max(float(np.linalg.norm(a1)), 1e-8)
    a2_orthogonal = a2 - b1 * float(np.dot(b1, a2))
    b2 = a2_orthogonal / max(float(np.linalg.norm(a2_orthogonal)), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=-1)


def _mat_to_euler_zyx(matrix: np.ndarray) -> np.ndarray:
    """Invert MHR's ``Rz(z) @ Ry(y) @ Rx(x)`` convention."""

    matrix = np.asarray(matrix, dtype=np.float64)
    x = np.arctan2(matrix[2, 1], matrix[2, 2])
    y = np.arctan2(
        -matrix[2, 0],
        np.sqrt(matrix[2, 1] ** 2 + matrix[2, 2] ** 2),
    )
    z = np.arctan2(matrix[1, 0], matrix[0, 0])
    return np.asarray([x, y, z], dtype=np.float32)


def _scale_params_68(value) -> np.ndarray:
    """Return the 68-dimensional MHR scale block stored in the JSON."""

    if value is None:
        return np.zeros(68, dtype=np.float32)
    scale = np.asarray(value, dtype=np.float32).reshape(-1)
    if scale.size >= 68 and np.isfinite(scale[:68]).all():
        return scale[:68]
    if scale.size != 28 or not np.isfinite(scale).all():
        return np.zeros(68, dtype=np.float32)

    basis_path = Path(
        "/home/haziq/sam-3d-body/checkpoints/"
        "sam-3d-body-dinov3/assets/mhr_scale_basis.npz"
    )
    if not basis_path.is_file():
        return np.zeros(68, dtype=np.float32)
    basis = np.load(basis_path)
    mean = np.asarray(basis["scale_mean"], dtype=np.float32)
    components = np.asarray(basis["scale_comps"], dtype=np.float32)
    return scale @ components + mean


def reconstruct_mhr_json_frame(frame: dict, mhr_model):
    """Reconstruct ``(vertices_m, joints_m, body_pose_params)`` from a frame.

    The JSON runner stores ``[global_rot6d(6) | body_pose(130)]``.  MHR's
    forward pass returns vertices and a 127-joint skeleton in centimetres;
    this function converts both to metres while preserving the model's stored
    coordinate frame.
    """

    params = np.asarray(frame["pred_body_params"], dtype=np.float32).reshape(-1)
    if params.size >= 136:
        global_orient = _mat_to_euler_zyx(_rot6d_to_matrix(params[:6]))
        body_pose = params[6:136]
    elif params.size >= 130:
        global_orient = np.zeros(3, dtype=np.float32)
        body_pose = params[:130]
    else:
        raise ValueError(
            f"MHR pose requires 130 or 136 values, got {params.size}."
        )

    model_params = torch.zeros(1, 204, dtype=torch.float32)
    model_params[0, 3:6] = torch.from_numpy(global_orient)
    model_params[0, 6:136] = torch.from_numpy(body_pose)
    model_params[0, 136:204] = torch.from_numpy(
        _scale_params_68(frame.get("pred_scale_params"))
    )
    shape = torch.from_numpy(_finite_vector(frame.get("shape_params"), 45)).unsqueeze(0)
    expression = torch.zeros(1, 72, dtype=torch.float32)

    with torch.no_grad():
        vertices_cm, skeleton_state = mhr_model(shape, model_params, expression)

    vertices_m = (vertices_cm[0] / 100.0).cpu().numpy().astype(np.float64)
    joints_m = (skeleton_state[0, :, :3] / 100.0).cpu().numpy().astype(np.float64)
    return vertices_m, joints_m, body_pose.astype(np.float64)


def load_mhr_model(mhr_root: str | Path = "/home/haziq/MHR"):
    """Load the Python MHR model and return ``(model, LOD1 faces)``."""

    from mhr.mhr import MHR

    root = Path(mhr_root)
    model = MHR.from_files(
        folder=root / "assets",
        device=torch.device("cpu"),
        lod=1,
    )
    model.eval()
    faces = np.asarray(model.character.mesh.faces, dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"Unexpected MHR topology shape: {faces.shape}")
    return model, faces


__all__ = [
    "load_mhr_json",
    "load_mhr_model",
    "reconstruct_mhr_json_frame",
    "select_mhr_json_frame",
]
