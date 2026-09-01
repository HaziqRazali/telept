"""Shared RGBD geometry for human and MMPose keypoint evaluation.

The module deliberately accepts semantic 2D points and returns derived 3D
points/angles. Annotation files should keep the 2D points as their source of
truth; derived values can be recomputed with this module.
"""
from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Mapping

import numpy as np


_SIDES = {"left": "Left", "right": "Right"}

METRIC_TEMPLATES: dict[str, tuple[str, ...]] = {}
METRIC_LABELS: dict[str, str] = {}


def _register(key: str, label: str, points: tuple[str, ...]) -> None:
    METRIC_TEMPLATES[key] = points
    METRIC_LABELS[key] = label


def _torso_pts(side_limb: str) -> tuple[str, ...]:
    # Only the working-side joints are needed. For the shoulder plane-angle tests
    # the ROM is measured as the limb elevation from the body's down axis, where
    # down = side_hip - side_shoulder (the trunk's inferior direction). So the
    # annotator places: shoulder (for the arm origin and the down reference),
    # elbow (limb direction), and hip (the down reference).
    return (f"{side_limb}_shoulder", f"{side_limb}_elbow", f"{side_limb}_hip")


def _pelvis_pts(side_limb: str) -> tuple[str, ...]:
    # Hip flexion: the thigh (knee - hip) elevation from the trunk down axis,
    # where down = side_hip - side_shoulder. Points: shoulder (down ref), hip
    # (thigh origin + down ref), knee (thigh direction).
    return (f"{side_limb}_shoulder", f"{side_limb}_hip", f"{side_limb}_knee")


for _side, _cap in _SIDES.items():
    # elbow
    _register(f"{_side}_elbow_flexion", f"{_cap} elbow flexion",
              (f"{_side}_shoulder", f"{_side}_elbow", f"{_side}_wrist"))
    _register(f"{_side}_elbow_extension", f"{_cap} elbow extension",
              (f"{_side}_shoulder", f"{_side}_elbow", f"{_side}_wrist"))
    # shoulder
    _register(f"{_side}_shoulder_flexion", f"{_cap} shoulder flexion", _torso_pts(_side))
    _register(f"{_side}_shoulder_extension", f"{_cap} shoulder extension", _torso_pts(_side))
    _register(f"{_side}_shoulder_abduction", f"{_cap} shoulder abduction", _torso_pts(_side))
    _register(f"{_side}_shoulder_adduction", f"{_cap} shoulder adduction", _torso_pts(_side))
    _register(f"{_side}_shoulder_flexion_extension", f"{_cap} shoulder flexion/extension", _torso_pts(_side))
    # hip (abduction deferred - requires a different view/frame)
    _register(f"{_side}_hip_flexion", f"{_cap} hip flexion", _pelvis_pts(_side))
    # knee
    _register(f"{_side}_knee_flexion", f"{_cap} knee flexion",
              (f"{_side}_hip", f"{_side}_knee", f"{_side}_ankle"))
    _register(f"{_side}_knee_extension", f"{_cap} knee extension",
              (f"{_side}_hip", f"{_side}_knee", f"{_side}_ankle"))
    # ankle
    _register(f"{_side}_ankle_dorsiflexion", f"{_cap} ankle dorsiflexion",
              (f"{_side}_knee", f"{_side}_ankle", f"{_side}_big_toe"))
    _register(f"{_side}_ankle_plantarflexion", f"{_cap} ankle plantarflexion",
              (f"{_side}_knee", f"{_side}_ankle", f"{_side}_big_toe"))

# Hinge-joint ROM tests (included angle, intrinsic) need no reference axis and
# are measured at a single frame. Plane-angle tests (shoulder/hip) are measured
# against a neutral reference axis, so they use a T1=neutral / T2=peak task.
_HINGE_MARKERS = ("elbow", "knee", "ankle")


def metric_needs_reference(metric: str) -> bool:
    """True for plane-angle ROM tests that need a neutral reference axis."""
    return not any(marker in metric for marker in _HINGE_MARKERS)


def metric_pose_points(metric: str) -> tuple[list[str] | None, list[str]]:
    """Per-pose required points for a metric.

    Returns ``(t1_points, t2_points)``. ``t1`` is the neutral reference pose
    (shoulder + hip on the working side) and may be ``None`` for hinge tests.
    ``t2`` is the peak/measurement pose (the moving limb points).
    """
    side = metric.split("_", 1)[0]
    if not metric_needs_reference(metric):
        return None, list(METRIC_TEMPLATES[metric])
    if "shoulder" in metric:
        return [f"{side}_shoulder", f"{side}_hip"], [f"{side}_shoulder", f"{side}_elbow"]
    if "hip" in metric:
        return [f"{side}_shoulder", f"{side}_hip"], [f"{side}_hip", f"{side}_knee"]
    return None, list(METRIC_TEMPLATES[metric])


# MMPose labels needed by the supported templates. Left elbow/wrist are
# included even though the current visualizer only needed the right arm.
MMPose_LABEL_ALIASES = {
    "nose": ("nose",),
    "left_shoulder": ("left_shoulder",),
    "right_shoulder": ("right_shoulder",),
    "left_hip": ("left_hip",),
    "right_hip": ("right_hip",),
    "left_elbow": ("left_elbow",),
    "right_elbow": ("right_elbow",),
    "left_wrist": ("left_wrist",),
    "right_wrist": ("right_wrist",),
    "left_knee": ("left_knee",),
    "right_knee": ("right_knee",),
    "left_ankle": ("left_ankle",),
    "right_ankle": ("right_ankle",),
    "left_big_toe": ("left_big_toe", "left_heel"),
    "right_big_toe": ("right_big_toe", "right_heel"),
}


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    frame_rate: float
    frame_count: int


@dataclass(frozen=True)
class RGBDPointData:
    points: dict[str, np.ndarray]
    pixels: dict[str, np.ndarray]
    depths: dict[str, float]
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    frame_index: int
    depth_timestamp: float | None


class DepthZipReader:
    """Read iPad DepthFloat32 frames and their per-frame metadata."""

    def __init__(self, filename: str):
        self.filename = filename
        self.archive = zipfile.ZipFile(filename)
        self.entries = sorted(
            [name for name in self.archive.namelist() if not name.endswith("/")],
            key=lambda name: _entry_number(name),
        )
        if not self.entries:
            self.archive.close()
            raise ValueError(f"Depth archive has no frame files: {filename}")
        self.width: int | None = None
        self.height: int | None = None
        self.fx: float | None = None
        self.fy: float | None = None
        self.cx: float | None = None
        self.cy: float | None = None
        self.orientation: str | None = None
        self._metadata_cache: dict[int, dict[str, str]] = {}
        self._cached_frame: int | None = None
        self._cached_depth: np.ndarray | None = None
        self._read_metadata(0)

    def _read_metadata(self, frame: int) -> dict[str, str]:
        if frame in self._metadata_cache:
            return self._metadata_cache[frame]
        raw = self.archive.read(self.entries[frame])
        separator = re.search(br"\r?\n\r?\n", raw)
        if separator is None:
            raise ValueError(f"Depth frame has no header separator: {self.entries[frame]}")
        header = raw[:separator.end()].decode("utf-8", errors="replace")
        metadata = {}
        for line in header.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip().lower()] = value.strip()
        self._metadata_cache[frame] = metadata
        self.width = int(metadata["width"])
        self.height = int(metadata["height"])
        self.fx = float(metadata["fx"])
        self.fy = float(metadata["fy"])
        self.cx = float(metadata["ox"])
        self.cy = float(metadata["oy"])
        self.orientation = metadata.get("orientation", self.orientation)
        return metadata

    def read(self, frame: int) -> np.ndarray | None:
        if frame < 0 or frame >= len(self.entries):
            return None
        if frame == self._cached_frame:
            return self._cached_depth
        raw = self.archive.read(self.entries[frame])
        separator = re.search(br"\r?\n\r?\n", raw)
        if separator is None:
            raise ValueError(f"Depth frame has no header separator: {self.entries[frame]}")
        metadata = self._read_metadata(frame)
        width = int(metadata["width"])
        height = int(metadata["height"])
        expected_values = width * height
        expected_bytes = int(metadata.get("data_length", expected_values * 4))
        body = raw[separator.end():separator.end() + expected_bytes]
        if len(body) < expected_values * 4:
            raise ValueError(f"Depth frame is truncated: {self.entries[frame]}")
        depth = np.frombuffer(
            body[:expected_values * 4], dtype="<f4", count=expected_values
        ).reshape(height, width).copy()
        depth[(~np.isfinite(depth)) | (depth <= 0)] = np.nan
        self._cached_frame = frame
        self._cached_depth = depth
        return depth

    def timestamp(self, frame: int) -> float | None:
        if frame < 0 or frame >= len(self.entries):
            return None
        metadata = self._read_metadata(frame)
        value = metadata.get("timestamp") or metadata.get("ts_ms")
        if value is None:
            return None
        timestamp = float(value)
        if metadata.get("timestamp") is None:
            timestamp /= 1000.0
        return timestamp

    def close(self) -> None:
        self.archive.close()


def _entry_number(name: str) -> int:
    match = re.search(r"(\d+)", os.path.basename(name))
    return int(match.group(1)) if match else -1


def video_metadata(filename: str) -> VideoMetadata:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for video RGBD evaluation") from error
    capture = cv2.VideoCapture(filename)
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {filename}")
    metadata = VideoMetadata(
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_rate=float(capture.get(cv2.CAP_PROP_FPS)),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    capture.release()
    if metadata.width <= 0 or metadata.height <= 0 or metadata.frame_count <= 0:
        raise ValueError(f"Video metadata is incomplete: {filename}")
    return metadata


def read_video_frame(filename: str, frame: int) -> np.ndarray:
    """Return one RGB frame as an HxWx3 uint8 array."""
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for video RGBD evaluation") from error
    capture = cv2.VideoCapture(filename)
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Could not open video: {filename}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise IndexError(f"Could not read video frame {frame}: {filename}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_calibration(filename: str | None) -> dict:
    if not filename or not os.path.isfile(filename):
        return {}
    with open(filename, encoding="utf-8") as file:
        return json.load(file)


def _sample_depth(depth: np.ndarray, u: float, v: float, radius: int) -> float:
    column = int(round(u))
    row = int(round(v))
    first_column = max(0, column - radius)
    last_column = min(depth.shape[1], column + radius + 1)
    first_row = max(0, row - radius)
    last_row = min(depth.shape[0], row + radius + 1)
    if first_column >= last_column or first_row >= last_row:
        return np.nan
    patch = depth[first_row:last_row, first_column:last_column]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(valid)) if valid.size else np.nan


def lift_points_2d(
    points_2d: Mapping[str, Mapping[str, float] | tuple[float, float] | list[float]],
    depth_reader: DepthZipReader,
    frame: int,
    rgb_width: int,
    rgb_height: int,
    depth_sample_radius: int = 2,
) -> RGBDPointData:
    depth = depth_reader.read(frame)
    if depth is None:
        raise IndexError(f"Depth frame {frame} is unavailable")
    if None in (depth_reader.fx, depth_reader.fy, depth_reader.cx, depth_reader.cy):
        raise ValueError("Depth frame does not contain pinhole intrinsics")
    points: dict[str, np.ndarray] = {}
    pixels: dict[str, np.ndarray] = {}
    depths: dict[str, float] = {}
    for name, value in points_2d.items():
        if isinstance(value, Mapping):
            u = float(value["x"])
            v = float(value["y"])
        else:
            u, v = float(value[0]), float(value[1])
        depth_pixel = np.array([
            (u + 0.5) * depth_reader.width / rgb_width - 0.5,
            (v + 0.5) * depth_reader.height / rgb_height - 0.5,
        ])
        depth_value = _sample_depth(
            depth,
            depth_pixel[0],
            depth_pixel[1],
            max(0, int(depth_sample_radius)),
        )
        pixels[name] = depth_pixel
        depths[name] = depth_value
        if not np.isfinite(depth_value):
            points[name] = np.full(3, np.nan)
            continue
        x = (depth_pixel[0] - depth_reader.cx) * depth_value / depth_reader.fx
        y = (depth_pixel[1] - depth_reader.cy) * depth_value / depth_reader.fy
        points[name] = np.asarray((x, y, depth_value), dtype=float)
    return RGBDPointData(
        points=points,
        pixels=pixels,
        depths=depths,
        width=depth_reader.width,
        height=depth_reader.height,
        fx=depth_reader.fx,
        fy=depth_reader.fy,
        cx=depth_reader.cx,
        cy=depth_reader.cy,
        frame_index=frame,
        depth_timestamp=depth_reader.timestamp(frame),
    )


def torso_frame_from_points(
    points: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(name not in points for name in required):
        return None
    if any(not np.isfinite(points[name]).all() for name in required):
        return None
    shoulder_mid = (points["left_shoulder"] + points["right_shoulder"]) / 2.0
    hip_mid = (points["left_hip"] + points["right_hip"]) / 2.0
    inferior = hip_mid - shoulder_mid
    inferior = _unit(inferior)
    if inferior is None:
        return None
    right = points["right_shoulder"] - points["left_shoulder"]
    right = right - np.dot(right, inferior) * inferior
    right = _unit(right)
    if right is None:
        return None
    anterior = _unit(np.cross(inferior, right))
    if anterior is None:
        return None
    if "nose" in points and np.isfinite(points["nose"]).all():
        nose_direction = points["nose"] - shoulder_mid
        nose_direction -= np.dot(nose_direction, inferior) * inferior
        if np.linalg.norm(nose_direction) >= 1e-9:
            if np.dot(anterior, nose_direction) < 0:
                anterior = -anterior
    return {"inferior": inferior, "right": right, "anterior": anterior}


def _included_flexion_deg(prox, joint, distal):
    """Flexion angle (deg) at a hinge joint. 0 = fully extended/straight."""
    upper = prox - joint
    lower = distal - joint
    upper_length = float(np.linalg.norm(upper))
    lower_length = float(np.linalg.norm(lower))
    if min(upper_length, lower_length) < 1e-9:
        return np.nan
    included = np.degrees(
        np.arccos(np.clip(np.dot(upper, lower) / (upper_length * lower_length),
                          -1.0, 1.0))
    )
    return float(180.0 - included)


def _ankle_angle_deg(knee, ankle, toe, plantar):
    """Ankle dorsi/plantarflexion (deg) from the shank-foot included angle.

    Neutral standing: the shank points down and the foot points forward, so the
    included angle is ~90 deg. Dorsiflexion (toe up) lowers it, plantarflexion
    (toe down) raises it.
    """
    shank = np.asarray(ankle) - np.asarray(knee)
    foot = np.asarray(toe) - np.asarray(ankle)
    shank_length = float(np.linalg.norm(shank))
    foot_length = float(np.linalg.norm(foot))
    if min(shank_length, foot_length) < 1e-9:
        return np.nan
    included = np.degrees(
        np.arccos(np.clip(np.dot(shank, foot) / (shank_length * foot_length),
                          -1.0, 1.0))
    )
    if plantar:
        return float(included - 90.0)
    return float(90.0 - included)


def _limb_elevation_deg(vec, down):
    """Elevation (deg) of a limb vector from a reference 'down' axis.

    down should be the trunk's inferior direction at the NEUTRAL pose
    (side_hip - side_shoulder from the neutral / T1 frame), which keeps the
    reference stable and flat even when the person leans at the peak of the
    movement. The test label (flexion vs extension vs abduction) plus the peak
    frame the annotator picks determines the direction.
    """
    vec_hat = _unit(np.asarray(vec, dtype=float))
    down_hat = _unit(np.asarray(down, dtype=float))
    if vec_hat is None or down_hat is None:
        return np.nan
    angle = np.degrees(np.arccos(np.clip(np.dot(vec_hat, down_hat), -1.0, 1.0)))
    return float(angle)


def compute_metric(
    points_3d: Mapping[str, np.ndarray],
    metric: str,
    reference_down: np.ndarray | None = None,
) -> float:
    if metric not in METRIC_TEMPLATES:
        raise KeyError(f"Unsupported metric: {metric}")
    required = METRIC_TEMPLATES[metric]
    if any(name not in points_3d for name in required):
        return np.nan
    if any(not np.isfinite(points_3d[name]).all() for name in required):
        return np.nan
    side = metric.split("_", 1)[0]

    if "elbow" in metric:
        return _included_flexion_deg(
            points_3d[f"{side}_shoulder"],
            points_3d[f"{side}_elbow"],
            points_3d[f"{side}_wrist"],
        )
    if "knee" in metric:
        return _included_flexion_deg(
            points_3d[f"{side}_hip"],
            points_3d[f"{side}_knee"],
            points_3d[f"{side}_ankle"],
        )
    if "ankle" in metric:
        return _ankle_angle_deg(
            points_3d[f"{side}_knee"],
            points_3d[f"{side}_ankle"],
            points_3d[f"{side}_big_toe"],
            plantar=("plantar" in metric),
        )
    if "shoulder" in metric:
        vec = points_3d[f"{side}_elbow"] - points_3d[f"{side}_shoulder"]
    elif "hip" in metric:
        vec = points_3d[f"{side}_knee"] - points_3d[f"{side}_hip"]
    else:
        raise KeyError(f"Unsupported metric: {metric}")
    if reference_down is None:
        # Fallback for a single-frame measurement: use this frame's trunk axis.
        reference_down = points_3d[f"{side}_hip"] - points_3d[f"{side}_shoulder"]
    return _limb_elevation_deg(vec, reference_down)


def compute_metrics(points_3d: Mapping[str, np.ndarray], metrics=None) -> dict[str, float]:
    names = tuple(metrics) if metrics is not None else tuple(METRIC_TEMPLATES)
    return {metric: compute_metric(points_3d, metric) for metric in names}


def _unit(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-9:
        return None
    return vector / length


def _plane_angle(
    direction: np.ndarray,
    reference_axis: np.ndarray,
    movement_axis: np.ndarray,
) -> float:
    reference_component = float(np.dot(direction, reference_axis))
    movement_component = float(np.dot(direction, movement_axis))
    if np.hypot(reference_component, movement_component) < 1e-9:
        return np.nan
    return float(np.degrees(np.arctan2(movement_component, reference_component)))


def normalize_points(points) -> dict[str, dict[str, float]]:
    """Normalize JSON-like point values to {x, y} dictionaries."""
    normalized = {}
    for name, value in points.items():
        if isinstance(value, Mapping):
            if "x" not in value or "y" not in value:
                continue
            normalized[name] = {"x": float(value["x"]), "y": float(value["y"])}
        elif len(value) >= 2:
            normalized[name] = {"x": float(value[0]), "y": float(value[1])}
    return normalized


def mmpose_points_for_frame(
    labels: list[str],
    keypoints: np.ndarray,
    scores: np.ndarray,
    frame: int,
    score_threshold: float = 0.2,
) -> dict[str, dict[str, float]]:
    index = {label: position for position, label in enumerate(labels)}
    output = {}
    for name, aliases in MMPose_LABEL_ALIASES.items():
        keypoint_index = next((index.get(alias) for alias in aliases if alias in index), None)
        if keypoint_index is None:
            continue
        point = keypoints[frame, keypoint_index]
        score = scores[frame, keypoint_index]
        if np.isfinite(point).all() and score >= score_threshold:
            output[name] = {"x": float(point[0]), "y": float(point[1])}
    return output


def load_mmpose_json(filename: str, score_threshold: float = 0.2) -> tuple[list[str], np.ndarray, np.ndarray]:
    with open(filename, encoding="utf-8") as file:
        payload = json.load(file)
    metadata = payload.get("meta_info", {})
    id_to_name = metadata.get("keypoint_id2name", {})
    keypoint_count = int(metadata.get("num_keypoints", len(id_to_name)))
    labels = [
        str(id_to_name.get(str(i), id_to_name.get(i, f"keypoint_{i}"))).strip()
        for i in range(keypoint_count)
    ]
    frame_info = payload.get("instance_info", [])
    keypoints = np.full((len(frame_info), keypoint_count, 2), np.nan, dtype=float)
    scores = np.zeros((len(frame_info), keypoint_count), dtype=float)
    for frame_index, frame in enumerate(frame_info):
        instances = frame.get("instances", [])
        if not instances:
            continue
        instance = max(instances, key=lambda value: float(value.get("bbox_score", 0.0)))
        raw_keypoints = np.asarray(instance.get("keypoints", []), dtype=float)
        if raw_keypoints.ndim != 2 or raw_keypoints.shape[1] < 2:
            continue
        raw_scores = np.asarray(
            instance.get("keypoint_scores", np.ones(raw_keypoints.shape[0])),
            dtype=float,
        ).reshape(-1)
        count = min(keypoint_count, raw_keypoints.shape[0])
        keypoints[frame_index, :count] = raw_keypoints[:count, :2]
        scores[frame_index, :min(count, raw_scores.shape[0])] = raw_scores[:count]
    invalid = (~np.isfinite(keypoints).all(axis=2)) | (scores < score_threshold)
    keypoints[invalid] = np.nan
    return labels, keypoints, scores
