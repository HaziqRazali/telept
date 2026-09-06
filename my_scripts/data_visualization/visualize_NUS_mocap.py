#!/usr/bin/env python3
"""Interactive NUS C3D and MMPose viewer with independent timelines.

The top row shows the C3D skeleton and MMPose 2D body keypoints. The bottom
row shows separate metric plots for each source. C3D and MMPose frames are
intentionally independent because the recordings need not start together.

Select a metric to show it in both source plots. Red rings mark the joints used
by the selected metric in both top displays.

MMPose shoulder values are image-plane projections. A single 2D view cannot
independently recover anatomical flexion and adduction without camera
orientation or depth information.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import zipfile
from dataclasses import dataclass

import numpy as np

try:
    import ezc3d
except ImportError:
    raise SystemExit("ezc3d is not installed in this Python environment.")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.widgets import CheckButtons, RadioButtons, Slider


DEFAULT_FILENAME = (
    "/home/haziq/datasets/telept/data/NUS/val/"
    "haziq_upperlimb_right_24082026/mocap/"
    "haziq_upperlimb_right_24082026 dynamic 03.c3d"
)

POINT_ALIASES = {
    "T10": ("T10",),
    "C7": ("C7",),
    "CLAV": ("CLAV",),
    "RShoulder": ("RGH", "rightUpperArmO", "RSHO"),
    "RElbow": ("REJC", "RELB"),
    "RWrist": ("RWJC", "RWRA", "RWRB"),
    "RHand": ("rightHandO", "RFIN"),
}

CONNECTIONS = (
    ("T10", "C7"),
    ("C7", "CLAV"),
    ("CLAV", "RShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("RWrist", "RHand"),
)

RAW_MOCAP_CHAINS = (
    (
        "thorax",
        ("T10", "C7", "RBAK", "CLAV", "STRN"),
        "#f28e2b",
    ),
    (
        "right upper arm",
        ("RSHO", "RUPA", "RUPB", "RUPC", "RELB"),
        "#59a14f",
    ),
    (
        "right forearm/hand",
        ("RWRA", "RWRB", "RFRA", "RFIN"),
        "#59a14f",
    ),
)
RAW_MOCAP_LABELS = tuple(
    dict.fromkeys(
        name
        for _, chain, _ in RAW_MOCAP_CHAINS
        for name in chain
    )
)
RAW_MOCAP_CONNECTIONS = tuple(
    ("thorax", first, second)
    for position, first in enumerate(RAW_MOCAP_CHAINS[0][1])
    for second in RAW_MOCAP_CHAINS[0][1][position + 1:]
) + tuple(
    (chain_name, first, second)
    for chain_name, chain, _ in RAW_MOCAP_CHAINS[1:2]
    for first, second in zip(chain, chain[1:])
) + tuple(
    ("right forearm/hand", first, second)
    for position, first in enumerate(RAW_MOCAP_CHAINS[2][1])
    for second in RAW_MOCAP_CHAINS[2][1][position + 1:]
)

REFERENCE_POINT_ALIASES = {
    "T10": ("T10",),
    "C7": ("C7",),
    "STRN": ("STRN",),
    "CLAV": ("CLAV",),
    "RShoulder": ("RGH", "rightUpperArmO", "RSHO"),
}

MMPPOSE_BODY_CONNECTIONS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
)

MMPPOSE_METRIC_POINTS = {
    "nose": "nose",
    "right_shoulder": "right_shoulder",
    "right_elbow": "right_elbow",
    "right_wrist": "right_wrist",
    "left_shoulder": "left_shoulder",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
}

RGBD_POINT_LABELS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "right_elbow",
    "right_wrist",
)

METRICS = (
    ("Elbow flexion", "elbow_flexion", "#e76f51"),
    ("Shoulder flex/ext (T1->T2)", "shoulder_flexion_extension", "#457b9d"),
    ("Shoulder adduction (T1->T2)", "shoulder_adduction", "#2a9d8f"),
)

METRIC_HIGHLIGHTS_3D = {
    "elbow_flexion": ("RShoulder", "RElbow", "RWrist"),
    "shoulder_flexion_extension": ("RShoulder", "RElbow"),
    "shoulder_adduction": ("RShoulder", "RElbow"),
}

METRIC_HIGHLIGHTS_2D = {
    "elbow_flexion": ("right_shoulder", "right_elbow", "right_wrist"),
    "shoulder_flexion_extension": ("right_shoulder", "right_elbow"),
    "shoulder_adduction": ("right_shoulder", "right_elbow"),
}

FIXED_METRIC_Y_LIMITS = (-180.0, 180.0)


@dataclass
class C3DData:
    xyz: np.ndarray
    residual: np.ndarray
    labels: list[str]
    frame_rate: float
    point_units: str
    angle_units: str


@dataclass
class MMPoseData:
    keypoints: np.ndarray
    scores: np.ndarray
    labels: list[str]
    frame_rate: float
    image_width: float
    image_height: float


@dataclass
class RGBDPointData:
    points: np.ndarray
    pixels: np.ndarray
    depths: np.ndarray
    labels: tuple[str, ...]
    frame_rate: float
    fx: float
    fy: float
    cx: float
    cy: float


class RGBVideoReader:
    """Seekable RGB reader for the original video behind MMPose keypoints."""

    def __init__(self, filename: str):
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError("OpenCV is required to display the RGB video.") from error
        self._cv2 = cv2
        self.capture = cv2.VideoCapture(filename)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Could not open RGB video: {filename}")
        self.filename = filename

    def read(self, frame: int) -> np.ndarray | None:
        self.capture.set(self._cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, bgr = self.capture.read()
        if not ok:
            return None
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.capture.release()


class DepthZipReader:
    """Lazy reader for the iPad DepthFloat32 frames stored in a ZIP archive."""

    def __init__(self, filename: str):
        self.archive = zipfile.ZipFile(filename)
        self.filename = filename
        self.entries = sorted(
            [name for name in self.archive.namelist() if not name.endswith("/")],
            key=lambda name: int(re.search(r"(\d+)", os.path.basename(name)).group(1))
            if re.search(r"(\d+)", os.path.basename(name))
            else -1,
        )
        if not self.entries:
            self.archive.close()
            raise RuntimeError(f"Depth archive has no frame files: {filename}")
        self.width = None
        self.height = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.orientation = None
        self._cached_frame = None
        self._cached_depth = None
        self.read(0)

    def read(self, frame: int) -> np.ndarray | None:
        if frame < 0 or frame >= len(self.entries):
            return None
        if frame == self._cached_frame:
            return self._cached_depth

        raw = self.archive.read(self.entries[frame])
        separator = re.search(br"\r?\n\r?\n", raw)
        if separator is None:
            raise RuntimeError(f"Depth frame has no header separator: {self.entries[frame]}")
        header = raw[:separator.end()].decode("utf-8", errors="replace")
        metadata = {}
        for line in header.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip().lower()] = value.strip()

        width = int(metadata["width"])
        height = int(metadata["height"])
        expected_values = width * height
        expected_bytes = int(metadata.get("data_length", expected_values * 4))
        body = raw[separator.end():separator.end() + expected_bytes]
        if len(body) < expected_values * 4:
            raise RuntimeError(f"Depth frame is truncated: {self.entries[frame]}")
        depth = np.frombuffer(
            body[:expected_values * 4], dtype="<f4", count=expected_values
        ).reshape(height, width).copy()
        depth[(~np.isfinite(depth)) | (depth <= 0)] = np.nan
        self.width = width
        self.height = height
        self.fx = float(metadata["fx"]) if "fx" in metadata else self.fx
        self.fy = float(metadata["fy"]) if "fy" in metadata else self.fy
        self.cx = float(metadata["ox"]) if "ox" in metadata else self.cx
        self.cy = float(metadata["oy"]) if "oy" in metadata else self.cy
        self.orientation = metadata.get("orientation", self.orientation)
        self._cached_frame = frame
        self._cached_depth = depth
        return depth

    def close(self) -> None:
        self.archive.close()


def _clean_label(value) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return str(value).strip()


def _wrap_degrees(value):
    return (np.asarray(value) + 180.0) % 360.0 - 180.0


def _video_metadata(filename: str | None) -> tuple[float | None, float | None, float | None]:
    if not filename or not os.path.isfile(filename):
        return None, None, None
    try:
        import cv2

        capture = cv2.VideoCapture(filename)
        if not capture.isOpened():
            return None, None, None
        width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
        height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        frame_rate = float(capture.get(cv2.CAP_PROP_FPS)) or None
        capture.release()
        return width, height, frame_rate
    except ImportError:
        return None, None, None


def load_c3d(filename: str) -> C3DData:
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)
    c3d = ezc3d.c3d(filename)
    points = np.asarray(c3d["data"]["points"], dtype=float)
    if points.ndim != 3 or points.shape[0] < 4:
        raise ValueError(f"Unexpected C3D point array shape: {points.shape}")
    params = c3d["parameters"]["POINT"]
    labels = [_clean_label(label) for label in params["LABELS"]["value"]]
    labels = labels[: points.shape[1]]
    if len(labels) != points.shape[1]:
        raise ValueError("C3D point count and label count do not match")
    return C3DData(
        xyz=points[:3],
        residual=points[3],
        labels=labels,
        frame_rate=float(params["RATE"]["value"][0]),
        point_units=_clean_label(params.get("UNITS", {"value": ["unknown"]})["value"][0]),
        angle_units=_clean_label(params.get("ANGLE_UNITS", {"value": ["unknown"]})["value"][0]),
    )


def find_mmpose_json(c3d_filename: str, requested: str | None = None) -> str | None:
    if requested:
        candidate = requested[:-4] + ".json" if requested.lower().endswith(".mp4") else requested
        if os.path.isfile(candidate):
            return candidate
    c3d_stem = os.path.splitext(os.path.basename(c3d_filename))[0]
    match = re.search(r"\s+(dynamic\s+\S+)\s*$", c3d_stem, re.IGNORECASE)
    trial_stem = re.sub(r"\s+", "_", match.group(1)) if match else c3d_stem
    subject_dir = os.path.dirname(os.path.dirname(os.path.abspath(c3d_filename)))
    candidates = glob.glob(os.path.join(subject_dir, "mmpose", "*", "*", "*.json"))
    matches = [path for path in candidates if os.path.basename(path) == f"{trial_stem}.json"]
    return sorted(matches)[0] if matches else None


def find_paired_video(c3d_filename: str, mmpose_json: str | None = None) -> str | None:
    subject_dir = os.path.dirname(os.path.dirname(os.path.abspath(c3d_filename)))
    trial_stem = os.path.splitext(os.path.basename(mmpose_json or c3d_filename))[0]
    candidates = glob.glob(os.path.join(subject_dir, "videos", "*", f"{trial_stem}.*"))
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    candidates = [path for path in candidates if os.path.splitext(path)[1].lower() in video_exts]
    if candidates:
        return sorted(candidates)[0]
    if mmpose_json:
        sibling = os.path.splitext(mmpose_json)[0] + ".mp4"
        if os.path.isfile(sibling):
            return sibling
    return None


def find_paired_depth(c3d_filename: str, requested: str | None = None) -> str | None:
    """Find the depth archive belonging to the C3D subject/session."""
    if requested and os.path.isfile(requested):
        return requested
    subject_dir = os.path.dirname(os.path.dirname(os.path.abspath(c3d_filename)))
    candidates = glob.glob(os.path.join(subject_dir, "depth", "*", "depth.zip"))
    return sorted(candidates)[0] if candidates else None


def _sample_depth(depth: np.ndarray, u: float, v: float, radius: int) -> float:
    """Return the median valid depth around a floating-point pixel."""
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


def lift_mmpose_to_rgbd(
    mmpose: MMPoseData,
    mmpose_points: dict[str, int | None],
    depth_reader: DepthZipReader,
    depth_sample_radius: int = 2,
) -> RGBDPointData:
    """Lift selected MMPose pixels into sparse 3D depth-camera points.

    The current iPad recording stores RGB and depth in the same orientation,
    with depth at half the RGB resolution. Normalized-coordinate scaling keeps
    this correct if another session uses a different but aligned resolution.
    """
    if not all(
        value is not None
        for value in (depth_reader.fx, depth_reader.fy, depth_reader.cx, depth_reader.cy)
    ):
        raise RuntimeError("Depth frame does not contain pinhole intrinsics")
    if depth_reader.width is None or depth_reader.height is None:
        raise RuntimeError("Depth frame dimensions are unavailable")

    frame_count = min(len(mmpose.keypoints), len(depth_reader.entries))
    point_count = len(RGBD_POINT_LABELS)
    points = np.full((frame_count, point_count, 3), np.nan, dtype=float)
    pixels = np.full((frame_count, point_count, 2), np.nan, dtype=float)
    depths = np.full((frame_count, point_count), np.nan, dtype=float)
    depth_width = float(depth_reader.width)
    depth_height = float(depth_reader.height)

    for frame in range(frame_count):
        depth = depth_reader.read(frame)
        if depth is None:
            continue
        for point_number, name in enumerate(RGBD_POINT_LABELS):
            mmpose_index = mmpose_points.get(name)
            if not valid_2d_point(mmpose, mmpose_index, frame):
                continue
            rgb_pixel = mmpose.keypoints[frame, mmpose_index]
            depth_pixel = np.array([
                (rgb_pixel[0] + 0.5) * depth_width / mmpose.image_width - 0.5,
                (rgb_pixel[1] + 0.5) * depth_height / mmpose.image_height - 0.5,
            ])
            depth_value = _sample_depth(
                depth,
                depth_pixel[0],
                depth_pixel[1],
                max(0, int(depth_sample_radius)),
            )
            if not np.isfinite(depth_value):
                continue
            x = (depth_pixel[0] - depth_reader.cx) * depth_value / depth_reader.fx
            y = (depth_pixel[1] - depth_reader.cy) * depth_value / depth_reader.fy
            points[frame, point_number] = (x, y, depth_value)
            pixels[frame, point_number] = depth_pixel
            depths[frame, point_number] = depth_value

    return RGBDPointData(
        points=points,
        pixels=pixels,
        depths=depths,
        labels=RGBD_POINT_LABELS,
        frame_rate=mmpose.frame_rate,
        fx=depth_reader.fx,
        fy=depth_reader.fy,
        cx=depth_reader.cx,
        cy=depth_reader.cy,
    )


def static_rgbd_torso_frame(rgbd: RGBDPointData) -> dict[str, np.ndarray] | None:
    """Estimate a fixed 3D torso frame from lifted shoulders, hips, and nose."""
    index = {name: position for position, name in enumerate(rgbd.labels)}
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(name not in index for name in required):
        return None
    left_shoulder = rgbd.points[:, index["left_shoulder"]]
    right_shoulder = rgbd.points[:, index["right_shoulder"]]
    left_hip = rgbd.points[:, index["left_hip"]]
    right_hip = rgbd.points[:, index["right_hip"]]
    valid = (
        np.isfinite(left_shoulder).all(axis=1)
        & np.isfinite(right_shoulder).all(axis=1)
        & np.isfinite(left_hip).all(axis=1)
        & np.isfinite(right_hip).all(axis=1)
    )
    if not valid.any():
        return None

    shoulder_mid = (left_shoulder[valid] + right_shoulder[valid]) / 2.0
    hip_mid = (left_hip[valid] + right_hip[valid]) / 2.0
    inferior = np.median(hip_mid - shoulder_mid, axis=0)
    length = np.linalg.norm(inferior)
    if length < 1e-9:
        return None
    inferior = inferior / length

    right = np.median(right_shoulder[valid] - left_shoulder[valid], axis=0)
    right = right - np.dot(right, inferior) * inferior
    length = np.linalg.norm(right)
    if length < 1e-9:
        return None
    right = right / length

    anterior = np.cross(inferior, right)
    length = np.linalg.norm(anterior)
    if length < 1e-9:
        return None
    anterior = anterior / length

    nose_index = index.get("nose")
    if nose_index is not None:
        nose = rgbd.points[:, nose_index]
        nose_valid = valid & np.isfinite(nose).all(axis=1)
        if nose_valid.any():
            nose_shoulder_mid = (
                left_shoulder[nose_valid] + right_shoulder[nose_valid]
            ) / 2.0
            nose_direction = nose[nose_valid] - nose_shoulder_mid
            nose_direction -= (
                np.sum(nose_direction * inferior, axis=1)[:, None] * inferior
            )
            nose_direction = np.median(nose_direction, axis=0)
            if np.linalg.norm(nose_direction) > 1e-9:
                if np.dot(anterior, nose_direction) < 0:
                    anterior = -anterior

    return {"inferior": inferior, "right": right, "anterior": anterior}


def rgbd_upper_arm_direction(
    rgbd: RGBDPointData,
    frame: int,
    radius: int = 0,
) -> np.ndarray | None:
    """Return the normalized lifted shoulder-to-elbow direction."""
    index = {name: position for position, name in enumerate(rgbd.labels)}
    if "right_shoulder" not in index or "right_elbow" not in index:
        return None
    frames = np.arange(
        max(0, frame - radius),
        min(rgbd.points.shape[0], frame + radius + 1),
    )
    vectors = (
        rgbd.points[frames, index["right_elbow"]]
        - rgbd.points[frames, index["right_shoulder"]]
    )
    valid = np.isfinite(vectors).all(axis=1)
    if not valid.any():
        return None
    direction = np.median(vectors[valid], axis=0)
    length = np.linalg.norm(direction)
    return direction / length if np.isfinite(length) and length >= 1e-9 else None


def rgbd_elbow_flexion_angle(
    rgbd: RGBDPointData,
    frame: int,
    radius: int = 0,
) -> float:
    """Compute RGBD 3D elbow flexion from lifted shoulder/elbow/wrist points."""
    index = {name: position for position, name in enumerate(rgbd.labels)}
    required = ("right_shoulder", "right_elbow", "right_wrist")
    if any(name not in index for name in required):
        return np.nan

    frames = np.arange(
        max(0, frame - radius),
        min(rgbd.points.shape[0], frame + radius + 1),
    )
    shoulder = rgbd.points[frames, index["right_shoulder"]]
    elbow = rgbd.points[frames, index["right_elbow"]]
    wrist = rgbd.points[frames, index["right_wrist"]]
    valid = (
        np.isfinite(shoulder).all(axis=1)
        & np.isfinite(elbow).all(axis=1)
        & np.isfinite(wrist).all(axis=1)
    )
    if not valid.any():
        return np.nan

    upper_arm = np.median(shoulder[valid] - elbow[valid], axis=0)
    forearm = np.median(wrist[valid] - elbow[valid], axis=0)
    upper_arm_length = np.linalg.norm(upper_arm)
    forearm_length = np.linalg.norm(forearm)
    if min(upper_arm_length, forearm_length) < 1e-9:
        return np.nan
    included = np.degrees(
        np.arccos(
            np.clip(
                np.dot(upper_arm, forearm)
                / (upper_arm_length * forearm_length),
                -1.0,
                1.0,
            )
        )
    )
    return float(180.0 - included)


def rgbd_elbow_flexion_angles(
    rgbd: RGBDPointData,
    frames: list[int],
    window_radius: int = 0,
) -> np.ndarray:
    """Compute RGBD elbow flexion for a sequence of frame indices."""
    return np.asarray(
        [rgbd_elbow_flexion_angle(rgbd, frame, window_radius) for frame in frames],
        dtype=float,
    )


def rgbd_shoulder_flexion_angles(
    rgbd: RGBDPointData,
    torso: dict[str, np.ndarray] | None,
    frames: list[int],
    window_radius: int = 0,
) -> np.ndarray:
    """Return absolute RGBD 3D shoulder flexion/extension angles."""
    values = []
    for frame in frames:
        direction = rgbd_upper_arm_direction(rgbd, frame, window_radius)
        if torso is None or direction is None or "anterior" not in torso:
            values.append(np.nan)
            continue
        values.append(
            _plane_angle(direction, torso["inferior"], torso["anterior"])
        )
    return np.asarray(values, dtype=float)


def shoulder_flexion_rgbd_t1_t2(
    rgbd_angles: np.ndarray,
    t1: int,
    t2: int,
    frame_start: int,
) -> float:
    """Return RGBD shoulder flexion/extension change from T1 to T2."""
    first = rgbd_angles[t1 - frame_start]
    second = rgbd_angles[t2 - frame_start]
    if not np.isfinite([first, second]).all():
        return np.nan
    return float(_wrap_degrees(second - first))


def rgbd_shoulder_adduction_angles(
    rgbd: RGBDPointData,
    torso: dict[str, np.ndarray] | None,
    frames: list[int],
    window_radius: int = 0,
) -> np.ndarray:
    """Return absolute adduction angles (positive toward the midline)."""
    values = []
    for frame in frames:
        direction = rgbd_upper_arm_direction(rgbd, frame, window_radius)
        if torso is None:
            values.append(np.nan)
            continue
        abduction = _plane_angle(direction, torso["inferior"], torso["right"])
        values.append(-abduction if np.isfinite(abduction) else np.nan)
    return np.asarray(values, dtype=float)


def shoulder_adduction_rgbd_t1_t2(
    rgbd_angles: np.ndarray,
    t1: int,
    t2: int,
    frame_start: int,
) -> float:
    """Return RGBD adduction change from two MMPose/RGBD frame indices."""
    first = rgbd_angles[t1 - frame_start]
    second = rgbd_angles[t2 - frame_start]
    if not np.isfinite([first, second]).all():
        return np.nan
    return float(_wrap_degrees(second - first))


def load_mmpose_json(
    filename: str,
    frame_rate: float = 30.0,
    score_threshold: float = 0.2,
    video_filename: str | None = None,
) -> MMPoseData:
    with open(filename) as file:
        payload = json.load(file)
    metadata = payload.get("meta_info", {})
    id_to_name = metadata.get("keypoint_id2name", {})
    keypoint_count = int(metadata.get("num_keypoints", len(id_to_name)))
    labels = [
        _clean_label(id_to_name.get(str(i), id_to_name.get(i, f"keypoint_{i}")))
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
        score_count = min(count, raw_scores.shape[0])
        keypoints[frame_index, :count] = raw_keypoints[:count, :2]
        scores[frame_index, :score_count] = raw_scores[:score_count]
        if score_count < count:
            scores[frame_index, score_count:count] = 1.0
    invalid = (~np.isfinite(keypoints).all(axis=2)) | (scores < score_threshold)
    keypoints[invalid] = np.nan
    width, height, _ = _video_metadata(video_filename)
    if width is None:
        values = keypoints[:, :, 0][np.isfinite(keypoints[:, :, 0])]
        width = float(np.nanmax(values)) if values.size else 1.0
    if height is None:
        values = keypoints[:, :, 1][np.isfinite(keypoints[:, :, 1])]
        height = float(np.nanmax(values)) if values.size else 1.0
    return MMPoseData(keypoints, scores, labels, frame_rate, width, height)


def valid_point(data: C3DData, index: int | None, frame: int) -> bool:
    return bool(
        index is not None
        and 0 <= frame < data.xyz.shape[2]
        and data.residual[index, frame] >= 0
        and np.isfinite(data.xyz[:, index, frame]).all()
    )


def resolve_alias(data: C3DData, aliases: tuple[str, ...]) -> int | None:
    index = {label: i for i, label in enumerate(data.labels)}
    for alias in aliases:
        if alias in index:
            return index[alias]
    return None


def resolve_skeleton(data: C3DData) -> dict[str, int | None]:
    return {name: resolve_alias(data, aliases) for name, aliases in POINT_ALIASES.items()}


def resolve_raw_markers(data: C3DData) -> dict[str, int | None]:
    indices = {label: index for index, label in enumerate(data.labels)}
    return {name: indices.get(name) for name in RAW_MOCAP_LABELS}


def axis_bounds(data: C3DData, skeleton: dict[str, int | None]) -> tuple[np.ndarray, float]:
    indices = [index for index in skeleton.values() if index is not None]
    if not indices:
        return np.zeros(3), 1.0
    values = data.xyz[:, indices, :]
    valid = np.isfinite(values).all(axis=0) & (data.residual[indices, :] >= 0)
    if not valid.any():
        return np.zeros(3), 1.0
    points = values.transpose(1, 2, 0)[valid]
    center = np.nanmedian(points, axis=0)
    extent = float(np.nanmax(np.abs(points - center)))
    return center, max(extent * 1.15, 1.0)


def projection_bounds(
    data: C3DData,
    skeleton: dict[str, int | None],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return readable per-axis limits around the displayed C3D points."""
    indices = [index for index in skeleton.values() if index is not None]
    if not indices:
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)
    values = data.xyz[:, indices, :]
    valid = np.isfinite(values).all(axis=0) & (data.residual[indices, :] >= 0)
    points = values.transpose(1, 2, 0)[valid]
    if not len(points):
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)
    limits = []
    for component in range(3):
        low, high = np.nanpercentile(points[:, component], [1.0, 99.0])
        padding = max((high - low) * 0.15, 1.0)
        limits.append((float(low - padding), float(high + padding)))
    return tuple(limits)


def robust_position(data: C3DData, index: int | None, frame: int | None = None, radius: int = 0) -> np.ndarray | None:
    if index is None:
        return None
    if frame is None:
        frames = np.arange(data.xyz.shape[2])
    else:
        frames = np.arange(max        http://127.0.0.1:8091/(0, frame - radius), min(data.xyz.shape[2], frame + radius + 1))
    values = data.xyz[:, index, frames].T
    valid = (data.residual[index, frames] >= 0) & np.isfinite(values).all(axis=1)
    return np.median(values[valid], axis=0) if valid.any() else None


def upper_arm_direction(data: C3DData, skeleton: dict[str, int | None], frame: int, radius: int = 0) -> np.ndarray | None:
    shoulder_index, elbow_index = skeleton.get("RShoulder"), skeleton.get("RElbow")
    if shoulder_index is None or elbow_index is None:
        return None
    frames = np.arange(max(0, frame - radius), min(data.xyz.shape[2], frame + radius + 1))
    shoulder = data.xyz[:, shoulder_index, frames].T
    elbow = data.xyz[:, elbow_index, frames].T
    valid = (
        (data.residual[shoulder_index, frames] >= 0)
        & (data.residual[elbow_index, frames] >= 0)
        & np.isfinite(shoulder).all(axis=1)
        & np.isfinite(elbow).all(axis=1)
    )
    if not valid.any():
        return None
    direction = np.median(elbow[valid] - shoulder[valid], axis=0)
    length = np.linalg.norm(direction)
    return direction / length if np.isfinite(length) and length >= 1e-9 else None


def _thorax_frame_from_frames(
    data: C3DData,
    skeleton: dict[str, int | None],
    frames: np.ndarray,
) -> dict[str, np.ndarray] | None:
    indices = {name: resolve_alias(data, aliases) for name, aliases in REFERENCE_POINT_ALIASES.items()}
    if skeleton.get("RShoulder") is not None:
        indices["RShoulder"] = skeleton["RShoulder"]
    required = ("T10", "C7", "STRN", "CLAV", "RShoulder")
    if any(indices[name] is None for name in required):
        return None
    frame_values = {
        name: data.xyz[:, index, frames].T for name, index in indices.items()
    }
    valid = np.ones(len(frames), dtype=bool)
    for name in required:
        index = indices[name]
        valid &= data.residual[index, frames] >= 0
        valid &= np.isfinite(frame_values[name]).all(axis=1)
    if not valid.any():
        return None

    superior_vectors = frame_values["C7"] - frame_values["T10"]
    superior_lengths = np.linalg.norm(superior_vectors, axis=1)
    valid &= superior_lengths >= 1e-9
    if not valid.any():
        return None

    superior = superior_vectors[valid] / superior_lengths[valid, None]
    anterior_vectors = frame_values["STRN"][valid] - frame_values["T10"][valid]
    anterior_vectors -= (
        np.sum(anterior_vectors * superior, axis=1)[:, None] * superior
    )
    anterior_lengths = np.linalg.norm(anterior_vectors, axis=1)
    valid_anterior = anterior_lengths >= 1e-9
    if not valid_anterior.any():
        return None

    superior = superior[valid_anterior]
    anterior = anterior_vectors[valid_anterior] / anterior_lengths[valid_anterior, None]
    shoulder_to_clavicle = (
        frame_values["RShoulder"][valid][valid_anterior]
        - frame_values["CLAV"][valid][valid_anterior]
    )
    right = np.cross(superior, anterior)
    right_lengths = np.linalg.norm(right, axis=1)
    valid_right = right_lengths >= 1e-9
    if not valid_right.any():
        return None

    superior = superior[valid_right]
    anterior = anterior[valid_right]
    right = right[valid_right] / right_lengths[valid_right, None]
    shoulder_to_clavicle = shoulder_to_clavicle[valid_right]
    right[np.sum(right * shoulder_to_clavicle, axis=1) < 0] *= -1

    inferior = -np.median(superior, axis=0)
    inferior /= np.linalg.norm(inferior)
    anterior = np.median(anterior, axis=0)
    anterior -= np.dot(anterior, inferior) * inferior
    length = np.linalg.norm(anterior)
    if length < 1e-9:
        return None
    anterior /= length
    right = np.cross(-inferior, anterior)
    length = np.linalg.norm(right)
    if length < 1e-9:
        return None
    right /= length
    if np.dot(right, np.median(shoulder_to_clavicle, axis=0)) < 0:
        right = -right
    return {"inferior": inferior, "anterior": anterior, "right": right}


def thorax_frame_at(
    data: C3DData,
    skeleton: dict[str, int | None],
    frame: int,
    radius: int = 0,
) -> dict[str, np.ndarray] | None:
    """Estimate the thorax basis from corresponding C3D marker frames."""
    frames = np.arange(
        max(0, frame - radius),
        min(data.xyz.shape[2], frame + radius + 1),
    )
    return _thorax_frame_from_frames(data, skeleton, frames)


def static_thorax_frame(data: C3DData, skeleton: dict[str, int | None]) -> dict[str, np.ndarray] | None:
    """Estimate one robust thorax basis from same-frame marker vectors."""
    return _thorax_frame_from_frames(
        data,
        skeleton,
        np.arange(data.xyz.shape[2]),
    )


def _plane_angle(direction: np.ndarray | None, reference_axis: np.ndarray, movement_axis: np.ndarray) -> float:
    if direction is None:
        return np.nan
    reference_component = float(np.dot(direction, reference_axis))
    movement_component = float(np.dot(direction, movement_axis))
    if np.hypot(reference_component, movement_component) < 1e-9:
        return np.nan
    return float(np.degrees(np.arctan2(movement_component, reference_component)))


def elbow_flexion_angle(data: C3DData, skeleton: dict[str, int | None], frame: int) -> float:
    indices = [skeleton.get("RShoulder"), skeleton.get("RElbow"), skeleton.get("RWrist")]
    if any(index is None for index in indices) or not all(valid_point(data, index, frame) for index in indices):
        return np.nan
    shoulder, elbow, wrist = [data.xyz[:, index, frame] for index in indices]
    upper, forearm = shoulder - elbow, wrist - elbow
    lengths = np.linalg.norm(upper), np.linalg.norm(forearm)
    if min(lengths) < 1e-9:
        return np.nan
    included = np.degrees(np.arccos(np.clip(np.dot(upper, forearm) / (lengths[0] * lengths[1]), -1.0, 1.0)))
    return float(180.0 - included)


def shoulder_flexion_extension_t1_t2(data: C3DData, skeleton: dict[str, int | None], t1: int, t2: int, thorax=None, window_radius: int = 5) -> float:
    first_thorax = thorax or thorax_frame_at(data, skeleton, t1, window_radius)
    second_thorax = thorax or thorax_frame_at(data, skeleton, t2, window_radius)
    if first_thorax is None or second_thorax is None:
        return np.nan
    first = _plane_angle(
        upper_arm_direction(data, skeleton, t1, window_radius),
        first_thorax["inferior"],
        first_thorax["anterior"],
    )
    second = _plane_angle(
        upper_arm_direction(data, skeleton, t2, window_radius),
        second_thorax["inferior"],
        second_thorax["anterior"],
    )
    return float(_wrap_degrees(second - first)) if np.isfinite([first, second]).all() else np.nan


def shoulder_adduction_t1_t2(data: C3DData, skeleton: dict[str, int | None], t1: int, t2: int, thorax=None, window_radius: int = 5) -> float:
    first_thorax = thorax or thorax_frame_at(data, skeleton, t1, window_radius)
    second_thorax = thorax or thorax_frame_at(data, skeleton, t2, window_radius)
    if first_thorax is None or second_thorax is None:
        return np.nan
    first = _plane_angle(
        upper_arm_direction(data, skeleton, t1, window_radius),
        first_thorax["inferior"],
        first_thorax["right"],
    )
    second = _plane_angle(
        upper_arm_direction(data, skeleton, t2, window_radius),
        second_thorax["inferior"],
        second_thorax["right"],
    )
    return float(_wrap_degrees(first - second)) if np.isfinite([first, second]).all() else np.nan


def resolve_mmpose_points(data: MMPoseData) -> dict[str, int | None]:
    index = {label: i for i, label in enumerate(data.labels)}
    return {name: index.get(label) for name, label in MMPPOSE_METRIC_POINTS.items()}


def valid_2d_point(data: MMPoseData, index: int | None, frame: int) -> bool:
    return bool(index is not None and 0 <= frame < len(data.keypoints) and np.isfinite(data.keypoints[frame, index]).all() and data.scores[frame, index] > 0)


def upper_arm_direction_2d(data: MMPoseData, points: dict[str, int | None], frame: int, radius: int = 0) -> np.ndarray | None:
    shoulder_index, elbow_index = points.get("right_shoulder"), points.get("right_elbow")
    if shoulder_index is None or elbow_index is None:
        return None
    frames = np.arange(max(0, frame - radius), min(len(data.keypoints), frame + radius + 1))
    shoulder, elbow = data.keypoints[frames, shoulder_index], data.keypoints[frames, elbow_index]
    valid = np.isfinite(shoulder).all(axis=1) & np.isfinite(elbow).all(axis=1)
    if not valid.any():
        return None
    direction = np.median(elbow[valid] - shoulder[valid], axis=0)
    length = np.linalg.norm(direction)
    return direction / length if np.isfinite(length) and length >= 1e-9 else None


def static_2d_torso_frame(data: MMPoseData, points: dict[str, int | None]) -> dict[str, np.ndarray] | None:
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(points.get(name) is None for name in required):
        return None
    left_shoulder = data.keypoints[:, points["left_shoulder"]]
    right_shoulder = data.keypoints[:, points["right_shoulder"]]
    left_hip = data.keypoints[:, points["left_hip"]]
    right_hip = data.keypoints[:, points["right_hip"]]
    valid = (
        np.isfinite(left_shoulder).all(axis=1)
        & np.isfinite(right_shoulder).all(axis=1)
        & np.isfinite(left_hip).all(axis=1)
        & np.isfinite(right_hip).all(axis=1)
    )
    if not valid.any():
        return None
    shoulder_mid = (left_shoulder[valid] + right_shoulder[valid]) / 2.0
    hip_mid = (left_hip[valid] + right_hip[valid]) / 2.0
    down = np.median(hip_mid - shoulder_mid, axis=0)
    length = np.linalg.norm(down)
    if length < 1e-9:
        return None
    down /= length
    right = np.median(right_shoulder[valid] - left_shoulder[valid], axis=0)
    right -= np.dot(right, down) * down
    length = np.linalg.norm(right)
    if length < 1e-9:
        return None
    right /= length
    return {"down": down, "right": right, "forward": np.array([down[1], -down[0]])}


def elbow_flexion_angle_2d(data: MMPoseData, points: dict[str, int | None], frame: int) -> float:
    indices = [points.get("right_shoulder"), points.get("right_elbow"), points.get("right_wrist")]
    if any(index is None for index in indices) or not all(valid_2d_point(data, index, frame) for index in indices):
        return np.nan
    shoulder, elbow, wrist = [data.keypoints[frame, index] for index in indices]
    upper, forearm = shoulder - elbow, wrist - elbow
    lengths = np.linalg.norm(upper), np.linalg.norm(forearm)
    if min(lengths) < 1e-9:
        return np.nan
    included = np.degrees(np.arccos(np.clip(np.dot(upper, forearm) / (lengths[0] * lengths[1]), -1.0, 1.0)))
    return float(180.0 - included)


def shoulder_angles_2d(data: MMPoseData, points: dict[str, int | None], frame: int, torso, radius: int = 0) -> tuple[float, float]:
    direction = upper_arm_direction_2d(data, points, frame, radius)
    if torso is None or direction is None:
        return np.nan, np.nan
    return _plane_angle(direction, torso["down"], torso["forward"]), _plane_angle(direction, torso["down"], torso["right"])


def shoulder_flexion_extension_2d_t1_t2(data: MMPoseData, points: dict[str, int | None], t1: int, t2: int, torso=None, window_radius: int = 1) -> float:
    torso = torso or static_2d_torso_frame(data, points)
    first, second = shoulder_angles_2d(data, points, t1, torso, window_radius)[0], shoulder_angles_2d(data, points, t2, torso, window_radius)[0]
    return float(_wrap_degrees(second - first)) if np.isfinite([first, second]).all() else np.nan


def shoulder_adduction_2d_t1_t2(data: MMPoseData, points: dict[str, int | None], t1: int, t2: int, torso=None, window_radius: int = 1) -> float:
    torso = torso or static_2d_torso_frame(data, points)
    first, second = shoulder_angles_2d(data, points, t1, torso, window_radius)[1], shoulder_angles_2d(data, points, t2, torso, window_radius)[1]
    return float(_wrap_degrees(first - second)) if np.isfinite([first, second]).all() else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize NUS C3D and MMPose with independent timelines.")
    parser.add_argument("--filename", nargs="+", default=[DEFAULT_FILENAME])
    parser.add_argument("--mmpose-json", default=None, help="MMPose JSON; auto-discovered when omitted.")
    parser.add_argument("--rgb-video", default=None, help="Original RGB video for the MMPose panel; auto-discovered when omitted.")
    parser.add_argument("--depth-zip", default=None, help="Depth ZIP for the MMPose panel; auto-discovered when omitted.")
    parser.add_argument("--depth-min", type=float, default=0.2, help="Minimum depth shown by the overlay, in metres.")
    parser.add_argument("--depth-max", type=float, default=6.0, help="Maximum depth shown by the overlay, in metres.")
    parser.add_argument("--depth-alpha", type=float, default=0.45, help="Depth overlay opacity (default: 0.45).")
    parser.add_argument("--depth-sample-radius", type=int, default=2, help="Radius of the valid-depth patch sampled around each 2D keypoint (default: 2 pixels).")
    parser.add_argument("--mmpose-fps", type=float, default=None, help="MMPose input FPS; inferred from original video or 30.")
    parser.add_argument("--mmpose-score-threshold", type=float, default=0.2)
    parser.add_argument("--c3d-start", type=int, default=0)
    parser.add_argument("--c3d-end", type=int, default=None)
    parser.add_argument("--mmpose-start", type=int, default=0)
    parser.add_argument("--mmpose-end", type=int, default=None)
    parser.add_argument("--window-ms", type=float, default=100.0)
    parser.add_argument("--auto-y", action="store_true")
    args = parser.parse_args()
    filename = " ".join(args.filename)
    try:
        c3d = load_c3d(filename)
    except (FileNotFoundError, ValueError, KeyError) as error:
        raise SystemExit(f"[error] Could not load C3D: {error}") from error

    mmpose_json = find_mmpose_json(filename, args.mmpose_json)
    mmpose = None
    mmpose_points = None
    mmpose_torso = None
    paired_video = None
    rgb_video = None
    rgb_reader = None
    depth_zip = None
    depth_reader = None
    rgbd = None
    rgbd_torso = None
    if mmpose_json:
        paired_video = find_paired_video(filename, mmpose_json)
        rgb_video = args.rgb_video or paired_video
        depth_zip = find_paired_depth(filename, args.depth_zip)
        _, _, paired_fps = _video_metadata(paired_video)
        try:
            mmpose = load_mmpose_json(
                mmpose_json,
                frame_rate=args.mmpose_fps or paired_fps or 30.0,
                score_threshold=args.mmpose_score_threshold,
                video_filename=rgb_video,
            )
            mmpose_points = resolve_mmpose_points(mmpose)
            mmpose_torso = static_2d_torso_frame(mmpose, mmpose_points)
            if rgb_video:
                try:
                    rgb_reader = RGBVideoReader(rgb_video)
                except RuntimeError as error:
                    print(f"[warning] Could not open RGB video: {error}")
            if depth_zip:
                try:
                    depth_reader = DepthZipReader(depth_zip)
                except (OSError, RuntimeError, KeyError, ValueError) as error:
                    print(f"[warning] Could not open depth archive: {error}")
            if depth_reader is not None:
                try:
                    print("  lifting sparse RGBD points from MMPose + depth...")
                    rgbd = lift_mmpose_to_rgbd(
                        mmpose,
                        mmpose_points,
                        depth_reader,
                        depth_sample_radius=args.depth_sample_radius,
                    )
                    rgbd_torso = static_rgbd_torso_frame(rgbd)
                    if rgbd_torso is None:
                        print("[warning] Could not construct the RGBD torso frame")
                except (RuntimeError, ValueError, KeyError, TypeError) as error:
                    print(f"[warning] Could not lift MMPose points with depth: {error}")
                    rgbd = None
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"[warning] Could not load MMPose JSON: {error}")
            mmpose = None
    else:
        print("[warning] MMPose JSON not found; showing C3D only")

    c3d_skeleton = resolve_skeleton(c3d)
    raw_marker_indices = resolve_raw_markers(c3d)
    missing_raw_markers = [
        name for name, index in raw_marker_indices.items() if index is None
    ]
    if missing_raw_markers:
        print(
            "[warning] raw mocap markers not found: "
            + ", ".join(missing_raw_markers)
        )
    display_skeleton = {
        **c3d_skeleton,
        **{
            f"raw_{name}": index
            for name, index in raw_marker_indices.items()
            if index is not None
        },
    }
    c3d_count = c3d.xyz.shape[2]
    mmpose_count = len(mmpose.keypoints) if mmpose is not None else 0
    c3d_start = max(0, min(args.c3d_start, c3d_count - 1))
    c3d_end = c3d_count - 1 if args.c3d_end is None else max(c3d_start, min(args.c3d_end, c3d_count - 1))
    if mmpose is not None:
        mmpose_start = max(0, min(args.mmpose_start, mmpose_count - 1))
        mmpose_end = mmpose_count - 1 if args.mmpose_end is None else max(mmpose_start, min(args.mmpose_end, mmpose_count - 1))
    else:
        mmpose_start = mmpose_end = 0

    print(f"Loading: {filename}")
    print(f"  C3D points : {len(c3d.labels)}")
    print(f"  C3D frames : {c3d_count} @ {c3d.frame_rate:.3f} Hz")
    if mmpose is not None:
        print(f"  MMPose JSON: {mmpose_json}")
        print(f"  RGB video: {rgb_video or 'not found'}")
        print(f"  Depth ZIP: {depth_zip or 'not found'}")
        if depth_reader is not None:
            print(f"  Depth frames: {len(depth_reader.entries)} @ {depth_reader.width}x{depth_reader.height}")
        print(f"  MMPose frames: {mmpose_count} @ {mmpose.frame_rate:.3f} Hz")
        print("  timelines: independent; no C3D/MMPose frame mapping is applied")
        print(f"  2D metric points: {mmpose_points}")

    c3d_window_radius = max(0, int(round(c3d.frame_rate * max(0.0, args.window_ms) / 1000.0 / 2.0)))
    mmpose_window_radius = 0 if mmpose is None else max(0, int(round(mmpose.frame_rate * max(0.0, args.window_ms) / 1000.0 / 2.0)))
    c3d_frames = list(range(c3d_start, c3d_end + 1))
    c3d_dirs = [upper_arm_direction(c3d, c3d_skeleton, frame, c3d_window_radius) for frame in c3d_frames]
    c3d_thorax_frames = [
        thorax_frame_at(c3d, c3d_skeleton, frame, c3d_window_radius)
        for frame in c3d_frames
    ]
    c3d_elbow = np.array([elbow_flexion_angle(c3d, c3d_skeleton, frame) for frame in c3d_frames])
    c3d_flexion = np.array([
        _plane_angle(direction, thorax["inferior"], thorax["anterior"])
        if direction is not None and thorax is not None
        else np.nan
        for direction, thorax in zip(c3d_dirs, c3d_thorax_frames)
    ])
    c3d_abduction = np.array([
        _plane_angle(direction, thorax["inferior"], thorax["right"])
        if direction is not None and thorax is not None
        else np.nan
        for direction, thorax in zip(c3d_dirs, c3d_thorax_frames)
    ])

    if mmpose is not None:
        mmpose_frames = list(range(mmpose_start, mmpose_end + 1))
        mmpose_dirs = [upper_arm_direction_2d(mmpose, mmpose_points, frame, mmpose_window_radius) for frame in mmpose_frames]
        mmpose_elbow = np.array([elbow_flexion_angle_2d(mmpose, mmpose_points, frame) for frame in mmpose_frames])
        mmpose_flexion = np.array([shoulder_angles_2d(mmpose, mmpose_points, frame, mmpose_torso, mmpose_window_radius)[0] for frame in mmpose_frames])
        mmpose_abduction = np.array([shoulder_angles_2d(mmpose, mmpose_points, frame, mmpose_torso, mmpose_window_radius)[1] for frame in mmpose_frames])
        if rgbd is not None:
            rgbd_elbow = rgbd_elbow_flexion_angles(
                rgbd,
                mmpose_frames,
                mmpose_window_radius,
            )
            rgbd_adduction = rgbd_shoulder_adduction_angles(
                rgbd,
                rgbd_torso,
                mmpose_frames,
                mmpose_window_radius,
            )
            rgbd_flexion = rgbd_shoulder_flexion_angles(
                rgbd,
                rgbd_torso,
                mmpose_frames,
                mmpose_window_radius,
            )
        else:
            rgbd_flexion = np.full(len(mmpose_frames), np.nan)
            rgbd_adduction = np.full(len(mmpose_frames), np.nan)
    else:
        mmpose_frames = []
        mmpose_dirs = []
        mmpose_elbow = mmpose_flexion = mmpose_abduction = np.array([])
        rgbd_elbow = np.array([])
        rgbd_flexion = np.array([])
        rgbd_adduction = np.array([])

    c3d_valid = [
        frame
        for frame, direction, thorax in zip(
            c3d_frames, c3d_dirs, c3d_thorax_frames
        )
        if direction is not None and thorax is not None
    ]
    mmpose_valid = [frame for frame, direction in zip(mmpose_frames, mmpose_dirs) if direction is not None]
    if rgbd is not None:
        rgbd_valid = {
            frame
            for frame, elbow, flexion, adduction in zip(
                mmpose_frames, rgbd_elbow, rgbd_flexion, rgbd_adduction
            )
            if np.isfinite([elbow, flexion, adduction]).all()
        }
        mmpose_valid = [frame for frame in mmpose_valid if frame in rgbd_valid]
    c3d_t1_default = c3d_valid[0] if c3d_valid else c3d_start
    c3d_t2_default = c3d_valid[-1] if c3d_valid else c3d_end
    complete_c3d_frames = [
        frame
        for frame in c3d_frames
        if all(
            valid_point(c3d, index, frame)
            for index in c3d_skeleton.values()
            if index is not None
        )
    ]
    c3d_frame_default = complete_c3d_frames[0] if complete_c3d_frames else c3d_t1_default
    mmpose_t1_default = mmpose_valid[0] if mmpose_valid else mmpose_start
    mmpose_t2_default = mmpose_valid[-1] if mmpose_valid else mmpose_end

    center, extent = axis_bounds(c3d, display_skeleton)
    readable_limits = projection_bounds(c3d, display_skeleton)
    fig = plt.figure(figsize=(15, 11))
    try:
        fig.canvas.manager.set_window_title("NUS C3D + MMPose Viewer")
    except AttributeError:
        pass
    grid = GridSpec(2, 2, height_ratios=(2.1, 1.0), hspace=0.28, wspace=0.15, left=0.05, right=0.78, top=0.94, bottom=0.32)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax2d = fig.add_subplot(grid[0, 1])
    ax_c3d_metric = fig.add_subplot(grid[1, 0])
    ax_mmpose_metric = fig.add_subplot(grid[1, 1])

    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlim(center[0] - extent, center[0] + extent)
    ax3d.set_ylim(center[1] - extent, center[1] + extent)
    ax3d.set_zlim(center[2] - extent, center[2] + extent)
    ax3d.set_xlabel(f"X ({c3d.point_units})")
    ax3d.set_ylabel(f"Y ({c3d.point_units})")
    ax3d.set_zlabel(f"Z ({c3d.point_units})")
    ax3d.set_title("C3D 3D")
    original_c3d_limits = (
        ax3d.get_xlim3d(),
        ax3d.get_ylim3d(),
        ax3d.get_zlim3d(),
    )
    point_lines = {name: ax3d.plot([], [], [], "o", ms=5, color="#1d3557")[0] for name in POINT_ALIASES}
    connection_lines = {(first, second): ax3d.plot([], [], [], "-", lw=2.5, color="#e63946")[0] for first, second in CONNECTIONS}
    labels_text = {name: ax3d.text(0, 0, 0, name, fontsize=7, visible=False) for name in POINT_ALIASES}
    raw_point_lines = {}
    raw_chain_lines = {}
    raw_labels_text = {}
    for chain_name, chain, color in RAW_MOCAP_CHAINS:
        for name in chain:
            if name not in raw_point_lines:
                raw_point_lines[name] = ax3d.plot(
                    [], [], [], "o", ms=4.5, color=color,
                    markeredgecolor="white", markeredgewidth=0.5,
                )[0]
                raw_labels_text[name] = ax3d.text(
                    0, 0, 0, name, fontsize=7, color=color, visible=False
                )
        chain_connections = tuple(
            (connection_name, first, second)
            for connection_name, first, second in RAW_MOCAP_CONNECTIONS
            if connection_name == chain_name
        )
        for _, first, second in chain_connections:
            raw_chain_lines[(chain_name, first, second)] = ax3d.plot(
                [], [], [], "-", lw=2.2, color=color, alpha=0.95
            )[0]
    ax3d.legend(
        handles=[
            Line2D([0], [0], color="#f28e2b", lw=2.2, marker="o", ms=4.5, label="Raw thorax markers"),
            Line2D([0], [0], color="#59a14f", lw=2.2, marker="o", ms=4.5, label="Raw right-arm markers"),
        ],
        loc="upper left",
        fontsize=7,
        framealpha=0.8,
    )
    highlight_3d = ax3d.scatter([], [], [], facecolors="none", edgecolors="#d62828", s=180, linewidths=2.5, depthshade=False, visible=False)

    mmpose_body_lines = {}
    mmpose_scatter = None
    highlight_2d = None
    rgb_image = None
    depth_image = None
    if mmpose is None:
        ax2d.axis("off")
        ax2d.text(0.5, 0.5, "MMPose JSON not found", ha="center", va="center")
    else:
        body_indices = list(range(min(23, len(mmpose.labels))))
        if rgb_reader is not None:
            first_rgb_frame = rgb_reader.read(mmpose_start)
            if first_rgb_frame is not None:
                rgb_image = ax2d.imshow(
                    first_rgb_frame,
                    extent=(0, mmpose.image_width, mmpose.image_height, 0),
                    origin="upper",
                    aspect="auto",
                    zorder=0,
                )
        if depth_reader is not None:
            first_depth_frame = depth_reader.read(mmpose_start)
            if first_depth_frame is not None:
                depth_cmap = plt.get_cmap("turbo").copy()
                depth_cmap.set_bad(alpha=0.0)
                depth_image = ax2d.imshow(
                    np.ma.masked_invalid(first_depth_frame),
                    cmap=depth_cmap,
                    vmin=args.depth_min,
                    vmax=args.depth_max,
                    alpha=args.depth_alpha,
                    extent=(0, mmpose.image_width, mmpose.image_height, 0),
                    origin="upper",
                    aspect="auto",
                    zorder=1,
                    visible=False,
                )
        ax2d.set_xlim(0, mmpose.image_width)
        ax2d.set_ylim(mmpose.image_height, 0)
        ax2d.set_aspect("equal", adjustable="box")
        ax2d.set_xlabel("image x (px)")
        ax2d.set_ylabel("image y (px)")
        if depth_image is not None and rgb_image is not None:
            title = "RGB + Depth + MMPose 2D"
        elif rgb_image is not None:
            title = "RGB + MMPose 2D"
        else:
            title = "MMPose 2D"
        ax2d.set_title(title + "  |  red rings = selected metric joints")
        mmpose_scatter = ax2d.scatter(
            [], [], s=15, c="#457b9d", alpha=0.9, zorder=3
        )
        for first, second in MMPPOSE_BODY_CONNECTIONS:
            mmpose_body_lines[(first, second)] = ax2d.plot(
                [], [], "-", lw=1.8, color="#8d99ae", zorder=2
            )[0]
        highlight_2d = ax2d.scatter(
            [], [], facecolors="none", edgecolors="#d62828", s=430,
            linewidths=3.0, zorder=5, visible=False,
        )

    c3d_time = np.asarray(c3d_frames, dtype=float) / c3d.frame_rate
    ax_c3d_metric.set_xlim(c3d_time[0], c3d_time[-1] if len(c3d_time) > 1 else c3d_time[0] + 1.0)
    ax_c3d_metric.set_title("C3D metrics")
    ax_c3d_metric.set_xlabel("C3D time (s)")
    ax_c3d_metric.set_ylabel("degrees")
    ax_c3d_metric.grid(alpha=0.3)
    if mmpose is not None:
        mmpose_time = np.asarray(mmpose_frames, dtype=float) / mmpose.frame_rate
        ax_mmpose_metric.set_xlim(mmpose_time[0], mmpose_time[-1] if len(mmpose_time) > 1 else mmpose_time[0] + 1.0)
        ax_mmpose_metric.set_title("MMPose 2D / RGBD 3D metrics")
        ax_mmpose_metric.set_xlabel("MMPose time (s)")
        ax_mmpose_metric.set_ylabel("degrees")
        ax_mmpose_metric.grid(alpha=0.3)
    else:
        ax_mmpose_metric.axis("off")

    c3d_metric_lines = {}
    mmpose_metric_lines = {}
    rgbd_metric_lines = {}
    for display_name, metric_key, color in METRICS:
        values = c3d_elbow if metric_key == "elbow_flexion" else np.full(len(c3d_time), np.nan)
        c3d_metric_lines[metric_key] = ax_c3d_metric.plot(c3d_time, values, color=color, lw=1.5, visible=False)[0]
        if mmpose is not None:
            values = mmpose_elbow if metric_key == "elbow_flexion" else np.full(len(mmpose_time), np.nan)
            mmpose_metric_lines[metric_key] = ax_mmpose_metric.plot(mmpose_time, values, color=color, lw=1.5, visible=False)[0]
        if rgbd is not None and metric_key in (
            "elbow_flexion",
            "shoulder_flexion_extension",
            "shoulder_adduction",
        ):
            if metric_key == "elbow_flexion":
                values = rgbd_elbow
            elif metric_key == "shoulder_flexion_extension":
                values = np.full(len(mmpose_time), np.nan)
            else:
                values = np.full(len(mmpose_time), np.nan)
            rgbd_metric_lines[metric_key] = ax_mmpose_metric.plot(
                mmpose_time,
                values,
                color=color,
                lw=2.0,
                linestyle=":",
                visible=False,
            )[0]

    c3d_cursor = ax_c3d_metric.axvline(c3d_t1_default / c3d.frame_rate, color="#222222", lw=1.3)
    c3d_t1_marker = ax_c3d_metric.axvline(c3d_t1_default / c3d.frame_rate, color="#6a4c93", lw=1.1, linestyle="--", visible=False)
    c3d_t2_marker = ax_c3d_metric.axvline(c3d_t2_default / c3d.frame_rate, color="#1982c4", lw=1.1, linestyle="--", visible=False)
    c3d_status = ax_c3d_metric.text(0.01, 0.96, "", transform=ax_c3d_metric.transAxes, va="top", fontsize=8, color="#555555")
    mmpose_cursor = mmpose_t1_marker = mmpose_t2_marker = mmpose_status = None
    if mmpose is not None:
        mmpose_cursor = ax_mmpose_metric.axvline(mmpose_t1_default / mmpose.frame_rate, color="#222222", lw=1.3)
        mmpose_t1_marker = ax_mmpose_metric.axvline(mmpose_t1_default / mmpose.frame_rate, color="#6a4c93", lw=1.1, linestyle="--", visible=False)
        mmpose_t2_marker = ax_mmpose_metric.axvline(mmpose_t2_default / mmpose.frame_rate, color="#1982c4", lw=1.1, linestyle="--", visible=False)
        mmpose_status = ax_mmpose_metric.text(0.01, 0.96, "", transform=ax_mmpose_metric.transAxes, va="top", fontsize=8, color="#555555")

    metric_axis = fig.add_axes([0.82, 0.43, 0.16, 0.25])
    metric_checks = CheckButtons(metric_axis, [name for name, _, _ in METRICS], [False] * len(METRICS))
    metric_axis.set_title("Show metrics", fontsize=9)
    view_axis = fig.add_axes([0.83, 0.74, 0.15, 0.13])
    view_labels = ["3-D labels", "Raw mocap markers"]
    view_active = [False, bool(raw_marker_indices)]
    if depth_image is not None:
        view_labels.append("Depth overlay")
        view_active.append(False)
    view_checks = CheckButtons(view_axis, view_labels, view_active)
    view_axis.set_title("3-D view", fontsize=9)
    projection_axis = fig.add_axes([0.82, 0.23, 0.16, 0.15])
    projection_checks = RadioButtons(projection_axis, ["3D", "X-Y", "X-Z", "Y-Z"], active=0)
    projection_axis.set_title("C3D projection", fontsize=9)

    c3d_frame_axis = fig.add_axes([0.12, 0.265, 0.64, 0.025])
    c3d_frame_slider = Slider(c3d_frame_axis, "C3D frame", c3d_start, c3d_end, valinit=c3d_frame_default, valstep=1, color="#457b9d")
    mmpose_frame_axis = fig.add_axes([0.12, 0.225, 0.64, 0.025])
    if mmpose is not None:
        mmpose_frame_slider = Slider(mmpose_frame_axis, "MMPose frame", mmpose_start, mmpose_end, valinit=mmpose_t1_default, valstep=1, color="#457b9d")
    else:
        mmpose_frame_axis.set_visible(False)
        mmpose_frame_slider = None

    c3d_t1_axis = fig.add_axes([0.12, 0.175, 0.64, 0.025])
    c3d_t1_slider = Slider(c3d_t1_axis, "C3D T1", c3d_start, c3d_end, valinit=c3d_t1_default, valstep=1, color="#6a4c93")
    c3d_t2_axis = fig.add_axes([0.12, 0.14, 0.64, 0.025])
    c3d_t2_slider = Slider(c3d_t2_axis, "C3D T2", c3d_start, c3d_end, valinit=c3d_t2_default, valstep=1, color="#1982c4")
    mmpose_t1_axis = fig.add_axes([0.12, 0.09, 0.64, 0.025])
    mmpose_t2_axis = fig.add_axes([0.12, 0.055, 0.64, 0.025])
    if mmpose is not None:
        mmpose_t1_slider = Slider(mmpose_t1_axis, "iPad T1", mmpose_start, mmpose_end, valinit=mmpose_t1_default, valstep=1, color="#6a4c93")
        mmpose_t2_slider = Slider(mmpose_t2_axis, "iPad T2", mmpose_start, mmpose_end, valinit=mmpose_t2_default, valstep=1, color="#1982c4")
    else:
        mmpose_t1_axis.set_visible(False)
        mmpose_t2_axis.set_visible(False)
        mmpose_t1_slider = mmpose_t2_slider = None
    for axis in (c3d_t1_axis, c3d_t2_axis, mmpose_t1_axis, mmpose_t2_axis):
        axis.set_visible(False)

    state = {
        "c3d_frame": c3d_frame_default,
        "mmpose_frame": mmpose_t1_default,
        "c3d_t1": c3d_t1_default,
        "c3d_t2": c3d_t2_default,
        "mmpose_t1": mmpose_t1_default,
        "mmpose_t2": mmpose_t2_default,
        "selected_metrics": set(),
        "labels_visible": False,
        "raw_markers_visible": bool(raw_marker_indices),
        "playing": False,
        "speed": 1.0,
        "active_source": "c3d",
        "updating_reference_sliders": False,
        "depth_visible": False,
    }

    def format_metric(value):
        return "--" if not np.isfinite(value) else f"{value:+.1f} deg"

    def refresh_controls():
        visible = bool(state["selected_metrics"] & {"shoulder_flexion_extension", "shoulder_adduction"})
        for axis in (c3d_t1_axis, c3d_t2_axis):
            axis.set_visible(visible)
        if mmpose is not None:
            for axis in (mmpose_t1_axis, mmpose_t2_axis):
                axis.set_visible(visible)
        for marker in (c3d_t1_marker, c3d_t2_marker, mmpose_t1_marker, mmpose_t2_marker):
            if marker is not None:
                marker.set_visible(visible)
        if depth_image is not None:
            depth_image.set_visible(state["depth_visible"])

    def refresh_metric_legend():
        for axis in (ax_c3d_metric, ax_mmpose_metric):
            legend = axis.get_legend()
            if legend is not None:
                legend.remove()
        for axis, lines, source in ((ax_c3d_metric, c3d_metric_lines, "C3D"), (ax_mmpose_metric, mmpose_metric_lines, "2D")):
            if not lines:
                continue
            selected = [(name, key) for name, key, _ in METRICS if key in state["selected_metrics"]]
            if selected:
                handles = [lines[key] for _, key in selected]
                labels = [f"{name} ({source})" for name, _ in selected]
                if axis is ax_mmpose_metric:
                    for selected_name, selected_key in selected:
                        if selected_key in rgbd_metric_lines:
                            handles.append(rgbd_metric_lines[selected_key])
                            labels.append(f"{selected_name} (RGBD 3D)")
                axis.legend(handles, labels, loc="upper right", fontsize=7, framealpha=0.8)

    def refresh_metric_plots():
        c3d_t1_index = state["c3d_t1"] - c3d_start
        c3d_flexion_relative = _wrap_degrees(c3d_flexion - c3d_flexion[c3d_t1_index]) if np.isfinite(c3d_flexion[c3d_t1_index]) else np.full(len(c3d_flexion), np.nan)
        c3d_adduction_relative = _wrap_degrees(-(c3d_abduction - c3d_abduction[c3d_t1_index])) if np.isfinite(c3d_abduction[c3d_t1_index]) else np.full(len(c3d_abduction), np.nan)
        c3d_metric_lines["elbow_flexion"].set_ydata(c3d_elbow)
        c3d_metric_lines["shoulder_flexion_extension"].set_ydata(c3d_flexion_relative)
        c3d_metric_lines["shoulder_adduction"].set_ydata(c3d_adduction_relative)
        if mmpose is not None:
            mmpose_t1_index = state["mmpose_t1"] - mmpose_start
            mmpose_flexion_relative = _wrap_degrees(mmpose_flexion - mmpose_flexion[mmpose_t1_index]) if np.isfinite(mmpose_flexion[mmpose_t1_index]) else np.full(len(mmpose_flexion), np.nan)
            mmpose_adduction_relative = _wrap_degrees(-(mmpose_abduction - mmpose_abduction[mmpose_t1_index])) if np.isfinite(mmpose_abduction[mmpose_t1_index]) else np.full(len(mmpose_abduction), np.nan)
            mmpose_metric_lines["elbow_flexion"].set_ydata(mmpose_elbow)
            mmpose_metric_lines["shoulder_flexion_extension"].set_ydata(mmpose_flexion_relative)
            mmpose_metric_lines["shoulder_adduction"].set_ydata(mmpose_adduction_relative)
            if rgbd is not None and "elbow_flexion" in rgbd_metric_lines:
                rgbd_metric_lines["elbow_flexion"].set_ydata(rgbd_elbow)
            if rgbd is not None and "shoulder_flexion_extension" in rgbd_metric_lines:
                rgbd_t1_index = state["mmpose_t1"] - mmpose_start
                if np.isfinite(rgbd_flexion[rgbd_t1_index]):
                    rgbd_flexion_relative = _wrap_degrees(
                        rgbd_flexion - rgbd_flexion[rgbd_t1_index]
                    )
                else:
                    rgbd_flexion_relative = np.full(len(rgbd_flexion), np.nan)
                rgbd_metric_lines["shoulder_flexion_extension"].set_ydata(
                    rgbd_flexion_relative
                )
            if rgbd is not None and "shoulder_adduction" in rgbd_metric_lines:
                rgbd_t1_index = state["mmpose_t1"] - mmpose_start
                if np.isfinite(rgbd_adduction[rgbd_t1_index]):
                    rgbd_adduction_relative = _wrap_degrees(
                        rgbd_adduction - rgbd_adduction[rgbd_t1_index]
                    )
                else:
                    rgbd_adduction_relative = np.full(len(rgbd_adduction), np.nan)
                rgbd_metric_lines["shoulder_adduction"].set_ydata(
                    rgbd_adduction_relative
                )
        for key in c3d_metric_lines:
            c3d_metric_lines[key].set_visible(key in state["selected_metrics"])
        for key in mmpose_metric_lines:
            mmpose_metric_lines[key].set_visible(key in state["selected_metrics"])
        for key in rgbd_metric_lines:
            rgbd_metric_lines[key].set_visible(key in state["selected_metrics"])
        c3d_t1_marker.set_xdata([state["c3d_t1"] / c3d.frame_rate] * 2)
        c3d_t2_marker.set_xdata([state["c3d_t2"] / c3d.frame_rate] * 2)
        if mmpose is not None:
            mmpose_t1_marker.set_xdata([state["mmpose_t1"] / mmpose.frame_rate] * 2)
            mmpose_t2_marker.set_xdata([state["mmpose_t2"] / mmpose.frame_rate] * 2)
        if not args.auto_y:
            ax_c3d_metric.set_ylim(*FIXED_METRIC_Y_LIMITS)
            if mmpose is not None:
                ax_mmpose_metric.set_ylim(*FIXED_METRIC_Y_LIMITS)
        refresh_controls()

    def refresh_status():
        c3d_frame = state["c3d_frame"]
        c3d_text = [f"frame {c3d_frame}/{c3d_count - 1}   t={c3d_frame / c3d.frame_rate:.3f}s"]
        if "elbow_flexion" in state["selected_metrics"]:
            c3d_text.append(f"elbow {format_metric(elbow_flexion_angle(c3d, c3d_skeleton, c3d_frame))}")
        if "shoulder_flexion_extension" in state["selected_metrics"]:
            c3d_text.append(f"flex/ext {format_metric(shoulder_flexion_extension_t1_t2(c3d, c3d_skeleton, state['c3d_t1'], state['c3d_t2'], None, c3d_window_radius))}")
        if "shoulder_adduction" in state["selected_metrics"]:
            c3d_text.append(f"adduction {format_metric(shoulder_adduction_t1_t2(c3d, c3d_skeleton, state['c3d_t1'], state['c3d_t2'], None, c3d_window_radius))}")
        c3d_status.set_text("   ".join(c3d_text))
        if mmpose is not None and mmpose_status is not None:
            frame = state["mmpose_frame"]
            text = [f"frame {frame}/{mmpose_count - 1}   t={frame / mmpose.frame_rate:.3f}s"]
            if "elbow_flexion" in state["selected_metrics"]:
                text.append(f"elbow {format_metric(elbow_flexion_angle_2d(mmpose, mmpose_points, frame))}")
                if rgbd is not None:
                    text.append(f"RGBD {format_metric(rgbd_elbow_flexion_angle(rgbd, frame, mmpose_window_radius))}")
            if "shoulder_flexion_extension" in state["selected_metrics"]:
                text.append(f"flex/ext {format_metric(shoulder_flexion_extension_2d_t1_t2(mmpose, mmpose_points, state['mmpose_t1'], state['mmpose_t2'], mmpose_torso, mmpose_window_radius))}")
                if rgbd is not None:
                    text.append(f"RGBD {format_metric(shoulder_flexion_rgbd_t1_t2(rgbd_flexion, state['mmpose_t1'], state['mmpose_t2'], mmpose_start))}")
            if "shoulder_adduction" in state["selected_metrics"]:
                text.append(f"adduction {format_metric(shoulder_adduction_2d_t1_t2(mmpose, mmpose_points, state['mmpose_t1'], state['mmpose_t2'], mmpose_torso, mmpose_window_radius))}")
                if rgbd is not None:
                    text.append(f"RGBD {format_metric(shoulder_adduction_rgbd_t1_t2(rgbd_adduction, state['mmpose_t1'], state['mmpose_t2'], mmpose_start))}")
            mmpose_status.set_text("   ".join(text))

    def selected_highlights():
        names_3d, names_2d = set(), set()
        for key in state["selected_metrics"]:
            names_3d.update(METRIC_HIGHLIGHTS_3D[key])
            names_2d.update(METRIC_HIGHLIGHTS_2D[key])
        return names_3d, names_2d

    def update_c3d(frame):
        frame = max(c3d_start, min(c3d_end, int(frame)))
        state["c3d_frame"] = frame
        positions = {}
        for name, index in c3d_skeleton.items():
            if index is None or not valid_point(c3d, index, frame):
                positions[name] = None
                point_lines[name].set_visible(False)
                labels_text[name].set_visible(False)
                continue
            position = c3d.xyz[:, index, frame]
            positions[name] = position
            point_lines[name].set_data([position[0]], [position[1]])
            point_lines[name].set_3d_properties([position[2]])
            point_lines[name].set_visible(False)
            labels_text[name].set_position(tuple(position))
            labels_text[name].set_visible(state["labels_visible"])
        raw_positions = {}
        for name, index in raw_marker_indices.items():
            if index is None or not valid_point(c3d, index, frame):
                raw_positions[name] = None
                raw_point_lines[name].set_visible(False)
                raw_labels_text[name].set_visible(False)
                continue
            position = c3d.xyz[:, index, frame]
            raw_positions[name] = position
            raw_point_lines[name].set_data([position[0]], [position[1]])
            raw_point_lines[name].set_3d_properties([position[2]])
            raw_point_lines[name].set_visible(state["raw_markers_visible"])
            raw_labels_text[name].set_position(tuple(position))
            raw_labels_text[name].set_visible(
                state["raw_markers_visible"] and state["labels_visible"]
            )
        for (chain_name, first, second), line in raw_chain_lines.items():
            a, b = raw_positions[first], raw_positions[second]
            if not state["raw_markers_visible"] or a is None or b is None:
                line.set_visible(False)
            else:
                line.set_data([a[0], b[0]], [a[1], b[1]])
                line.set_3d_properties([a[2], b[2]])
                line.set_visible(True)
        for line in connection_lines.values():
            line.set_visible(False)
        names_3d, _ = selected_highlights()
        values = np.asarray([positions[name] for name in names_3d if positions.get(name) is not None])
        if values.size:
            highlight_3d.set_offsets(values[:, :2])
            highlight_3d.set_3d_properties(values[:, 2], "z")
            highlight_3d.set_visible(True)
        else:
            highlight_3d.set_visible(False)
        c3d_cursor.set_xdata([frame / c3d.frame_rate] * 2)
        refresh_status()

    def update_mmpose(frame):
        if mmpose is None or mmpose_scatter is None:
            return
        frame = max(mmpose_start, min(mmpose_end, int(frame)))
        state["mmpose_frame"] = frame
        if rgb_reader is not None and rgb_image is not None:
            current_rgb_frame = rgb_reader.read(frame)
            if current_rgb_frame is not None:
                rgb_image.set_data(current_rgb_frame)
        if depth_reader is not None and depth_image is not None:
            current_depth_frame = depth_reader.read(frame)
            if current_depth_frame is not None:
                depth_image.set_data(np.ma.masked_invalid(current_depth_frame))
            depth_image.set_visible(state["depth_visible"])
        values = mmpose.keypoints[frame]
        body = values[body_indices]
        valid = np.isfinite(body).all(axis=1)
        mmpose_scatter.set_offsets(body[valid])
        index_by_name = {label: index for index, label in enumerate(mmpose.labels)}
        for connection, line in mmpose_body_lines.items():
            first, second = index_by_name.get(connection[0]), index_by_name.get(connection[1])
            if first is None or second is None or not np.isfinite(values[[first, second]]).all():
                line.set_visible(False)
            else:
                line.set_data([values[first, 0], values[second, 0]], [values[first, 1], values[second, 1]])
                line.set_visible(True)
        _, names_2d = selected_highlights()
        indices = [mmpose_points[name] for name in names_2d if mmpose_points.get(name) is not None]
        selected = np.asarray([values[index] for index in indices if np.isfinite(values[index]).all()])
        if selected.size:
            highlight_2d.set_offsets(selected)
            highlight_2d.set_visible(True)
        else:
            highlight_2d.set_visible(False)
        mmpose_cursor.set_xdata([frame / mmpose.frame_rate] * 2)
        refresh_status()

    def redraw():
        refresh_metric_plots()
        update_c3d(state["c3d_frame"])
        if mmpose is not None:
            update_mmpose(state["mmpose_frame"])
        refresh_metric_legend()
        fig.canvas.draw_idle()

    c3d_frame_slider.on_changed(lambda value: (state.update(active_source="c3d"), update_c3d(value), refresh_metric_plots(), fig.canvas.draw_idle()) if not state["playing"] else None)
    if mmpose_frame_slider is not None:
        mmpose_frame_slider.on_changed(lambda value: (state.update(active_source="mmpose"), update_mmpose(value), refresh_metric_plots(), fig.canvas.draw_idle()) if not state["playing"] else None)

    def set_c3d_projection(mode):
        views = {
            "3D": (30, -60, (True, True, True), "C3D 3D"),
            "X-Y": (90, -90, (True, True, False), "C3D X-Y projection"),
            "X-Z": (0, -90, (True, False, True), "C3D X-Z projection (Y hidden)"),
            "Y-Z": (0, 0, (False, True, True), "C3D Y-Z projection (X hidden)"),
        }
        elevation, azimuth, visible_axes, title = views[mode]
        ax3d.set_proj_type("persp" if mode == "3D" else "ortho")
        ax3d.view_init(elev=elevation, azim=azimuth)
        axis_objects = (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis)
        axis_labels = (
            f"X ({c3d.point_units})",
            f"Y ({c3d.point_units})",
            f"Z ({c3d.point_units})",
        )
        for axis, visible, label in zip(axis_objects, visible_axes, axis_labels):
            axis.set_visible(visible)
            axis.label.set_text(label if visible else "")
            for tick_label in axis.get_ticklabels():
                tick_label.set_visible(visible)
        if mode == "3D":
            ax3d.set_xlim3d(original_c3d_limits[0])
            ax3d.set_ylim3d(original_c3d_limits[1])
            ax3d.set_zlim3d(original_c3d_limits[2])
        else:
            ax3d.set_xlim3d(readable_limits[0])
            ax3d.set_ylim3d(readable_limits[1])
            ax3d.set_zlim3d(readable_limits[2])
        if mode == "3D":
            ax3d.set_box_aspect((1, 1, 1))
        else:
            ax3d.set_box_aspect(
                tuple(high - low for low, high in readable_limits)
            )
        ax3d.set_title(title)
        fig.canvas.draw_idle()

    projection_checks.on_clicked(set_c3d_projection)

    def reference_changed(source, which, value):
        if state["updating_reference_sliders"]:
            return
        key = f"{source}_{which}"
        state[key] = int(value)
        other_key = f"{source}_{'t2' if which == 't1' else 't1'}"
        other_slider = {
            ("c3d", "t1"): c3d_t2_slider,
            ("c3d", "t2"): c3d_t1_slider,
            ("mmpose", "t1"): mmpose_t2_slider,
            ("mmpose", "t2"): mmpose_t1_slider,
        }[(source, which)]
        if (which == "t1" and state[key] > state[other_key]) or (which == "t2" and state[key] < state[other_key]):
            state[other_key] = state[key]
            state["updating_reference_sliders"] = True
            other_slider.set_val(state[key])
            state["updating_reference_sliders"] = False
        redraw()

    c3d_t1_slider.on_changed(lambda value: reference_changed("c3d", "t1", value))
    c3d_t2_slider.on_changed(lambda value: reference_changed("c3d", "t2", value))
    if mmpose is not None:
        mmpose_t1_slider.on_changed(lambda value: reference_changed("mmpose", "t1", value))
        mmpose_t2_slider.on_changed(lambda value: reference_changed("mmpose", "t2", value))

    def metric_clicked(display_name):
        key = next(metric_key for name, metric_key, _ in METRICS if name == display_name)
        if key in state["selected_metrics"]:
            state["selected_metrics"].remove(key)
        else:
            state["selected_metrics"].add(key)
        redraw()

    metric_checks.on_clicked(metric_clicked)

    def view_clicked(label):
        if label == "3-D labels":
            state["labels_visible"] = not state["labels_visible"]
            update_c3d(state["c3d_frame"])
        elif label == "Raw mocap markers":
            state["raw_markers_visible"] = not state["raw_markers_visible"]
            update_c3d(state["c3d_frame"])
        elif label == "Depth overlay":
            state["depth_visible"] = not state["depth_visible"]
            update_mmpose(state["mmpose_frame"])
        fig.canvas.draw_idle()

    view_checks.on_clicked(view_clicked)

    def on_key(event):
        if event.key in (" ", "p"):
            state["playing"] = not state["playing"]
            timer.start() if state["playing"] else timer.stop()
        elif event.key in ("left", "right"):
            slider = c3d_frame_slider if state["active_source"] == "c3d" else mmpose_frame_slider
            if slider is not None:
                slider.set_val(max(slider.valmin, min(slider.valmax, slider.val + (1 if event.key == "right" else -1))))
        elif event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    if rgb_reader is not None or depth_reader is not None:
        def close_readers(_event):
            if rgb_reader is not None:
                rgb_reader.close()
            if depth_reader is not None:
                depth_reader.close()

        fig.canvas.mpl_connect("close_event", close_readers)

    def advance(_event):
        if not state["playing"]:
            return
        c3d_next = state["c3d_frame"] + max(1, int(round(c3d.frame_rate / 30.0 * state["speed"])))
        state["c3d_frame"] = c3d_start if c3d_next > c3d_end else c3d_next
        c3d_frame_slider.set_val(state["c3d_frame"])
        if mmpose is not None:
            mmpose_next = state["mmpose_frame"] + max(1, int(round(mmpose.frame_rate / 30.0 * state["speed"])))
            state["mmpose_frame"] = mmpose_start if mmpose_next > mmpose_end else mmpose_next
            mmpose_frame_slider.set_val(state["mmpose_frame"])
        redraw()

    timer = animation.FuncAnimation(fig, advance, interval=33, blit=False, cache_frame_data=False)
    timer.event_source.stop()
    refresh_metric_plots()
    refresh_controls()
    update_c3d(state["c3d_frame"])
    if mmpose is not None:
        update_mmpose(state["mmpose_frame"])
    refresh_metric_legend()
    fig.canvas.draw_idle()
    print("GUI ready. C3D and MMPose have independent sliders; shoulder metrics have separate T1/T2 sliders.")
    plt.show()


if __name__ == "__main__":
    main()
