#!/usr/bin/env python3
"""Small paired-video ROM annotation server.

``rom_measure2`` intentionally has a narrower workflow than the original
RGBD/MMPose application:

* an admin imports one left/side session folder and one right/front session
  folder, or uploads one left video and one right/reference video;
* the admin chooses a frame independently in each video and saves a task;
* an annotator receives those fixed frames and places points on the task's
  target view: the left video for side-view tasks and the right video for
  front-view tasks;
* for shoulder-flexion tasks, the annotator places anonymous A-B-C points on
  the side view; for right hip abduction, the points are also placed on the
  side view; the server calculates the target-view angle at B and compares it
  with the corresponding MMPose point-cloud angle;
* for right hip internal/external rotation, the annotator places anonymous
  A-B-C points on the side view (right hip, right knee, right ankle), and the
  server measures the distal-leg excursion around the femur relative to the
  neutral starting side frame;
* the server calculates the task-specific instantaneous angle relative to its
  anatomical reference line; side-view shoulder angles are reconstructed from
  RGBD point-cloud coordinates.

The non-target video is stored and displayed for visual context. For a
side-view shoulder-flexion task, its fixed front frame is also used for the
separate front-view MMPose reference angle.
"""

from __future__ import annotations

import argparse
import hmac
from io import BytesIO
import json
import math
import os
import re
import secrets
import shutil
import time
import subprocess
import threading
import uuid
import zlib
from functools import wraps
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
from flask import (
    Flask,
    jsonify,
    redirect,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DEFAULT_STORAGE_DIR = (
    Path("/home/haziq/datasets/telept/data/milestone2") / "rom_measure2"
).resolve()
LEGACY_STORAGE_DIR = (BASE_DIR / "data").resolve()
_storage_override = os.environ.get("ROM2_STORAGE_DIR")
STORAGE_DIR = (
    Path(_storage_override).expanduser().resolve()
    if _storage_override
    else DEFAULT_STORAGE_DIR
)

# Keep existing local work available when the default storage location moves
# out of the source tree. This runs only on a new process, so the old server
# cannot be reading files while they are being relocated.
if not _storage_override and LEGACY_STORAGE_DIR.is_dir() and not STORAGE_DIR.exists():
    STORAGE_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(LEGACY_STORAGE_DIR), str(STORAGE_DIR))

UPLOADS_DIR = STORAGE_DIR / "uploads"
ANNOTATIONS_DIR = STORAGE_DIR / "annotations"
BROWSER_VIDEOS_DIR = STORAGE_DIR / "browser_videos"
PAIRS_FILE = STORAGE_DIR / "pairs.json"
TASKS_FILE = STORAGE_DIR / "tasks.json"

for directory in (STORAGE_DIR, UPLOADS_DIR, ANNOTATIONS_DIR, BROWSER_VIDEOS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

_BROWSER_PROXY_LOCK = threading.Lock()
_BROWSER_PROXY_STATUS_LOCK = threading.Lock()
_BROWSER_PROXY_STATUS = {}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
TASK_SCHEMA_VERSION = 1
ANNOTATION_SCHEMA_VERSION = 1

# The milestone2 iPad export stores all depth frames in one raw-DEFLATE
# stream. Each frame is 320 x 240 little-endian float32 values. The RGB video
# is portrait (240 x 320 aspect after rotation), while the stored depth map is
# landscape (320 x 240), so the map is rotated 90 degrees clockwise for
# display/registration.
RAW_DEPTH_WIDTH = 320
RAW_DEPTH_HEIGHT = 240
RAW_DEPTH_FRAME_BYTES = RAW_DEPTH_WIDTH * RAW_DEPTH_HEIGHT * 4
RAW_DEPTH_DISPLAY_WIDTH = RAW_DEPTH_HEIGHT
RAW_DEPTH_DISPLAY_HEIGHT = RAW_DEPTH_WIDTH
RAW_DEPTH_ROTATION = "cw90"
# These values are used only to make the depth layer visually readable. The
# actual 3-D calculation always uses the original floating-point depth.
DEPTH_COLOR_MIN_METERS = 0.2
DEPTH_COLOR_MAX_METERS = 8.0
DEPTH_COLOR_VALID_ALPHA = 150
DEPTH_COLOR_INVALID_ALPHA = 165
MMPose_MIN_KEYPOINT_SCORE = 0.20
MMPose_DISPLAY_POINTS = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)
MMPose_DISPLAY_SKELETON = (
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_heel", "left_big_toe"),
    ("left_heel", "left_small_toe"),
    ("left_big_toe", "left_small_toe"),
    ("right_heel", "right_big_toe"),
    ("right_heel", "right_small_toe"),
    ("right_big_toe", "right_small_toe"),
)

# Fallback for pose JSON files that do not include meta_info. The supplied
# RTMW export does include a full name-to-index map, but these first 23 names
# also make the reader work with ordinary COCO-style MMPose output.
COCO_KEYPOINT_NAME_TO_ID = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
    "left_big_toe": 17,
    "left_small_toe": 18,
    "left_heel": 19,
    "right_big_toe": 20,
    "right_small_toe": 21,
    "right_heel": 22,
}

_configured_source_roots = os.environ.get(
    "ROM2_ALLOWED_FOLDER_ROOTS",
    "/home/haziq/datasets/telept/data",
)
ALLOWED_FOLDER_ROOTS = tuple(
    Path(value).expanduser().resolve()
    for value in _configured_source_roots.split(os.pathsep)
    if value.strip()
)


# These templates describe the measurement, not just the three points in an
# included-angle calculation. ``view`` refers to the fixed pair convention:
# the left video is the side view and the right video is the front view.
# ``angle_points`` is the human-readable order for simple joint angles.
TASK_TEMPLATES = {
    "shoulder_flexion": {
        "label": "Right shoulder flexion",
        "view": "side",
        "angle_method": "side_trunk",
        "points": ["right_hip", "right_shoulder", "right_elbow"],
        "angle_points": ["right_hip", "right_shoulder", "right_elbow"],
        "vertex": "right_shoulder",
        "annotation_labels": ["A", "B", "C"],
        "direction": "flexion",
        "requires_depth": True,
        "description": "3-D side-view angle between the downward trunk line and the humerus.",
    },
    "shoulder_flexion_left": {
        "label": "Left shoulder flexion",
        "view": "side",
        "angle_method": "side_trunk",
        "points": ["left_hip", "left_shoulder", "left_elbow"],
        "angle_points": ["left_hip", "left_shoulder", "left_elbow"],
        "vertex": "left_shoulder",
        "annotation_labels": ["A", "B", "C"],
        "direction": "flexion",
        "requires_depth": True,
        "description": "3-D side-view angle between the downward trunk line and the humerus.",
    },
    "shoulder_extension": {
        "label": "Right shoulder extension",
        "view": "side",
        "angle_method": "side_trunk",
        "points": ["point_a", "point_b", "point_c"],
        "angle_points": ["point_a", "point_b", "point_c"],
        "vertex": "point_b",
        "annotation_labels": ["A", "B", "C"],
        "direction": "extension",
        "requires_depth": True,
        "description": "3-D side-view angle between the downward trunk line and the humerus.",
        "mmpose_points": ["right_hip", "right_shoulder", "right_elbow"],
        "mmpose_angle_method": "side_trunk",
        "mmpose_angle_points": ["right_hip", "right_shoulder", "right_elbow"],
        "mmpose_angle_vertex": "right_shoulder",
    },
    "shoulder_abduction": {
        "label": "Right shoulder abduction",
        "view": "front",
        "angle_method": "front_abduction_manual",
        "points": ["point_a", "point_b", "point_c"],
        "angle_points": ["point_a", "point_b", "point_c"],
        "vertex": "point_b",
        "annotation_labels": ["A", "B", "C"],
        "direction": "abduction",
        "requires_depth": True,
        # Brett's A-B-C points remain anonymous and can be placed according
        # to the clinician's reference line. MMPose uses the semantic torso
        # landmarks below to construct the shared frontal plane and its own
        # shoulder-to-elbow measurement.
        "mmpose_points": [
            "left_hip", "right_hip", "left_shoulder", "right_shoulder",
            "right_elbow",
        ],
        "mmpose_angle_method": "front_trunk",
        "mmpose_angle_points": ["right_hip", "right_shoulder", "right_elbow"],
        "mmpose_angle_vertex": "right_shoulder",
        "description": "Front-view 3-D shoulder-abduction angle using Brett's A-B-C reference and humerus points projected into the torso frontal plane.",
    },
    "shoulder_adduction": {
        "label": "Shoulder adduction",
        "view": "front",
        "angle_method": "front_trunk",
        "points": [
            "left_hip", "right_hip", "left_shoulder", "right_shoulder",
            "right_elbow",
        ],
        "angle_points": ["right_hip", "right_shoulder", "right_elbow"],
        "vertex": "right_shoulder",
        "direction": "adduction",
        "description": "Front-view angle after removing the anterior/posterior arm component.",
    },
    "shoulder_internal_rotation": {
        "label": "Right shoulder internal rotation",
        "view": "front",
        "angle_method": "shoulder_axial_manual",
        "points": ["point_a", "point_b", "point_c"],
        "angle_points": ["point_a", "point_b", "point_c"],
        "vertex": "point_b",
        "annotation_labels": ["A", "B", "C"],
        "direction": "internal rotation",
        "requires_depth": True,
        "mmpose_points": [
            "left_hip", "right_hip", "left_shoulder", "right_shoulder",
            "right_elbow", "right_wrist",
        ],
        "mmpose_angle_method": "shoulder_axial",
        "mmpose_angle_points": ["right_shoulder", "right_elbow", "right_wrist"],
        "mmpose_angle_vertex": "right_elbow",
        "annotation_instruction": "Place A at the shoulder, B at the elbow, and C at the wrist. The forearm's rotation around the A→B humerus axis is measured against the torso/chest reference.",
        "description": "Forearm orientation around the humerus relative to the torso anterior axis.",
    },
    "shoulder_external_rotation": {
        "label": "Right shoulder external rotation",
        "view": "front",
        "angle_method": "shoulder_axial_manual",
        "points": ["point_a", "point_b", "point_c"],
        "angle_points": ["point_a", "point_b", "point_c"],
        "vertex": "point_b",
        "annotation_labels": ["A", "B", "C"],
        "direction": "external rotation",
        "requires_depth": True,
        "mmpose_points": [
            "left_hip", "right_hip", "left_shoulder", "right_shoulder",
            "right_elbow", "right_wrist",
        ],
        "mmpose_angle_method": "shoulder_axial",
        "mmpose_angle_points": ["right_shoulder", "right_elbow", "right_wrist"],
        "mmpose_angle_vertex": "right_elbow",
        "annotation_instruction": "Place A at the shoulder, B at the elbow, and C at the wrist. The forearm's rotation around the A→B humerus axis is measured against the torso/chest reference.",
        "description": "Forearm orientation around the humerus relative to the torso anterior axis.",
    },
    "elbow_flexion": {
        "label": "Right elbow flexion",
        "view": "side",
        "angle_method": "included_complement",
        "points": ["point_a", "point_b", "point_c"],
        "angle_points": ["point_a", "point_b", "point_c"],
        "vertex": "point_b",
        "annotation_labels": ["A", "B", "C"],
        "direction": "flexion",
        "requires_depth": True,
        "mmpose_points": ["right_shoulder", "right_elbow", "right_wrist"],
        "mmpose_angle_method": "included_complement",
        "mmpose_angle_points": ["right_shoulder", "right_elbow", "right_wrist"],
        "mmpose_angle_vertex": "right_elbow",
        "description": "3-D shoulder–elbow–wrist angle, reported as flexion from full extension.",
    },
    "elbow_extension": {
        "label": "Elbow extension",
        "view": "side",
        "angle_method": "included_complement",
        "points": ["right_shoulder", "right_elbow", "right_wrist"],
        "angle_points": ["right_shoulder", "right_elbow", "right_wrist"],
        "vertex": "right_elbow",
        "direction": "extension",
        "description": "Shoulder–elbow–wrist angle, reported as flexion from full extension.",
    },
    "hip_flexion": {
        "label": "Right hip flexion",
        "view": "side",
        "angle_method": "side_trunk",
        "points": ["point_a", "point_b", "point_c"],
        "angle_points": ["point_a", "point_b", "point_c"],
        "vertex": "point_b",
        "annotation_labels": ["A", "B", "C"],
        "joint": "hip",
        "direction": "flexion",
        "requires_depth": True,
        "mmpose_points": ["right_shoulder", "right_hip", "right_knee"],
        "mmpose_angle_method": "side_trunk",
        "mmpose_angle_points": ["right_shoulder", "right_hip", "right_knee"],
        "mmpose_angle_vertex": "right_hip",
        "description": "3-D hip flexion from the trunk line to the thigh, with neutral at 0°.",
        "annotation_instruction": "Place A at the right shoulder, B at the right hip, and C at the right knee. The angle is measured at B.",
    },
    "hip_extension": {
        "label": "Hip extension",
        "view": "side",
        "angle_method": "side_trunk",
        "points": ["right_shoulder", "right_hip", "right_knee"],
        "angle_points": ["right_shoulder", "right_hip", "right_knee"],
        "vertex": "right_hip",
        "direction": "extension",
        "requires_depth": True,
        "description": "3-D side-view angle between the downward trunk line and the thigh.",
    },
    "hip_abduction": {
        "label": "Right hip abduction",
        "view": "side",
        "angle_method": "side_trunk",
        "points": ["point_a", "point_b", "point_c"],
        "angle_points": ["point_a", "point_b", "point_c"],
        "vertex": "point_b",
        "annotation_labels": ["A", "B", "C"],
        "joint": "hip",
        "direction": "abduction",
        "description": "Side-view 3-D hip-abduction estimate from the shoulder-to-hip trunk direction to the right thigh under the supine floor-sliding protocol.",
        "requires_depth": True,
        "mmpose_points": ["right_shoulder", "right_hip", "right_knee"],
        "mmpose_angle_method": "side_trunk",
        "mmpose_angle_points": ["right_shoulder", "right_hip", "right_knee"],
        "mmpose_angle_vertex": "right_hip",
        # The target/user annotation and the primary MMPose comparison use
        # the side view. The independent front-view computation is kept as
        # the paired reference result.
        "reference_mmpose_points": [
            "left_hip", "right_hip", "left_shoulder", "right_shoulder",
            "right_knee",
        ],
        "reference_mmpose_angle_method": "front_hip_abduction",
        "reference_mmpose_angle_points": ["right_hip", "right_knee"],
        "reference_mmpose_angle_vertex": "right_hip",
        "compare_reference_view": True,
        "geometry_version": 2,
        "annotation_instruction": "On the side-view image, place A at the right shoulder, B at the right hip, and C at the right knee. The angle is measured at B. The shoulder-to-hip direction is the trunk reference; under the supine floor-sliding protocol, the right thigh angle is reported as hip abduction.",
    },
    "hip_adduction": {
        "label": "Hip adduction",
        "view": "front",
        "angle_method": "front_pelvis",
        "points": [
            "left_hip", "right_hip", "left_shoulder", "right_shoulder",
            "right_knee",
        ],
        "angle_points": ["right_shoulder", "right_hip", "right_knee"],
        "vertex": "right_hip",
        "direction": "adduction",
        "description": "Front-view thigh angle relative to the pelvis/trunk downward axis.",
    },
    "hip_internal_rotation": {
        "label": "Right hip internal rotation",
        "view": "side",
        "angle_method": "hip_axial_manual",
        "points": ["point_a", "point_b", "point_c"],
        "angle_points": ["point_a", "point_b", "point_c"],
        "vertex": "point_b",
        "annotation_labels": ["A", "B", "C"],
        "direction": "internal rotation",
        "requires_depth": True,
        "mmpose_points": ["right_hip", "right_knee", "right_ankle"],
        "mmpose_angle_method": "hip_axial_side",
        "mmpose_angle_points": ["right_hip", "right_knee", "right_ankle"],
        "mmpose_angle_vertex": "right_knee",
        "neutral_frame_views": ["left"],
        "compare_reference_view": False,
        "annotation_instruction": "Place A at the right hip, B at the right knee, and C at the right ankle. The A→B line defines the femur axis; the B→C line defines the lower-leg direction. The 3-D hip rotation is measured relative to the admin-selected neutral side frame.",
        "description": "Side-view 3-D hip axial rotation around the femur, relative to an admin-selected neutral frame.",
    },
    "hip_external_rotation": {
        "label": "Right hip external rotation",
        "view": "side",
        "angle_method": "hip_axial_manual",
        "points": ["point_a", "point_b", "point_c"],
        "angle_points": ["point_a", "point_b", "point_c"],
        "vertex": "point_b",
        "annotation_labels": ["A", "B", "C"],
        "direction": "external rotation",
        "requires_depth": True,
        "mmpose_points": ["right_hip", "right_knee", "right_ankle"],
        "mmpose_angle_method": "hip_axial_side",
        "mmpose_angle_points": ["right_hip", "right_knee", "right_ankle"],
        "mmpose_angle_vertex": "right_knee",
        "neutral_frame_views": ["left"],
        "compare_reference_view": False,
        "annotation_instruction": "Place A at the right hip, B at the right knee, and C at the right ankle. The A→B line defines the femur axis; the B→C line defines the lower-leg direction. The 3-D hip rotation is measured relative to the admin-selected neutral side frame.",
        "description": "Side-view 3-D hip axial rotation around the femur, relative to an admin-selected neutral frame.",
    },
    "knee_flexion": {
        "label": "Right knee flexion",
        "view": "side",
        "angle_method": "included_complement",
        "points": ["point_a", "point_b", "point_c"],
        "angle_points": ["point_a", "point_b", "point_c"],
        "vertex": "point_b",
        "annotation_labels": ["A", "B", "C"],
        "direction": "flexion",
        "requires_depth": True,
        "mmpose_points": ["right_hip", "right_knee", "right_ankle"],
        "mmpose_angle_method": "included_complement",
        "mmpose_angle_points": ["right_hip", "right_knee", "right_ankle"],
        "mmpose_angle_vertex": "right_knee",
        "annotation_instruction": "Place A at the right hip, B at the right knee, and C at the right ankle. The angle is measured at B and reported as flexion from full knee extension.",
        "description": "3-D hip–knee–ankle angle, reported as flexion from full extension.",
    },
    "knee_extension": {
        "label": "Right knee extension",
        "view": "side",
        "angle_method": "included_complement",
        "points": ["point_a", "point_b", "point_c"],
        "angle_points": ["point_a", "point_b", "point_c"],
        "vertex": "point_b",
        "annotation_labels": ["A", "B", "C"],
        "direction": "extension",
        "requires_depth": True,
        "mmpose_points": ["right_hip", "right_knee", "right_ankle"],
        "mmpose_angle_method": "included_complement",
        "mmpose_angle_points": ["right_hip", "right_knee", "right_ankle"],
        "mmpose_angle_vertex": "right_knee",
        "annotation_instruction": "Place A at the right hip, B at the right knee, and C at the right ankle. The angle is measured at B and reported as extension from the fully extended knee position.",
        "description": "3-D hip–knee–ankle angle, with full knee extension reported as 0°.",
    },
    "ankle_dorsiflexion": {
        "label": "Right ankle dorsiflexion",
        "view": "side",
        "angle_method": "ankle_sagittal",
        "points": ["point_a", "point_b", "point_c"],
        "angle_points": ["point_a", "point_b", "point_c"],
        "vertex": "point_b",
        "annotation_labels": ["A", "B", "C"],
        "direction": "dorsiflexion",
        "requires_depth": True,
        "mmpose_points": [
            "right_knee", "right_ankle", "right_big_toe",
            "right_small_toe", "right_heel",
        ],
        "mmpose_angle_method": "ankle_sagittal",
        "mmpose_angle_points": ["right_knee", "right_ankle", "right_big_toe"],
        "mmpose_angle_vertex": "right_ankle",
        "compare_reference_view": False,
        "annotation_instruction": "Place A at the right knee, B at the right ankle, and C along the right foot toward the midpoint of the toes. The angle is measured at B and reported from the neutral ankle position (0°).",
        "description": "3-D ankle dorsiflexion from the tibia and foot axes, with neutral at 0°.",
    },
    "ankle_plantarflexion": {
        "label": "Right ankle plantarflexion",
        "view": "side",
        "angle_method": "ankle_sagittal",
        "points": ["point_a", "point_b", "point_c"],
        "angle_points": ["point_a", "point_b", "point_c"],
        "vertex": "point_b",
        "annotation_labels": ["A", "B", "C"],
        "direction": "plantarflexion",
        "requires_depth": True,
        "mmpose_points": [
            "right_knee", "right_ankle", "right_big_toe",
            "right_small_toe", "right_heel",
        ],
        "mmpose_angle_method": "ankle_sagittal",
        "mmpose_angle_points": ["right_knee", "right_ankle", "right_big_toe"],
        "mmpose_angle_vertex": "right_ankle",
        "compare_reference_view": False,
        "annotation_instruction": "Place A at the right knee, B at the right ankle, and C along the right foot toward the midpoint of the toes. The angle is measured at B and reported from the neutral ankle position (0°).",
        "description": "3-D ankle plantarflexion from the tibia and foot axes, with neutral at 0°.",
    },
}

# Keep the first validation slice deliberately small. The remaining templates
# stay in the module as design placeholders, but are not offered to admins
# until their view-specific clinical definitions have been validated.
ACTIVE_TEMPLATE_KEYS = (
    "shoulder_flexion",
    "shoulder_flexion_left",
    "shoulder_abduction",
    "shoulder_extension",
    "elbow_flexion",
    "shoulder_internal_rotation",
    "shoulder_external_rotation",
    "hip_abduction",
    "hip_internal_rotation",
    "hip_external_rotation",
    "hip_flexion",
    "knee_flexion",
    "knee_extension",
    "ankle_dorsiflexion",
    "ankle_plantarflexion",
)

HIP_ABDUCTION_GEOMETRY_VERSION = 2


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


MAX_UPLOAD_BYTES = _int_env("ROM2_MAX_UPLOAD_MB", 2048) * 1024 * 1024
ADMIN_USERNAME = os.environ.get("ROM2_ADMIN_USERNAME", "admin")
# Local-only defaults make the first run easy.  Set both variables before
# exposing the app to anyone else.
ADMIN_PASSWORD = os.environ.get("ROM2_ADMIN_PASSWORD", "123")
ANNOTATOR_PASSWORD = os.environ.get("ROM2_ANNOTATOR_PASSWORD", "123")
ROM2_USERS = {}
for user_pair in os.environ.get("ROM2_USERS", "").split(","):
    if ":" in user_pair:
        username, password = user_pair.split(":", 1)
        if username.strip():
            ROM2_USERS[username.strip()] = password


app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
app.secret_key = os.environ.get("ROM2_SECRET", secrets.token_hex(32))


# ---------- small persistence helpers ----------
def _read_json(path: Path, default):
    if not path.is_file():
        return default
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_pairs() -> list[dict]:
    payload = _read_json(PAIRS_FILE, {"pairs": []})
    pairs = payload.get("pairs", []) if isinstance(payload, dict) else []
    if not isinstance(pairs, list):
        raise ValueError("pairs index is invalid")
    return pairs


def _save_pairs(pairs: list[dict]) -> None:
    _write_json(PAIRS_FILE, {"schema_version": 1, "pairs": pairs})


def _upgrade_loaded_task(task: dict) -> bool:
    """Apply current hip-abduction view metadata to older saved tasks.

    Tasks are normally snapshots of the template at creation time. Hip
    abduction is the one intentional protocol change here: existing tasks
    created with the old front-view target should follow the new side-view
    workflow as well. The selected frame pair, task name, and annotation data
    are preserved.
    """
    if not isinstance(task, dict) or task.get("template") != "hip_abduction":
        return False
    template = TASK_TEMPLATES["hip_abduction"]
    version = int(template.get("geometry_version", HIP_ABDUCTION_GEOMETRY_VERSION))
    if task.get("geometry_version") == version:
        return False

    task.update({
        "required_points": list(template["points"]),
        "annotation_labels": list(template.get("annotation_labels", [])),
        "angle_points": list(template.get("angle_points", [])),
        "angle_vertex": template["vertex"],
        "view": "side",
        "view_label": "side view (left video)",
        "angle_method": template["angle_method"],
        "mmpose_points": list(template.get("mmpose_points", [])),
        "mmpose_angle_method": template.get("mmpose_angle_method"),
        "mmpose_angle_points": list(template.get("mmpose_angle_points", [])),
        "mmpose_angle_vertex": template.get("mmpose_angle_vertex"),
        "reference_mmpose_points": list(
            template.get("reference_mmpose_points", [])
        ),
        "reference_mmpose_angle_method": template.get(
            "reference_mmpose_angle_method"
        ),
        "reference_mmpose_angle_points": list(
            template.get("reference_mmpose_angle_points", [])
        ),
        "reference_mmpose_angle_vertex": template.get(
            "reference_mmpose_angle_vertex"
        ),
        "annotation_instruction": template.get("annotation_instruction", ""),
        "neutral_frame_index": None,
        "neutral_frame_indices": None,
        "neutral_frame_views": [],
        "compare_reference_view": True,
        "requires_depth": bool(template.get("requires_depth", False)),
        "direction": template.get("direction"),
        "joint": template.get("joint"),
        "angle_description": template.get("description", ""),
        "geometry_version": version,
        "target_frame_index": task.get("left_frame_index"),
        "angle_dimension": "3d",
    })
    # Warnings are frame/view-specific and must be recalculated for the new
    # side target and front reference. Do not expose stale old metadata.
    task.pop("mmpose_depth_warnings", None)

    # Keep the admin task card's MMPose availability fields aligned with the
    # new target frame when the pair index is available.
    try:
        pair = next(
            item for item in _load_pairs()
            if item.get("pair_id") == task.get("pair_id")
        )
        frame_info = _mmpose_frame_info(
            pair["left"],
            int(task["left_frame_index"]),
        )
    except (KeyError, TypeError, ValueError, OSError, FileNotFoundError, StopIteration, NameError):
        frame_info = None
    if isinstance(frame_info, dict):
        task["mmpose_available"] = frame_info.get("available", False)
        task["mmpose_frame_available"] = frame_info.get("frame_available", False)
        task["mmpose_frame_id"] = frame_info.get("frame_id")
    return True


def _load_tasks() -> list[dict]:
    payload = _read_json(TASKS_FILE, {"tasks": []})
    tasks = payload.get("tasks", []) if isinstance(payload, dict) else []
    if not isinstance(tasks, list):
        raise ValueError("tasks index is invalid")
    for task in tasks:
        _upgrade_loaded_task(task)
    return tasks


def _save_tasks(tasks: list[dict]) -> None:
    _write_json(TASKS_FILE, {"schema_version": TASK_SCHEMA_VERSION, "tasks": tasks})


# ---------- validation and metadata ----------
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_POINT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_ -]{0,31}$")
_USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_. -]{0,63}$")


def _safe_id(value, field: str) -> str:
    value = str(value or "").strip()
    if not value or not _ID_RE.fullmatch(value):
        raise ValueError(f"invalid {field}")
    return value


def _safe_username(value) -> str:
    value = str(value or "").strip()
    if not _USERNAME_RE.fullmatch(value):
        raise ValueError("username must use letters, numbers, spaces, . _ or -")
    return value


def _task_name(value, fallback: str) -> str:
    value = str(value or fallback).strip()
    if not value or len(value) > 120 or not value.isprintable():
        raise ValueError("task name must be printable and 1-120 characters")
    return value


def _video_metadata(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path.name}")
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_rate = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if width <= 0 or height <= 0 or frame_count <= 0:
        raise ValueError(f"video has no readable frames: {path.name}")
    if not math.isfinite(frame_rate) or frame_rate <= 0:
        frame_rate = 30.0
    return {
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "frame_rate": frame_rate,
        "duration_sec": max(0.0, (frame_count - 1) / frame_rate),
    }


def _frame_index(raw_value, video: dict, field: str) -> int:
    if isinstance(raw_value, bool):
        raise ValueError(f"{field} must be an integer")
    try:
        frame = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error
    if frame < 0 or frame >= int(video["frame_count"]):
        raise ValueError(f"{field} is outside the video: {frame}")
    return frame


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _allowed_source_path(raw_value, *, directory: bool = False) -> Path:
    raw_value = str(raw_value or "").strip()
    if not raw_value:
        raise ValueError("source path is required")
    path = Path(raw_value).expanduser()
    if not path.is_absolute():
        raise ValueError("source path must be an absolute path")
    path = path.resolve()
    if not any(_path_is_within(path, root) for root in ALLOWED_FOLDER_ROOTS):
        allowed = ", ".join(str(root) for root in ALLOWED_FOLDER_ROOTS)
        raise ValueError(f"source path must be inside an allowed root: {allowed}")
    if directory and not path.is_dir():
        raise FileNotFoundError(f"folder not found: {path}")
    if not directory and not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    return path


def _parse_header(raw: bytes) -> tuple[dict[str, str], int]:
    separator = re.search(br"\r?\n\r?\n", raw)
    if separator is None:
        raise ValueError("depth frame has no header separator")
    header = raw[:separator.end()].decode("utf-8", errors="replace")
    metadata = {}
    for line in header.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip().lower()] = value.strip()
    return metadata, separator.end()


class _ZipDepthReader:
    """Minimal reader for the original rom_measure depth.zip format."""

    def __init__(self, path: Path):
        self.path = path
        self.archive = ZipFile(path)
        self.entries = sorted(
            [name for name in self.archive.namelist() if not name.endswith("/")],
            key=lambda name: self._entry_number(name),
        )
        if not self.entries:
            self.archive.close()
            raise ValueError(f"depth archive has no frame files: {path}")
        self._cache: dict[int, tuple[np.ndarray, dict[str, str]]] = {}

    @staticmethod
    def _entry_number(name: str) -> int:
        match = re.search(r"(\d+)", Path(name).name)
        return int(match.group(1)) if match else -1

    def read(self, frame: int) -> tuple[np.ndarray | None, dict[str, str] | None]:
        if frame < 0 or frame >= len(self.entries):
            return None, None
        if frame in self._cache:
            return self._cache[frame]
        raw = self.archive.read(self.entries[frame])
        metadata, body_start = _parse_header(raw)
        width = int(metadata["width"])
        height = int(metadata["height"])
        expected_values = width * height
        expected_bytes = int(metadata.get("data_length", expected_values * 4))
        body = raw[body_start:body_start + expected_bytes]
        if len(body) < expected_values * 4:
            raise ValueError(f"depth frame is truncated: {self.entries[frame]}")
        depth = np.frombuffer(
            body[:expected_values * 4], dtype="<f4", count=expected_values
        ).reshape(height, width).copy()
        depth[(~np.isfinite(depth)) | (depth <= 0)] = np.nan
        result = (depth, metadata)
        self._cache[frame] = result
        return result

    def close(self) -> None:
        self.archive.close()


class _RawDeflateDepthReader:
    """Read one frame from milestone2's concatenated raw-DEFLATE depth stream.

    The stream has no per-frame headers. The accompanying instr_matrix JSON
    gives the RGB frame IDs that have depth; those IDs are sorted numerically
    because the JSON was produced asynchronously while the binary stream is
    chronological.
    """

    def __init__(self, path: Path, frame_indices: tuple[int, ...]):
        self.path = path
        self.frame_indices = frame_indices
        self.ordinal_by_frame = {
            frame: ordinal for ordinal, frame in enumerate(frame_indices)
        }
        self._cache: dict[int, np.ndarray] = {}
        # Keep the raw-DEFLATE stream open and advance it between sequential
        # reads. Re-starting the decoder from byte zero for every frame makes
        # an admin depth overlay unusably slow during video playback.
        self._stream_lock = threading.Lock()
        self._stream_file = None
        self._stream_decoder = None
        self._stream_pending = b""
        self._stream_produced = 0

    def _reset_stream(self) -> None:
        if self._stream_file is not None:
            self._stream_file.close()
        self._stream_file = self.path.open("rb")
        self._stream_decoder = zlib.decompressobj(-15)
        self._stream_pending = b""
        self._stream_produced = 0

    def read(self, frame: int) -> np.ndarray | None:
        ordinal = self.ordinal_by_frame.get(int(frame))
        if ordinal is None:
            return None
        with self._stream_lock:
            if ordinal in self._cache:
                return self._cache[ordinal]

            target_start = ordinal * RAW_DEPTH_FRAME_BYTES
            target_end = target_start + RAW_DEPTH_FRAME_BYTES
            if (
                self._stream_decoder is None
                or target_start < self._stream_produced
            ):
                self._reset_stream()

            selected = bytearray()
            while self._stream_produced < target_end:
                if not self._stream_pending:
                    self._stream_pending = self._stream_file.read(1024 * 1024)
                    if not self._stream_pending:
                        break
                try:
                    chunk = self._stream_decoder.decompress(
                        self._stream_pending,
                        min(
                            1024 * 1024,
                            target_end - self._stream_produced,
                        ),
                    )
                except zlib.error as error:
                    raise ValueError(
                        f"invalid raw depth DEFLATE stream: {self.path}"
                    ) from error
                self._stream_pending = self._stream_decoder.unconsumed_tail
                if chunk:
                    chunk_start = self._stream_produced
                    chunk_end = chunk_start + len(chunk)
                    overlap_start = max(target_start, chunk_start)
                    overlap_end = min(target_end, chunk_end)
                    if overlap_start < overlap_end:
                        selected.extend(
                            chunk[
                                overlap_start - chunk_start:
                                overlap_end - chunk_start
                            ]
                        )
                    self._stream_produced = chunk_end

                # A raw stream can end in the middle of an input block. Keep
                # any bytes after it as input for a possible concatenated
                # stream, then continue reading from the same file position.
                if self._stream_decoder.eof:
                    self._stream_pending = self._stream_decoder.unused_data
                    self._stream_decoder = zlib.decompressobj(-15)

            if len(selected) != RAW_DEPTH_FRAME_BYTES:
                raise ValueError(
                    f"raw depth stream ended before frame {frame} was complete"
                )
            depth = np.frombuffer(
                selected,
                dtype="<f4",
                count=RAW_DEPTH_WIDTH * RAW_DEPTH_HEIGHT,
            ).reshape(RAW_DEPTH_HEIGHT, RAW_DEPTH_WIDTH).copy()
            depth[(~np.isfinite(depth)) | (depth <= 0)] = np.nan
            if len(self._cache) >= 8:
                self._cache.pop(next(iter(self._cache)))
            self._cache[ordinal] = depth
            return depth

    def close(self) -> None:
        with self._stream_lock:
            if self._stream_file is not None:
                self._stream_file.close()
                self._stream_file = None
            self._stream_decoder = None
            self._stream_pending = b""


@lru_cache(maxsize=8)
def _raw_depth_reader(path_string: str, frame_indices: tuple[int, ...]):
    return _RawDeflateDepthReader(Path(path_string), frame_indices)


@lru_cache(maxsize=16)
def _matrix_payload(path_string: str) -> dict:
    payload = _read_json(Path(path_string), {})
    if not isinstance(payload, dict):
        raise ValueError(f"camera matrix file is not an object: {path_string}")
    return payload


def _matrix_intrinsics(matrix) -> dict[str, float]:
    try:
        fx = float(matrix[0][0])
        fy = float(matrix[1][1])
        cx = float(matrix[2][0])
        cy = float(matrix[2][1])
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError("camera matrix must contain a 3x3 numeric matrix") from error
    values = (fx, fy, cx, cy)
    if not all(math.isfinite(value) for value in values) or min(fx, fy) <= 0:
        raise ValueError("camera matrix contains invalid intrinsics")
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def _raw_frame_indices(matrix_path: Path) -> tuple[int, ...]:
    payload = _matrix_payload(str(matrix_path))
    indices = []
    for key in payload:
        try:
            indices.append(int(key))
        except (TypeError, ValueError) as error:
            raise ValueError(f"camera matrix has a non-numeric frame key: {key}") from error
    indices = tuple(sorted(set(indices)))
    if not indices:
        raise ValueError(f"camera matrix has no frame entries: {matrix_path.name}")
    return indices


def _raw_depth_descriptor(depth_path: Path, matrix_path: Path) -> dict:
    frame_indices = _raw_frame_indices(matrix_path)
    reader = _raw_depth_reader(str(depth_path), frame_indices)
    # Reading the first frame verifies that this is a raw-DEFLATE float32
    # stream before it is saved in the pair index.
    first_depth = reader.read(frame_indices[0])
    if first_depth is None or first_depth.shape != (RAW_DEPTH_HEIGHT, RAW_DEPTH_WIDTH):
        raise ValueError(f"unsupported raw depth stream: {depth_path.name}")
    return {
        "format": "raw_deflate_float32",
        "path": str(depth_path),
        "intrinsics_path": str(matrix_path),
        "frame_indices": list(frame_indices),
        "width": RAW_DEPTH_WIDTH,
        "height": RAW_DEPTH_HEIGHT,
        "display_width": RAW_DEPTH_DISPLAY_WIDTH,
        "display_height": RAW_DEPTH_DISPLAY_HEIGHT,
        "rotation": RAW_DEPTH_ROTATION,
    }


def _zip_depth_descriptor(depth_path: Path) -> dict:
    reader = _ZipDepthReader(depth_path)
    try:
        depth, metadata = reader.read(0)
        if depth is None or metadata is None:
            raise ValueError(f"depth archive has no readable first frame: {depth_path.name}")
        return {
            "format": "depth_zip",
            "path": str(depth_path),
            "frame_count": len(reader.entries),
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
        }
    finally:
        reader.close()


def _find_first(files: list[Path], predicate) -> Path | None:
    matches = sorted((path for path in files if predicate(path)), key=lambda path: path.name.lower())
    return matches[0] if matches else None


def _find_mmpose_json(folder: Path) -> Path | None:
    """Find the RGB MMPose export belonging to a session folder.

    The milestone2 export currently uses ``mmpose/<model>/rgb/rgb.json``.
    Keeping this discovery narrow avoids accidentally treating unrelated JSON
    files in a session as pose results.
    """
    candidates = sorted(
        (
            path for path in folder.glob("mmpose/*/rgb/rgb.json")
            if path.is_file()
        ),
        key=lambda path: str(path).lower(),
    )
    return candidates[0] if candidates else None


def _mmpose_descriptor(json_path: Path | None) -> dict | None:
    if json_path is None:
        return None
    video_path = json_path.with_name("rgb.mp4")
    return {
        "format": "mmpose_json",
        "path": str(json_path),
        "model": json_path.parent.parent.name,
        "video_path": str(video_path) if video_path.is_file() else None,
    }


def _discover_session_folder(raw_folder) -> dict:
    folder = _allowed_source_path(raw_folder, directory=True)
    files = sorted(
        (
            path for path in folder.iterdir()
            if path.is_file() and not path.name.startswith(".")
        ),
        key=lambda path: path.name.lower(),
    )
    video_candidates = [
        path for path in files if path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    preferred_videos = [
        path for path in video_candidates
        if path.name.lower().startswith("rgb_video_")
    ]
    video_candidates = preferred_videos or video_candidates
    if not video_candidates:
        raise FileNotFoundError(f"no supported RGB video found in {folder}")
    if len(video_candidates) > 1:
        names = ", ".join(path.name for path in video_candidates)
        raise ValueError(f"folder contains multiple videos; choose a session folder with one RGB video: {names}")
    video_path = video_candidates[0]

    raw_depth_path = _find_first(
        files,
        lambda path: path.name.lower().startswith("depth_data_")
        or path.name.lower() == "depth_data",
    )
    matrix_path = _find_first(
        files,
        lambda path: path.name.lower().startswith("instr_matrix_")
        and not path.name.lower().startswith("rgb_instr_matrix_")
        and path.suffix.lower() == ".json",
    )
    zip_path = _find_first(
        files,
        lambda path: path.suffix.lower() == ".zip"
        and "depth" in path.name.lower(),
    )
    depth = None
    if raw_depth_path is not None:
        if matrix_path is None:
            raise FileNotFoundError(
                f"raw depth is present but instr_matrix_*.json is missing in {folder}"
            )
        depth = _raw_depth_descriptor(raw_depth_path, matrix_path)
    elif zip_path is not None:
        depth = _zip_depth_descriptor(zip_path)

    rgb_matrix_path = _find_first(
        files,
        lambda path: path.name.lower().startswith("rgb_instr_matrix_")
        and path.suffix.lower() == ".json",
    )
    mmpose_json_path = _find_mmpose_json(folder)
    return {
        "folder_path": str(folder),
        "folder_name": folder.name,
        "video_path": str(video_path),
        "original_name": video_path.name,
        "metadata": _video_metadata(video_path),
        "depth": depth,
        "rgb_intrinsics_path": str(rgb_matrix_path) if rgb_matrix_path else None,
        "mmpose": _mmpose_descriptor(mmpose_json_path),
    }


def _public_depth(depth: dict | None) -> dict | None:
    if not isinstance(depth, dict):
        return None
    return {
        "format": depth.get("format"),
        "width": depth.get("width"),
        "height": depth.get("height"),
        "display_width": depth.get("display_width"),
        "display_height": depth.get("display_height"),
        # ``ccw90`` was written by an earlier rom_measure2 build. The raw
        # milestone2 sensor layout is now known to require clockwise rotation,
        # so expose the effective transform rather than stale index metadata.
        "rotation": (
            RAW_DEPTH_ROTATION
            if depth.get("format") == "raw_deflate_float32"
            else depth.get("rotation")
        ),
        "frame_count": depth.get("frame_count", len(depth.get("frame_indices", []))),
    }


def _public_mmpose(mmpose: dict | None) -> dict:
    if not isinstance(mmpose, dict) or not mmpose.get("path"):
        return {"available": False}
    return {
        "available": True,
        "format": mmpose.get("format"),
        "model": mmpose.get("model"),
        "video_available": bool(mmpose.get("video_path")),
    }


def _public_discovery(discovery: dict) -> dict:
    return {
        "folder": discovery["folder_path"],
        "folder_name": discovery["folder_name"],
        "video": {
            "original_name": discovery["original_name"],
            **discovery["metadata"],
        },
        "depth": _public_depth(discovery.get("depth")),
        "mmpose": _public_mmpose(discovery.get("mmpose")),
        "has_rgb_intrinsics": bool(discovery.get("rgb_intrinsics_path")),
    }


def _folder_entry(discovery: dict) -> dict:
    return {
        "source_type": "folder",
        "source_folder": discovery["folder_path"],
        "video_path": discovery["video_path"],
        "original_name": discovery["original_name"],
        "rgb_intrinsics_path": discovery.get("rgb_intrinsics_path"),
        **discovery["metadata"],
        "depth": discovery.get("depth"),
        "mmpose": discovery.get("mmpose"),
    }


def _depth_frame_available(entry: dict, frame: int) -> bool:
    depth = entry.get("depth")
    if not isinstance(depth, dict):
        return True
    if depth.get("format") == "raw_deflate_float32":
        return int(frame) in {
            int(value) for value in depth.get("frame_indices", [])
        }
    if depth.get("format") == "depth_zip":
        return 0 <= int(frame) < int(depth.get("frame_count", 0))
    return False


VIEW_TO_SIDE = {
    "side": "left",
    "front": "right",
}


def _task_view(value) -> str:
    view = str(value or "side").strip().lower()
    if view not in VIEW_TO_SIDE:
        raise ValueError("view must be side or front")
    return view


def _view_entry(pair: dict, view: str) -> dict:
    view = _task_view(view)
    entry = pair.get(VIEW_TO_SIDE[view])
    if not isinstance(entry, dict):
        raise FileNotFoundError(f"{view} video is missing")
    return entry


def _view_frame(task: dict, view: str) -> int:
    view = _task_view(view)
    field = "left_frame_index" if view == "side" else "right_frame_index"
    return int(task[field])


def _neutral_frame_for_view(
    task: dict,
    view: str,
    default: int | None = None,
) -> int | None:
    """Return the task's hidden neutral frame for a paired-video view."""
    view = _task_view(view)
    side = VIEW_TO_SIDE[view]
    neutral_frames = task.get("neutral_frame_indices")
    if isinstance(neutral_frames, dict) and side in neutral_frames:
        try:
            return int(neutral_frames[side])
        except (TypeError, ValueError):
            return default
    # Older hip-rotation tasks used one side-only neutral frame.
    if view == "side" and task.get("neutral_frame_index") is not None:
        try:
            return int(task["neutral_frame_index"])
        except (TypeError, ValueError):
            return default
    return default


def _view_label(view: str) -> str:
    return "side view (left video)" if _task_view(view) == "side" else "front view (right video)"


def _new_pair(left: dict, right: dict) -> dict:
    return {
        "pair_id": f"pair_{int(time.time())}_{uuid.uuid4().hex[:10]}",
        "left": left,
        "right": right,
        "created_at": time.time(),
    }


def _find_pair(pair_id: str) -> dict | None:
    pair_id = _safe_id(pair_id, "pair_id")
    return next((pair for pair in _load_pairs() if pair.get("pair_id") == pair_id), None)


def _find_task(task_id: str) -> dict | None:
    task_id = _safe_id(task_id, "task_id")
    return next((task for task in _load_tasks() if task.get("task_id") == task_id), None)


def _pair_video_path(pair: dict, side: str) -> Path:
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    pair_id = _safe_id(pair["pair_id"], "pair_id")
    entry = pair.get(side)
    if not isinstance(entry, dict):
        raise FileNotFoundError(f"{side} video is missing")
    if entry.get("source_type") == "folder" or entry.get("video_path"):
        # Imported files are never copied into the app. Revalidate the stored
        # path on every request so a pair cannot become a file server for an
        # arbitrary path after the index is edited.
        return _allowed_source_path(entry.get("video_path"))
    filename = Path(str(entry.get("filename", ""))).name
    if (
        not filename
        or filename != secure_filename(filename)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", filename)
    ):
        raise ValueError("invalid video filename")
    pair_dir = (UPLOADS_DIR / pair_id).resolve()
    if pair_dir.parent != UPLOADS_DIR:
        raise ValueError("invalid pair storage path")
    path = (pair_dir / filename).resolve()
    if path.parent != pair_dir or not path.is_file():
        raise FileNotFoundError(f"{side} video file is missing")
    return path


def _exact_frame_jpeg(pair: dict, side: str, raw_frame: int) -> bytes:
    """Encode one exact source-video frame as a JPEG.

    Browser video seeking is deliberately not used here. The browser-safe
    H.264 proxy can have slightly different timing/frame counts from the
    original iPad HEVC file, while MMPose frame IDs refer to the original RGB
    sequence. Serving the decoded source frame makes the annotator image and
    its MMPose coordinates use the same zero-based frame index.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    entry = pair.get(side)
    if not isinstance(entry, dict):
        raise FileNotFoundError(f"{side} video is missing")
    frame = _frame_index(raw_frame, entry, f"{side} frame")
    source = _pair_video_path(pair, side)
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f"could not open video: {source.name}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        readable, image = capture.read()
    finally:
        capture.release()
    if not readable or image is None:
        raise ValueError(f"could not decode {side} frame {frame}")
    expected_width = int(entry.get("width", 0) or 0)
    expected_height = int(entry.get("height", 0) or 0)
    if expected_width and expected_height and (
        image.shape[1] != expected_width or image.shape[0] != expected_height
    ):
        raise ValueError(
            f"decoded {side} frame dimensions do not match the imported RGB video"
        )
    encoded, jpeg = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
    )
    if not encoded:
        raise ValueError(f"could not encode {side} frame {frame}")
    return jpeg.tobytes()


@lru_cache(maxsize=64)
def _video_codec(path_string: str) -> str | None:
    """Read the video codec without decoding the full file."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path_string,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    codec = result.stdout.strip().splitlines()
    return codec[0].strip().lower() if codec else None


@lru_cache(maxsize=1)
def _h264_ffmpeg() -> str | None:
    """Find an FFmpeg executable that actually provides the libx264 encoder."""
    candidates = []
    first_candidate = shutil.which("ffmpeg")
    if first_candidate:
        candidates.append(first_candidate)
    for candidate in ("/usr/bin/ffmpeg", "/bin/ffmpeg"):
        if candidate not in candidates and Path(candidate).is_file():
            candidates.append(candidate)
    for candidate in candidates:
        try:
            result = subprocess.run(
                [candidate, "-hide_banner", "-encoders"],
                check=True,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if "libx264" in (result.stdout + result.stderr):
            return candidate
    return None


def _set_browser_proxy_status(
    pair_id: str,
    side: str,
    state: str,
    progress: float | None = None,
    message: str = "",
) -> None:
    with _BROWSER_PROXY_STATUS_LOCK:
        _BROWSER_PROXY_STATUS[(pair_id, side)] = {
            "state": state,
            "progress": (
                None
                if progress is None
                else max(0.0, min(100.0, float(progress)))
            ),
            "message": message,
        }


def _browser_proxy_status(pair: dict, side: str) -> dict:
    """Return conversion status without starting a conversion."""
    pair_id = _safe_id(pair["pair_id"], "pair_id")
    source = _pair_video_path(pair, side)
    proxy = (BROWSER_VIDEOS_DIR / pair_id / f"{side}.mp4").resolve()
    if proxy.is_file() and proxy.stat().st_mtime >= source.stat().st_mtime:
        return {"state": "ready", "progress": 100.0, "message": "Ready"}
    if _video_codec(str(source)) in {None, "h264", "avc1"}:
        return {"state": "ready", "progress": 100.0, "message": "Ready"}
    with _BROWSER_PROXY_STATUS_LOCK:
        status = dict(_BROWSER_PROXY_STATUS.get((pair_id, side), {}))
    return status or {
        "state": "idle",
        "progress": 0.0,
        "message": "Waiting for the browser video request.",
    }


def _browser_video_path(pair: dict, side: str) -> Path:
    """Return a browser-compatible H.264 video for a pair member.

    iPad exports are commonly HEVC/H.265. OpenCV and ffmpeg can read those
    files, but Chrome on Linux may leave an HEVC ``<video>`` element at 0:00.
    The proxy keeps the original resolution and frame order so the existing
    RGB/MMPose coordinates remain valid.
    """
    source = _pair_video_path(pair, side)
    if _video_codec(str(source)) in {None, "h264", "avc1"}:
        return source

    pair_id = _safe_id(pair["pair_id"], "pair_id")
    proxy_dir = (BROWSER_VIDEOS_DIR / pair_id).resolve()
    if proxy_dir.parent != BROWSER_VIDEOS_DIR:
        raise ValueError("invalid browser proxy path")
    proxy_dir.mkdir(parents=True, exist_ok=True)
    proxy = proxy_dir / f"{side}.mp4"
    pair_id = _safe_id(pair["pair_id"], "pair_id")

    if proxy.is_file() and proxy.stat().st_mtime >= source.stat().st_mtime:
        _set_browser_proxy_status(pair_id, side, "ready", 100.0, "Ready")
        return proxy

    _set_browser_proxy_status(
        pair_id,
        side,
        "queued",
        0.0,
        "Waiting for another video conversion to finish.",
    )

    with _BROWSER_PROXY_LOCK:
        if proxy.is_file() and proxy.stat().st_mtime >= source.stat().st_mtime:
            _set_browser_proxy_status(pair_id, side, "ready", 100.0, "Ready")
            return proxy
        ffmpeg = _h264_ffmpeg()
        if not ffmpeg:
            message = "this video uses a browser-incompatible codec, but no FFmpeg build with libx264 was found"
            _set_browser_proxy_status(pair_id, side, "error", 0.0, message)
            raise RuntimeError(
                message
            )
        temporary = proxy_dir / f".{side}.{uuid.uuid4().hex}.tmp.mp4"
        duration = float(pair.get(side, {}).get("duration_sec", 0.0) or 0.0)
        _set_browser_proxy_status(
            pair_id,
            side,
            "converting",
            0.0,
            "Creating browser-compatible video…",
        )
        try:
            print(
                f"[rom_measure2] Creating browser video proxy for {source.name} ({side}); please wait...",
                flush=True,
            )
            with subprocess.Popen(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel", "error",
                    "-y",
                    "-i", str(source),
                    "-nostats",
                    "-progress", "pipe:1",
                    "-map", "0:v:0",
                    "-an",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    # Keep the source frame order/count for RGB/MMPose
                    # coordinate alignment. ``-vsync 0`` is supported by the
                    # older FFmpeg builds commonly present on workstations.
                    "-vsync", "0",
                    str(temporary),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            ) as process:
                for line in process.stdout or ():
                    key, separator, value = line.strip().partition("=")
                    if not separator:
                        continue
                    if key in {"out_time_us", "out_time_ms"}:
                        try:
                            elapsed = float(value) / 1_000_000.0
                        except ValueError:
                            continue
                        progress = 0.0
                        if duration > 0:
                            progress = min(99.0, max(0.0, elapsed / duration * 100.0))
                        _set_browser_proxy_status(
                            pair_id,
                            side,
                            "converting",
                            progress,
                            f"Creating browser-compatible video… {progress:.0f}%",
                        )
                return_code = process.wait(timeout=1800)
                error_output = process.stderr.read() if process.stderr else ""
            if return_code:
                raise subprocess.CalledProcessError(
                    return_code,
                    ffmpeg,
                    stderr=error_output,
                )
            os.replace(temporary, proxy)
            _set_browser_proxy_status(pair_id, side, "ready", 100.0, "Ready")
            print(f"[rom_measure2] Browser video proxy ready: {proxy}", flush=True)
        except subprocess.CalledProcessError as error:
            temporary.unlink(missing_ok=True)
            detail = (error.stderr or "").strip().splitlines()
            message = detail[-1] if detail else "ffmpeg could not convert the video"
            _set_browser_proxy_status(pair_id, side, "error", 0.0, message)
            raise RuntimeError(message) from error
        except subprocess.TimeoutExpired as error:
            temporary.unlink(missing_ok=True)
            message = "H.264 browser proxy conversion timed out"
            _set_browser_proxy_status(pair_id, side, "error", 0.0, message)
            raise RuntimeError(message) from error
        except OSError as error:
            temporary.unlink(missing_ok=True)
            _set_browser_proxy_status(pair_id, side, "error", 0.0, str(error))
            raise
    return proxy


def _public_video(pair_id: str, side: str, entry: dict) -> dict:
    return {
        "original_name": entry.get("original_name", entry.get("filename", "")),
        "filename": entry.get("filename", ""),
        "source_type": entry.get("source_type", "upload"),
        "source_folder_name": Path(str(entry.get("source_folder", ""))).name or None,
        "width": entry.get("width"),
        "height": entry.get("height"),
        "frame_count": entry.get("frame_count"),
        "frame_rate": entry.get("frame_rate"),
        "duration_sec": entry.get("duration_sec"),
        "depth": _public_depth(entry.get("depth")),
        "mmpose": _public_mmpose(entry.get("mmpose")),
        "url": url_for("media_file", pair_id=pair_id, side=side),
    }


def _public_pair(pair: dict) -> dict:
    pair_id = _safe_id(pair["pair_id"], "pair_id")
    return {
        "pair_id": pair_id,
        "created_at": pair.get("created_at"),
        "left": _public_video(pair_id, "left", pair["left"]),
        "right": _public_video(pair_id, "right", pair["right"]),
    }


_NEUTRAL_METADATA_FIELDS = {
    "neutral_frame_index",
    "neutral_frame_indices",
    "neutral_frame_views",
    "reference_frame_index",
    "neutral_mmpose_frame_id",
}


def _redact_neutral_metadata(value):
    """Remove neutral-frame identifiers from annotator-facing payloads."""
    if not isinstance(value, dict):
        return value
    result = {
        key: item
        for key, item in value.items()
        if key not in _NEUTRAL_METADATA_FIELDS
    }
    for key in ("mmpose", "reference_mmpose", "side_mmpose", "front_mmpose"):
        if isinstance(result.get(key), dict):
            result[key] = _redact_neutral_metadata(result[key])
    return result


def _public_task(
    task: dict,
    pair: dict | None = None,
    include_neutral_metadata: bool = True,
) -> dict:
    result = dict(task)
    # Tasks created before side-trunk depth was made mandatory have the old
    # 2-D metadata on disk. Keep their public representation consistent with
    # the calculation path used by the current server.
    if result.get("angle_method") == "side_trunk":
        result["requires_depth"] = True
        result["angle_dimension"] = "3d"
    if pair is not None:
        result["pair"] = _public_pair(pair)
    if not include_neutral_metadata:
        # Invalid-depth diagnostics are an admin frame-selection aid and are
        # not part of the annotator-facing task payload.
        result.pop("mmpose_depth_warnings", None)
        result = _redact_neutral_metadata(result)
    return result


def _annotation_path(username: str, task_id: str) -> Path:
    username = _safe_username(username)
    task_id = _safe_id(task_id, "task_id")
    # secure_filename is only used for the directory component; username has
    # already been restricted to a single safe path component above.
    username_dir = secure_filename(username)
    if not username_dir:
        raise ValueError("invalid username")
    return ANNOTATIONS_DIR / username_dir / f"{task_id}.json"


def _load_annotation(username: str, task_id: str) -> dict | None:
    path = _annotation_path(username, task_id)
    if not path.is_file():
        return None
    payload = _read_json(path, None)
    return payload if isinstance(payload, dict) else None


def _save_annotation(annotation: dict) -> None:
    path = _annotation_path(annotation["username"], annotation["task_id"])
    _write_json(path, annotation)


# ---------- angle calculation ----------
def _sample_depth(depth: np.ndarray, u: float, v: float, radius: int) -> float:
    column = int(round(u))
    row = int(round(v))
    first_column = max(0, column - radius)
    last_column = min(depth.shape[1], column + radius + 1)
    first_row = max(0, row - radius)
    last_row = min(depth.shape[0], row + radius + 1)
    if first_column >= last_column or first_row >= last_row:
        return math.nan
    patch = depth[first_row:last_row, first_column:last_column]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(valid)) if valid.size else math.nan


def _depth_intrinsics(depth: dict, frame: int) -> dict[str, float] | None:
    matrix_path = _allowed_source_path(depth["intrinsics_path"])
    matrix = _matrix_payload(str(matrix_path)).get(str(int(frame)))
    if matrix is None:
        return None
    return _matrix_intrinsics(matrix)


def _read_depth_for_frame(entry: dict, frame: int) -> dict | None:
    depth = entry.get("depth")
    if not isinstance(depth, dict):
        return None
    depth_format = depth.get("format")
    if depth_format == "raw_deflate_float32":
        frame_indices = tuple(int(value) for value in depth.get("frame_indices", []))
        if int(frame) not in frame_indices:
            return {
                "available": False,
                "format": depth_format,
                "width": depth.get("width", RAW_DEPTH_WIDTH),
                "height": depth.get("height", RAW_DEPTH_HEIGHT),
            }
        depth_path = _allowed_source_path(depth["path"])
        reader = _raw_depth_reader(str(depth_path), frame_indices)
        depth_array = reader.read(int(frame))
        intrinsics = _depth_intrinsics(depth, int(frame))
        if depth_array is None or intrinsics is None:
            return {
                "available": False,
                "format": depth_format,
                "width": depth.get("width", RAW_DEPTH_WIDTH),
                "height": depth.get("height", RAW_DEPTH_HEIGHT),
            }
        return {
            "available": True,
            "format": depth_format,
            "array": depth_array,
            "width": int(depth.get("width", RAW_DEPTH_WIDTH)),
            "height": int(depth.get("height", RAW_DEPTH_HEIGHT)),
            "display_width": int(depth.get("display_width", RAW_DEPTH_DISPLAY_WIDTH)),
            "display_height": int(depth.get("display_height", RAW_DEPTH_DISPLAY_HEIGHT)),
            # The first rom_measure2 version stored ``ccw90`` here, but that
            # was a registration mistake rather than source-data metadata.
            # Normalize every milestone2 raw stream to the verified transform
            # so existing pair indexes are corrected on reload.
            "rotation": RAW_DEPTH_ROTATION,
            **intrinsics,
        }
    if depth_format == "depth_zip":
        depth_path = _allowed_source_path(depth["path"])
        reader = _ZipDepthReader(depth_path)
        try:
            depth_array, metadata = reader.read(int(frame))
        finally:
            reader.close()
        if depth_array is None or metadata is None:
            return {
                "available": False,
                "format": depth_format,
                "width": depth.get("width"),
                "height": depth.get("height"),
            }
        return {
            "available": True,
            "format": depth_format,
            "array": depth_array,
            "width": int(metadata["width"]),
            "height": int(metadata["height"]),
            "rotation": None,
            "fx": float(metadata["fx"]),
            "fy": float(metadata["fy"]),
            "cx": float(metadata["ox"]),
            "cy": float(metadata["oy"]),
        }
    raise ValueError(f"unsupported depth format: {depth_format}")


def _depth_color_png(entry: dict, frame: int) -> bytes:
    """Create a small transparent depth-color layer for one RGB frame.

    The raw depth is kept at its native resolution and rotated into the same
    portrait orientation as the RGB video. The browser stretches this layer
    over the RGB frame. Valid depth is colored blue-to-red (near-to-far); no
    depth is rendered as zero-RGB black so it is clearly visible as an area to
    avoid when annotating.
    """
    depth_info = _read_depth_for_frame(entry, frame)
    if depth_info is None:
        raise FileNotFoundError("this session has no depth source")
    if not depth_info.get("available"):
        raise FileNotFoundError(f"RGB frame {frame} has no depth frame")

    depth = np.asarray(depth_info["array"], dtype=np.float32)
    if depth_info.get("rotation") == RAW_DEPTH_ROTATION:
        depth = np.rot90(depth, 3)
    valid = np.isfinite(depth) & (depth > 0)

    # OpenCV's JET map is blue at the near end and red at the far end. Values
    # outside this display range are clipped only for visualization.
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(
            (depth[valid] - DEPTH_COLOR_MIN_METERS)
            / (DEPTH_COLOR_MAX_METERS - DEPTH_COLOR_MIN_METERS),
            0.0,
            1.0,
        )
        normalized[valid] = np.rint(clipped * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)
    rgba[valid, 3] = DEPTH_COLOR_VALID_ALPHA
    # Invalid depth is deliberately black rather than transparent. Zeroing the
    # RGB channels lets the annotator distinguish "no measurable point cloud"
    # from valid depth at a glance; the alpha keeps the underlying frame
    # visible enough to identify the body and scene.
    rgba[~valid, 0:3] = 0
    rgba[~valid, 3] = DEPTH_COLOR_INVALID_ALPHA

    encoded, buffer = cv2.imencode(
        ".png",
        rgba,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    )
    if not encoded:
        raise RuntimeError("could not encode the depth overlay")
    return buffer.tobytes()


def _rgb_to_depth_pixel(
    u: float,
    v: float,
    rgb_video: dict,
    depth_info: dict,
) -> tuple[float, float]:
    if depth_info.get("rotation") == RAW_DEPTH_ROTATION:
        # First map RGB coordinates to the displayed (portrait) depth map,
        # then undo the 90-degree clockwise display rotation for array sampling and
        # the original depth-camera pinhole intrinsics.
        display_u = (u + 0.5) * depth_info["display_width"] / rgb_video["width"] - 0.5
        display_v = (v + 0.5) * depth_info["display_height"] / rgb_video["height"] - 0.5
        return (
            display_v,
            depth_info["height"] - 1.0 - display_u,
        )
    return (
        (u + 0.5) * depth_info["width"] / rgb_video["width"] - 0.5,
        (v + 0.5) * depth_info["height"] / rgb_video["height"] - 0.5,
    )


def _lift_points(
    points: dict[str, dict[str, float]],
    video: dict,
    frame: int,
    depth_sample_radius: int = 2,
) -> dict:
    depth_info = _read_depth_for_frame(video, frame)
    if depth_info is None:
        return {
            "has_depth": False,
            "frame_available": False,
            "points": {},
            "pixels": {},
            "depths": {},
        }
    if not depth_info.get("available"):
        return {
            "has_depth": True,
            "frame_available": False,
            "format": depth_info.get("format"),
            "points": {},
            "pixels": {},
            "depths": {},
        }

    point_3d = {}
    pixels = {}
    depths = {}
    for name, value in points.items():
        depth_u, depth_v = _rgb_to_depth_pixel(
            float(value["x"]),
            float(value["y"]),
            video,
            depth_info,
        )
        depth_value = _sample_depth(
            depth_info["array"],
            depth_u,
            depth_v,
            max(0, int(depth_sample_radius)),
        )
        pixels[name] = {"x": depth_u, "y": depth_v}
        depths[name] = depth_value
        if not math.isfinite(depth_value):
            continue
        point_3d[name] = np.asarray(
            (
                (depth_u - depth_info["cx"]) * depth_value / depth_info["fx"],
                (depth_v - depth_info["cy"]) * depth_value / depth_info["fy"],
                depth_value,
            ),
            dtype=float,
        )
    return {
        "has_depth": True,
        "frame_available": True,
        "format": depth_info.get("format"),
        "points": point_3d,
        "pixels": pixels,
        "depths": depths,
    }


@lru_cache(maxsize=8)
def _mmpose_payload(path_string: str) -> dict:
    """Load and index one MMPose RGB JSON export.

    The current export has one record per frame, uses one-based frame IDs,
    and stores one or more instances under ``instances``. The highest-scoring
    instance is used when more than one person is present.
    """
    path = _allowed_source_path(path_string)
    payload = _read_json(path, None)
    if not isinstance(payload, dict):
        raise ValueError("MMPose JSON must contain an object")

    meta_info = payload.get("meta_info")
    meta_info = meta_info if isinstance(meta_info, dict) else {}
    raw_mapping = meta_info.get("keypoint_name2id")
    mapping = {}
    if isinstance(raw_mapping, dict):
        for name, index in raw_mapping.items():
            try:
                mapping[str(name)] = int(index)
            except (TypeError, ValueError):
                continue
    if not mapping:
        raw_reverse_mapping = meta_info.get("keypoint_id2name")
        if isinstance(raw_reverse_mapping, dict):
            for index, name in raw_reverse_mapping.items():
                try:
                    mapping[str(name)] = int(index)
                except (TypeError, ValueError):
                    continue
    if not mapping:
        mapping = dict(COCO_KEYPOINT_NAME_TO_ID)

    frame_records = {}
    raw_records = payload.get("instance_info", [])
    if not isinstance(raw_records, list):
        raise ValueError("MMPose JSON has no instance_info list")
    for record in raw_records:
        if not isinstance(record, dict):
            continue
        try:
            frame_id = int(record["frame_id"])
        except (KeyError, TypeError, ValueError):
            continue
        raw_instances = record.get("instances")
        if not isinstance(raw_instances, list):
            raw_instances = [record]
        instances = [item for item in raw_instances if isinstance(item, dict)]
        if not instances:
            continue

        def instance_score(instance: dict) -> float:
            try:
                value = float(instance.get("bbox_score", 0.0))
            except (TypeError, ValueError):
                return 0.0
            return value if math.isfinite(value) else 0.0

        instance = max(instances, key=instance_score)
        keypoints = np.asarray(instance.get("keypoints", []), dtype=float)
        scores = np.asarray(instance.get("keypoint_scores", []), dtype=float).reshape(-1)
        if keypoints.ndim != 2 or keypoints.shape[1] < 2:
            continue
        frame_records[frame_id] = {
            "keypoints": keypoints[:, :2],
            "scores": scores,
        }
    if not frame_records:
        raise ValueError("MMPose JSON has no readable frame records")

    frame_id_base = 0 if min(frame_records) == 0 else 1
    return {
        "mapping": mapping,
        "frames": frame_records,
        "frame_id_base": frame_id_base,
        "frame_count": len(frame_records),
    }


@lru_cache(maxsize=16)
def _cached_video_metadata(path_string: str) -> dict:
    return _video_metadata(_allowed_source_path(path_string))


def _mmpose_points(
    entry: dict,
    task: dict,
    video: dict,
    frame: int,
) -> dict:
    """Return semantic MMPose points for the task's exact target frame."""
    description = entry.get("mmpose")
    required = list(
        task.get("mmpose_points")
        or task.get("required_points", [])
    )
    empty = {
        "available": False,
        "frame_available": False,
        "points": {},
        "scores": {},
        "missing_points": list(required),
        "frame_id": None,
        "reason": "No MMPose RGB JSON was found for this session.",
    }
    if not isinstance(description, dict) or not description.get("path"):
        return empty

    try:
        pose = _mmpose_payload(str(description["path"]))
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        return {
            **empty,
            "reason": f"MMPose output could not be read: {error}",
        }

    frame_id = int(frame) + int(pose.get("frame_id_base", 1))
    record = pose["frames"].get(frame_id)
    if record is None:
        return {
            "available": True,
            "frame_available": False,
            "points": {},
            "scores": {},
            "missing_points": list(required),
            "frame_id": frame_id,
            "reason": f"MMPose has no result for RGB frame {frame} (pose frame_id {frame_id}).",
        }

    source_width = float(video.get("width", 0) or 0)
    source_height = float(video.get("height", 0) or 0)
    pose_video_path = description.get("video_path")
    if pose_video_path:
        try:
            pose_metadata = _cached_video_metadata(str(pose_video_path))
            if pose_metadata.get("width") and pose_metadata.get("height"):
                source_width = float(pose_metadata["width"])
                source_height = float(pose_metadata["height"])
        except (FileNotFoundError, OSError, TypeError, ValueError):
            # The JSON coordinates are normally already in the source RGB
            # coordinate system, so the target dimensions remain a safe
            # fallback when the derived MMPose preview video is absent.
            pass

    target_width = float(video.get("width", 0) or 0)
    target_height = float(video.get("height", 0) or 0)
    scale_x = target_width / source_width if source_width > 0 else 1.0
    scale_y = target_height / source_height if source_height > 0 else 1.0
    points = {}
    scores = {}
    missing = []
    mapping = pose.get("mapping", {})
    keypoints = record["keypoints"]
    keypoint_scores = record.get("scores", np.asarray([], dtype=float))
    for name in required:
        index = mapping.get(name)
        if index is None or index < 0 or index >= len(keypoints):
            missing.append(name)
            continue
        coordinate = keypoints[index]
        score = (
            float(keypoint_scores[index])
            if index < len(keypoint_scores)
            else 1.0
        )
        x = float(coordinate[0]) * scale_x
        y = float(coordinate[1]) * scale_y
        if (
            not math.isfinite(x)
            or not math.isfinite(y)
            or not math.isfinite(score)
            or score < MMPose_MIN_KEYPOINT_SCORE
            or not (0 <= x <= target_width and 0 <= y <= target_height)
        ):
            missing.append(name)
            continue
        points[name] = {"x": x, "y": y}
        scores[name] = score

    return {
        "available": True,
        "frame_available": True,
        "points": points,
        "scores": scores,
        "missing_points": missing,
        "frame_id": frame_id,
        "reason": (
            ""
            if not missing
            else "MMPose keypoints below confidence threshold or unavailable: "
            + ", ".join(missing)
        ),
    }


def _side_hip_neutral_reference(task: dict, pair: dict) -> dict:
    """Build the one-sided hip-rotation reference from the neutral side frame.

    The neutral reference intentionally uses only the working leg. It avoids
    treating the right shoulder-to-right-hip edge of the body as the torso
    centreline, which would be biased because shoulder and pelvic widths differ.
    """
    entry = _view_entry(pair, "side")
    neutral_frame = _neutral_frame_for_view(task, "side", default=0)
    if neutral_frame is None:
        neutral_frame = 0
    required = ["right_hip", "right_knee", "right_ankle"]
    model = _mmpose_points(
        entry,
        {"required_points": required},
        entry,
        neutral_frame,
    )
    lifted = _lift_points(model.get("points", {}), entry, neutral_frame)
    points_3d = lifted.get("points", {})
    missing = list(model.get("missing_points", []))
    missing.extend(name for name in required if name not in points_3d)
    missing = list(dict.fromkeys(missing))
    if (
        not model.get("available")
        or not model.get("frame_available")
        or not lifted.get("frame_available")
        or missing
    ):
        return {
            "available": False,
            "frame_index": neutral_frame,
            "frame_id": model.get("frame_id"),
            "missing_points": missing,
            "reason": model.get("reason") or "Neutral side frame has no valid 3-D hip-rotation landmarks.",
        }

    femur_axis = _unit_vector(
        points_3d["right_knee"] - points_3d["right_hip"]
    )
    radial_reference = _unit_vector(
        _project_out(
            points_3d["right_ankle"] - points_3d["right_knee"],
            femur_axis,
        )
    )
    if femur_axis is None or radial_reference is None:
        return {
            "available": False,
            "frame_index": neutral_frame,
            "frame_id": model.get("frame_id"),
            "missing_points": [],
            "reason": "Neutral side frame has an unstable femur or lower-leg axis.",
        }
    return {
        "available": True,
        "frame_index": neutral_frame,
        "frame_id": model.get("frame_id"),
        "femur_axis": femur_axis,
        "radial_reference": radial_reference,
        "points_3d": points_3d,
        "missing_points": [],
        "model": model.get("model"),
    }


def _hip_abduction_neutral_reference(
    task: dict,
    pair: dict,
    view: str,
) -> dict:
    """Build a one-sided neutral thigh reference for hip abduction.

    Hip abduction is evaluated from the right hip-to-knee direction. The
    neutral frame makes the calculation usable even when the contralateral
    shoulder and hip are occluded in the side view; those landmarks are not
    part of this reference.
    """
    view = _task_view(view)
    entry = _view_entry(pair, view)
    neutral_frame = _neutral_frame_for_view(task, view)
    required = ["right_hip", "right_knee"]
    if neutral_frame is None:
        return {
            "available": False,
            "view": view,
            "frame_index": None,
            "frame_id": None,
            "missing_points": ["neutral frame"],
            "reason": f"No neutral frame was selected for the {view} view.",
        }

    model = _mmpose_points(
        entry,
        {"required_points": required},
        entry,
        neutral_frame,
    )
    lifted = _lift_points(model.get("points", {}), entry, neutral_frame)
    points_3d = lifted.get("points", {})
    missing = list(model.get("missing_points", []))
    missing.extend(name for name in required if name not in points_3d)
    missing = list(dict.fromkeys(missing))
    if (
        not model.get("available")
        or not model.get("frame_available")
        or not lifted.get("frame_available")
        or missing
    ):
        return {
            "available": False,
            "view": view,
            "frame_index": neutral_frame,
            "frame_id": model.get("frame_id"),
            "missing_points": missing,
            "reason": model.get("reason") or "Neutral frame has no valid 3-D hip landmarks.",
        }

    thigh_reference = _unit_vector(
        points_3d["right_knee"] - points_3d["right_hip"]
    )
    if thigh_reference is None:
        return {
            "available": False,
            "view": view,
            "frame_index": neutral_frame,
            "frame_id": model.get("frame_id"),
            "missing_points": [],
            "reason": "Neutral frame has an unstable right hip-to-knee axis.",
        }
    return {
        "available": True,
        "view": view,
        "frame_index": neutral_frame,
        "frame_id": model.get("frame_id"),
        "thigh_reference": thigh_reference,
        "points_3d": points_3d,
        "missing_points": [],
        "model": model.get("model"),
    }


def _mmpose_all_points(entry: dict, video: dict) -> dict:
    """Return all display landmarks indexed by zero-based RGB frame."""
    description = entry.get("mmpose")
    if not isinstance(description, dict) or not description.get("path"):
        return {
            "available": False,
            "frames": {},
            "skeleton": [list(link) for link in MMPose_DISPLAY_SKELETON],
            "reason": "No MMPose RGB JSON was found for this session.",
        }
    try:
        pose = _mmpose_payload(str(description["path"]))
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        return {
            "available": False,
            "frames": {},
            "skeleton": [list(link) for link in MMPose_DISPLAY_SKELETON],
            "reason": f"MMPose output could not be read: {error}",
        }

    task = {"required_points": list(MMPose_DISPLAY_POINTS)}
    frames = {}
    frame_base = int(pose.get("frame_id_base", 1))
    frame_count = int(video.get("frame_count", 0) or 0)
    for frame_id in sorted(pose.get("frames", {})):
        frame = int(frame_id) - frame_base
        if frame < 0 or frame >= frame_count:
            continue
        result = _mmpose_points(entry, task, video, frame)
        frames[str(frame)] = {
            "available": result.get("available", False),
            "frame_available": result.get("frame_available", False),
            "frame_index": frame,
            "frame_id": result.get("frame_id"),
            "points": result.get("points", {}),
            "scores": result.get("scores", {}),
            "missing_points": result.get("missing_points", []),
            "reason": result.get("reason", ""),
        }
    return {
        "available": True,
        "frames": frames,
        "frame_count": len(frames),
        "skeleton": [list(link) for link in MMPose_DISPLAY_SKELETON],
    }


def _mmpose_frame_info(entry: dict, frame: int) -> dict:
    """Report whether the selected RGB frame has an exact pose record."""
    description = entry.get("mmpose")
    if not isinstance(description, dict) or not description.get("path"):
        return {
            "available": False,
            "frame_available": False,
            "frame_id": None,
        }
    try:
        pose = _mmpose_payload(str(description["path"]))
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return {
            "available": False,
            "frame_available": False,
            "frame_id": None,
        }
    frame_id = int(frame) + int(pose.get("frame_id_base", 1))
    return {
        "available": True,
        "frame_available": frame_id in pose.get("frames", {}),
        "frame_id": frame_id,
    }


def _normalise_points(raw_points, task: dict, video: dict) -> dict[str, dict[str, float]]:
    if raw_points is None:
        raw_points = {}
    if not isinstance(raw_points, dict):
        raise ValueError("points must be an object")
    required = task["required_points"]
    normalised = {}
    for name in required:
        if name not in raw_points or raw_points[name] is None:
            continue
        value = raw_points[name]
        if not isinstance(value, dict):
            raise ValueError(f"point {name} must contain x and y")
        try:
            x = float(value["x"])
            y = float(value["y"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"point {name} must contain numeric x and y") from error
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(f"point {name} is not finite")
        if not (0 <= x <= video["width"] and 0 <= y <= video["height"]):
            raise ValueError(f"point {name} is outside the target video")
        normalised[name] = {"x": x, "y": y}
    return normalised


def _normalise_left_points(raw_points, task: dict, left_video: dict) -> dict[str, dict[str, float]]:
    """Backwards-compatible wrapper for old left-only task payloads."""
    return _normalise_points(raw_points, task, left_video)


def _angle_deg(points: dict, task: dict) -> float | None:
    order = task["angle_points"]
    if any(name not in points for name in order):
        return None
    first = points[order[0]]
    vertex = points[order[1]]
    second = points[order[2]]
    first_vector = (first["x"] - vertex["x"], first["y"] - vertex["y"])
    second_vector = (second["x"] - vertex["x"], second["y"] - vertex["y"])
    first_length = math.hypot(*first_vector)
    second_length = math.hypot(*second_vector)
    if first_length < 1e-9 or second_length < 1e-9:
        return None
    cosine = (
        first_vector[0] * second_vector[0]
        + first_vector[1] * second_vector[1]
    ) / (first_length * second_length)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _angle_3d(points: dict, task: dict) -> float | None:
    return _angle_3d_order(points, task["angle_points"])


def _angle_3d_order(points: dict, order: list[str] | tuple[str, ...]) -> float | None:
    if any(name not in points for name in order):
        return None
    first = np.asarray(points[order[0]], dtype=float)
    vertex = np.asarray(points[order[1]], dtype=float)
    second = np.asarray(points[order[2]], dtype=float)
    first_vector = first - vertex
    second_vector = second - vertex
    first_length = float(np.linalg.norm(first_vector))
    second_length = float(np.linalg.norm(second_vector))
    if first_length < 1e-9 or second_length < 1e-9:
        return None
    cosine = float(np.dot(first_vector, second_vector) / (first_length * second_length))
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def _unit_vector(value) -> np.ndarray | None:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size != 3 or not np.isfinite(vector).all():
        return None
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        return None
    return vector / length


def _project_out(vector, axis) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=float)
    axis = _unit_vector(axis)
    if axis is None or vector.size != 3 or not np.isfinite(vector).all():
        return None
    return vector - float(np.dot(vector, axis)) * axis


def _signed_angle_3d(first, second, axis) -> float | None:
    first = _unit_vector(first)
    second = _unit_vector(second)
    axis = _unit_vector(axis)
    if first is None or second is None or axis is None:
        return None
    cosine = max(-1.0, min(1.0, float(np.dot(first, second))))
    sine = float(np.dot(axis, np.cross(first, second)))
    return math.degrees(math.atan2(sine, cosine))


def _rotation_aligning_vectors(first, second) -> np.ndarray | None:
    """Return the smallest 3-D rotation that maps ``first`` onto ``second``."""
    first = _unit_vector(first)
    second = _unit_vector(second)
    if first is None or second is None:
        return None

    cross = np.cross(first, second)
    sine = float(np.linalg.norm(cross))
    cosine = max(-1.0, min(1.0, float(np.dot(first, second))))
    if sine < 1e-9:
        if cosine > 0:
            return np.eye(3, dtype=float)
        # The anti-parallel case has infinitely many solutions. Choose a
        # deterministic axis perpendicular to ``first`` so neutral-reference
        # transport remains stable instead of producing NaNs.
        basis = np.eye(3, dtype=float)[int(np.argmin(np.abs(first)))]
        axis = _unit_vector(np.cross(first, basis))
        if axis is None:
            return None
        return -np.eye(3, dtype=float) + 2.0 * np.outer(axis, axis)

    axis = cross / sine
    skew = np.array(
        (
            (0.0, -axis[2], axis[1]),
            (axis[2], 0.0, -axis[0]),
            (-axis[1], axis[0], 0.0),
        ),
        dtype=float,
    )
    return (
        np.eye(3, dtype=float)
        + skew
        + skew @ skew * ((1.0 - cosine) / (sine * sine))
    )


def _hip_axial_angle(
    points_3d: dict[str, np.ndarray],
    order: list[str] | tuple[str, ...],
    neutral_reference: dict | None,
) -> dict:
    """Measure side-view hip rotation relative to the neutral side frame.

    ``order`` is either the anonymous Brett order (A, B, C) or the semantic
    MMPose order (right hip, right knee, right ankle). The first segment is
    the femur axis and the second segment is the rotating lower-leg vector;
    this deliberately is not the ordinary included angle at the knee.
    """
    if len(order) != 3:
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": [],
            "note": "Hip axial rotation requires hip, knee, and ankle points.",
        }

    missing = [name for name in order if name not in points_3d]
    if missing:
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": missing,
            "note": "Side-view 3-D hip rotation requires valid depth for the hip, knee, and ankle.",
        }
    if not isinstance(neutral_reference, dict) or not neutral_reference.get("available"):
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": ["neutral reference"],
            "note": "Side-view 3-D hip rotation requires a valid neutral starting-frame reference.",
        }

    femur_axis = _unit_vector(
        np.asarray(points_3d[order[1]], dtype=float)
        - np.asarray(points_3d[order[0]], dtype=float)
    )
    observed = _project_out(
        np.asarray(points_3d[order[2]], dtype=float)
        - np.asarray(points_3d[order[1]], dtype=float),
        femur_axis,
    )
    neutral_axis = _unit_vector(neutral_reference.get("femur_axis"))
    neutral_radial = _unit_vector(neutral_reference.get("radial_reference"))
    rotation = _rotation_aligning_vectors(neutral_axis, femur_axis)
    if (
        femur_axis is None
        or observed is None
        or neutral_axis is None
        or neutral_radial is None
        or rotation is None
    ):
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": [],
            "note": "Side-view 3-D hip rotation has an unstable femur or lower-leg axis in this frame.",
        }

    # Parallel-transport the neutral lower-leg direction with the femur. This
    # removes whole-leg repositioning while retaining rotation about the
    # femur, without inventing a torso centreline from one shoulder and hip.
    transported_reference = rotation @ neutral_radial
    reference = _project_out(transported_reference, femur_axis)
    signed = _signed_angle_3d(reference, observed, femur_axis)
    return {
        "angle": abs(signed) if signed is not None else None,
        "signed_angle": signed,
        "dimension": "3d",
        "missing_depth": [],
        "reference_frame_index": neutral_reference.get("frame_index"),
        "note": "Side-view 3-D hip axial-rotation excursion around the hip-to-knee femur axis, relative to the neutral starting frame.",
    }


def _hip_abduction_neutral_angle(
    points_3d: dict[str, np.ndarray],
    order: list[str] | tuple[str, ...],
    neutral_reference: dict | None,
) -> dict:
    """Measure right-thigh excursion from a selected neutral frame.

    This intentionally uses only the tested right hip and knee. It is a
    neutral-referenced 3-D estimate for the paired-view setup, which prevents
    an unreliable far-side shoulder or hip from defining the side-view zero.
    """
    if len(order) != 2:
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": [],
            "note": "Hip abduction requires right hip and right knee MMPose points.",
        }
    missing = [name for name in order if name not in points_3d]
    if missing:
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": missing,
            "note": "Hip abduction requires valid depth for the right hip and right knee.",
        }
    if not isinstance(neutral_reference, dict) or not neutral_reference.get("available"):
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": ["neutral reference"],
            "note": "Hip abduction requires a valid selected neutral-frame reference.",
        }

    neutral_thigh = _unit_vector(neutral_reference.get("thigh_reference"))
    current_thigh = _unit_vector(
        np.asarray(points_3d[order[1]], dtype=float)
        - np.asarray(points_3d[order[0]], dtype=float)
    )
    if neutral_thigh is None or current_thigh is None:
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": [],
            "note": "Hip abduction has an unstable neutral or current right thigh axis.",
        }
    cosine = max(-1.0, min(1.0, float(np.dot(neutral_thigh, current_thigh))))
    angle = math.degrees(math.acos(cosine))
    return {
        "angle": angle,
        "signed_angle": None,
        "dimension": "3d",
        "missing_depth": [],
        "reference_frame_index": neutral_reference.get("frame_index"),
        "note": "3-D right-thigh excursion from the selected neutral frame; the neutral frame is treated as 0°.",
    }


def _ankle_sagittal_angle(
    points_3d: dict[str, np.ndarray],
    order: list[str] | tuple[str, ...],
    direction: str | None = None,
) -> dict:
    """Measure ankle sagittal ROM from tibia and foot axes.

    Brett supplies three anonymous points (knee, ankle, and foot direction).
    For MMPose, the foot direction is stabilized with the midpoint of the
    big/small toes and the heel-to-toe axis. The tibia-foot included angle is
    approximately 90 degrees at neutral; report the excursion from that
    neutral position so neutral is 0 degrees for both task directions.
    """
    semantic_required = (
        "right_knee",
        "right_ankle",
        "right_big_toe",
        "right_small_toe",
        "right_heel",
    )
    semantic_mode = any(name in points_3d for name in semantic_required)
    if semantic_mode:
        missing = [name for name in semantic_required if name not in points_3d]
        if missing:
            return {
                "angle": None,
                "signed_angle": None,
                "dimension": "3d",
                "missing_depth": missing,
                "note": "3-D ankle ROM requires valid depth for the knee, ankle, heel, and both toe landmarks.",
            }
        tibia = (
            np.asarray(points_3d["right_knee"], dtype=float)
            - np.asarray(points_3d["right_ankle"], dtype=float)
        )
        toe_midpoint = (
            np.asarray(points_3d["right_big_toe"], dtype=float)
            + np.asarray(points_3d["right_small_toe"], dtype=float)
        ) / 2.0
        foot = toe_midpoint - np.asarray(points_3d["right_heel"], dtype=float)
    else:
        if len(order) != 3:
            return {
                "angle": None,
                "signed_angle": None,
                "dimension": "3d",
                "missing_depth": [],
                "note": "Ankle ROM requires exactly three annotation points.",
            }
        missing = [name for name in order if name not in points_3d]
        if missing:
            return {
                "angle": None,
                "signed_angle": None,
                "dimension": "3d",
                "missing_depth": missing,
                "note": "3-D ankle ROM requires valid depth for all three annotation points.",
            }
        tibia = (
            np.asarray(points_3d[order[0]], dtype=float)
            - np.asarray(points_3d[order[1]], dtype=float)
        )
        foot = (
            np.asarray(points_3d[order[2]], dtype=float)
            - np.asarray(points_3d[order[1]], dtype=float)
        )

    tibia = _unit_vector(tibia)
    foot = _unit_vector(foot)
    if tibia is None or foot is None:
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": [],
            "note": "Ankle ROM has an unstable tibia or foot axis.",
        }
    cosine = max(-1.0, min(1.0, float(np.dot(tibia, foot))))
    included = math.degrees(math.acos(cosine))
    return {
        "angle": abs(90.0 - included),
        "signed_angle": None,
        "dimension": "3d",
        "missing_depth": [],
        "included_angle_deg": included,
        "note": (
            f"3-D ankle {direction or 'sagittal'} ROM from the tibia and foot axes; "
            "the 90° neutral tibia-foot position is reported as 0°."
        ),
    }


def _torso_axes(points: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
    required = ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
    if any(name not in points for name in required):
        return None
    if any(
        (
            np.asarray(points[name], dtype=float).reshape(-1).size != 3
            or not np.isfinite(np.asarray(points[name], dtype=float)).all()
        )
        for name in required
    ):
        return None
    mid_hip = (points["left_hip"] + points["right_hip"]) / 2.0
    mid_shoulder = (
        points["left_shoulder"] + points["right_shoulder"]
    ) / 2.0
    down = _unit_vector(mid_hip - mid_shoulder)
    lateral = _unit_vector(
        (points["right_shoulder"] - points["left_shoulder"])
        + (points["right_hip"] - points["left_hip"])
    )
    if down is None or lateral is None:
        return None
    lateral = _unit_vector(_project_out(lateral, down))
    if lateral is None:
        return None
    normal = _unit_vector(np.cross(lateral, down))
    if normal is None:
        return None
    # The depth camera convention used by the imported data has +Z away from
    # the camera. In the front view the subject's anterior direction therefore
    # points toward -Z. This only fixes the sign; unsigned angles are unchanged.
    if normal[2] > 0:
        normal = -normal
    return {
        "down": down,
        "lateral": lateral,
        "normal": normal,
        "mid_hip": mid_hip,
        "mid_shoulder": mid_shoulder,
    }


def _task_angle(
    task: dict,
    points_2d: dict[str, dict[str, float]],
    lifted: dict,
) -> dict:
    method = task.get("angle_method", "included")
    points_3d = lifted.get("points", {})
    order = list(task.get("angle_points", []))

    if method == "side_trunk":
        vertex_name = task.get("angle_vertex") or task.get("vertex")
        segment = "humerus" if str(vertex_name).endswith("shoulder") else "thigh"
        missing_depth = [name for name in order if name not in points_3d]
        if not missing_depth:
            # A side-view goniometer angle is the included angle at the joint,
            # but it must be measured in the reconstructed camera-space point
            # cloud so out-of-plane arm motion is not silently discarded.
            angle = _angle_3d_order(points_3d, order)
            if (
                str(vertex_name).endswith("hip")
                or task.get("joint") == "hip"
            ) and angle is not None:
                # At the hip, the trunk and thigh point in opposite directions
                # from the joint when standing neutral. Convert that included
                # angle to the goniometer-style excursion from neutral.
                angle = 180.0 - angle
            return {
                "angle": angle,
                "signed_angle": None,
                "dimension": "3d",
                "missing_depth": [],
                "note": f"3-D point-cloud goniometer angle from the trunk line to the {segment}, with neutral at 0°.",
            }
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": missing_depth,
            "note": f"3-D point-cloud angle requires valid depth for the trunk and {segment} landmarks.",
        }

    if method == "ankle_sagittal":
        return _ankle_sagittal_angle(
            points_3d,
            order,
            task.get("direction"),
        )

    if method == "front_hip_abduction_manual":
        missing_depth = [name for name in order if name not in points_3d]
        if missing_depth:
            return {
                "angle": None,
                "signed_angle": None,
                "dimension": "3d",
                "missing_depth": missing_depth,
                "note": "Front-view 3-D hip-abduction angle requires valid depth for Brett's A-B-C points.",
            }
        angle = _angle_3d_order(points_3d, order)
        return {
            "angle": angle,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": [],
            "note": "Front-view 3-D hip-abduction angle at B from Brett's pelvic reference line to the right thigh.",
        }

    if method == "front_hip_abduction":
        torso_required = (
            "left_hip",
            "right_hip",
            "left_shoulder",
            "right_shoulder",
        )
        missing_depth = list(dict.fromkeys(
            [name for name in order if name not in points_3d]
            + [name for name in torso_required if name not in points_3d]
        ))
        if len(order) != 2 or missing_depth:
            return {
                "angle": None,
                "signed_angle": None,
                "dimension": "3d",
                "missing_depth": missing_depth,
                "note": "Front-view hip-abduction MMPose angle requires the right hip, right knee, and bilateral torso landmarks.",
            }
        axes = _torso_axes(points_3d)
        if axes is None:
            return {
                "angle": None,
                "signed_angle": None,
                "dimension": "3d",
                "missing_depth": list(torso_required),
                "note": "Front-view hip-abduction MMPose angle could not form a stable pelvic reference plane.",
            }
        thigh = points_3d[order[1]] - points_3d[order[0]]
        projected_thigh = _project_out(thigh, axes["normal"])
        projected_down = _project_out(axes["down"], axes["normal"])
        signed = _signed_angle_3d(
            projected_down,
            projected_thigh,
            axes["normal"],
        )
        return {
            "angle": abs(signed) if signed is not None else None,
            "signed_angle": signed,
            "dimension": "3d",
            "missing_depth": [],
            "note": "Front-view 3-D hip-abduction angle from the downward midpoint-shoulder-to-midpoint-hip pelvic reference to the right thigh, projected into the torso frontal plane.",
        }

    if method == "front_abduction_manual":
        # Brett's points are deliberately anonymous. A-B is the clinician's
        # chosen downward reference and B-C is the humerus. Both vectors are
        # projected into the frontal plane defined by the semantic MMPose
        # torso landmarks, so clicking a reference point slightly off the
        # visible body does not introduce an anterior/posterior component.
        reference_points = lifted.get("reference_points_3d", {})
        missing_depth = [name for name in order if name not in points_3d]
        torso_required = (
            "left_hip",
            "right_hip",
            "left_shoulder",
            "right_shoulder",
        )
        missing_reference = [
            name for name in torso_required if name not in reference_points
        ]
        if missing_depth or missing_reference:
            return {
                "angle": None,
                "signed_angle": None,
                "dimension": "3d",
                "missing_depth": list(dict.fromkeys(missing_depth + missing_reference)),
                "note": "Front-view 3-D shoulder-abduction angle requires valid depth for Brett's points and the MMPose torso plane.",
            }
        axes = _torso_axes(reference_points)
        if axes is None:
            return {
                "angle": None,
                "signed_angle": None,
                "dimension": "3d",
                "missing_depth": list(torso_required),
                "note": "Front-view 3-D shoulder-abduction angle could not form a stable torso plane.",
            }
        reference = points_3d[order[0]] - points_3d[order[1]]
        arm = points_3d[order[2]] - points_3d[order[1]]
        projected_reference = _project_out(reference, axes["normal"])
        projected_arm = _project_out(arm, axes["normal"])
        signed = _signed_angle_3d(
            projected_reference,
            projected_arm,
            axes["normal"],
        )
        return {
            "angle": abs(signed) if signed is not None else None,
            "signed_angle": signed,
            "dimension": "3d",
            "missing_depth": [],
            "note": "Front-view 3-D shoulder-abduction angle after projecting Brett's reference and humerus vectors into the torso frontal plane.",
        }

    if method in {"front_trunk", "front_pelvis"}:
        axes = _torso_axes(points_3d)
        vertex_name = task.get("angle_vertex") or task.get("vertex")
        side = str(vertex_name or "right_shoulder").split("_", 1)[0]
        limb_name = f"{side}_elbow" if method == "front_trunk" else f"{side}_knee"
        if axes is not None and limb_name in points_3d and vertex_name in points_3d:
            limb = points_3d[limb_name] - points_3d[vertex_name]
            projected_limb = _project_out(limb, axes["normal"])
            projected_down = _project_out(axes["down"], axes["normal"])
            signed = _signed_angle_3d(
                projected_down,
                projected_limb,
                axes["normal"],
            )
            return {
                "angle": abs(signed) if signed is not None else None,
                "signed_angle": signed,
                "dimension": "3d",
                "missing_depth": [],
                "note": "Front-view goniometer angle after removing the anterior/posterior component.",
            }
        if task.get("requires_depth"):
            required_depth = [
                "left_hip",
                "right_hip",
                "left_shoulder",
                "right_shoulder",
                limb_name,
                vertex_name,
            ]
            return {
                "angle": None,
                "signed_angle": None,
                "dimension": "3d",
                "missing_depth": list(dict.fromkeys(
                    name for name in required_depth if name not in points_3d
                )),
                "note": "Front-view 3-D angle requires valid depth for the torso plane and the target limb.",
            }
        return {
            "angle": _angle_deg(points_2d, task),
            "signed_angle": None,
            "dimension": "2d",
            "missing_depth": [],
            "note": "2-D front-view fallback; depth was unavailable for the anatomical plane projection.",
        }

    if method in {"included", "included_complement"}:
        angle = _angle_3d_order(points_3d, order)
        dimension = "3d"
        if angle is None:
            if task.get("requires_depth"):
                return {
                    "angle": None,
                    "signed_angle": None,
                    "dimension": "3d",
                    "missing_depth": [name for name in order if name not in points_3d],
                    "note": "3-D included angle requires valid depth for all three landmarks.",
                }
            angle = _angle_deg(points_2d, task)
            dimension = "2d"
        if method == "included_complement" and angle is not None:
            angle = 180.0 - angle
        return {
            "angle": angle,
            "signed_angle": None,
            "dimension": dimension,
            "missing_depth": [],
            "note": "Included joint angle at the middle landmark.",
        }

    if method == "shoulder_axial_manual":
        # A-B defines the humerus axis; B-C is the forearm. The torso normal
        # supplies the subject-relative chest/anterior reference. Translating
        # that reference to the elbow is implicit because these are free
        # vectors; only their directions matter.
        reference_points = lifted.get("reference_points_3d", {})
        if len(order) != 3:
            return {
                "angle": None,
                "signed_angle": None,
                "dimension": "3d",
                "missing_depth": [],
                "note": "Shoulder axial rotation requires exactly three annotation points.",
            }
        missing_depth = [name for name in order if name not in points_3d]
        torso_required = (
            "left_hip",
            "right_hip",
            "left_shoulder",
            "right_shoulder",
        )
        missing_reference = [
            name for name in torso_required if name not in reference_points
        ]
        if missing_depth or missing_reference:
            return {
                "angle": None,
                "signed_angle": None,
                "dimension": "3d",
                "missing_depth": list(dict.fromkeys(missing_depth + missing_reference)),
                "note": "Shoulder axial rotation requires valid depth for A-B-C and the MMPose torso reference.",
            }
        axes = _torso_axes(reference_points)
        if axes is None:
            return {
                "angle": None,
                "signed_angle": None,
                "dimension": "3d",
                "missing_depth": list(torso_required),
                "note": "Shoulder axial rotation could not form a stable torso reference.",
            }
        humerus = points_3d[order[1]] - points_3d[order[0]]
        forearm = points_3d[order[2]] - points_3d[order[1]]
        humerus_axis = _unit_vector(humerus)
        reference = _project_out(axes["normal"], humerus_axis)
        observed = _project_out(forearm, humerus_axis)
        signed = _signed_angle_3d(reference, observed, humerus_axis)
        return {
            "angle": abs(signed) if signed is not None else None,
            "signed_angle": signed,
            "dimension": "3d",
            "missing_depth": [],
            "note": "3-D shoulder axial-rotation angle around the humerus, relative to the torso/chest reference direction.",
        }

    if method == "hip_abduction_neutral":
        return _hip_abduction_neutral_angle(
            points_3d,
            order,
            lifted.get("hip_abduction_reference"),
        )

    if method in {"hip_axial_manual", "hip_axial_side"}:
        # The side-only hip task uses the same geometry for Brett and MMPose:
        # hip-to-knee is the femur axis and knee-to-ankle is the distal-leg
        # direction. Its reference is built from the neutral side frame, so a
        # one-sided shoulder/hip line is never mistaken for the body centreline.
        return _hip_axial_angle(
            points_3d,
            order,
            lifted.get("hip_axial_reference"),
        )

    if method in {"shoulder_axial", "hip_axial"}:
        required_points = task.get("required_points", task.get("points", []))
        missing_depth = [
            name for name in required_points if name not in points_3d
        ]
        axes = _torso_axes(points_3d)
        if method == "shoulder_axial":
            proximal_name = "right_shoulder"
            joint_name = "right_elbow"
            distal_name = "right_wrist"
        else:
            proximal_name = "right_hip"
            joint_name = "right_knee"
            distal_name = "right_ankle"
        if axes is None:
            missing_depth.extend(
                name
                for name in ("left_hip", "right_hip", "left_shoulder", "right_shoulder")
                if name not in missing_depth and name not in points_3d
            )
        if not missing_depth and axes is not None:
            long_axis = _unit_vector(points_3d[joint_name] - points_3d[proximal_name])
            distal_axis = points_3d[distal_name] - points_3d[joint_name]
            reference = _project_out(axes["normal"], long_axis)
            observed = _project_out(distal_axis, long_axis)
            signed = _signed_angle_3d(reference, observed, long_axis)
            return {
                "angle": abs(signed) if signed is not None else None,
                "signed_angle": signed,
                "dimension": "3d",
                "missing_depth": [],
                "note": "Axial orientation around the long axis relative to the torso anterior direction; use a standardized rotation posture.",
            }
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": sorted(set(missing_depth)),
            "note": "Axial rotation requires valid depth for the torso, proximal segment, and distal segment.",
        }

    return {
        "angle": _angle_deg(points_2d, task),
        "signed_angle": None,
        "dimension": "2d",
        "missing_depth": [],
        "note": "2-D included angle fallback.",
    }


def _front_shoulder_reference_points(task: dict) -> list[str]:
    """Return the 3-D landmarks needed for a front-view shoulder angle.

    A front-view flexion reference needs both sides of the torso to establish
    the anatomical lateral axis. The arm itself is represented by the same
    side's shoulder and elbow as the side-view task.
    """
    vertex = str(
        task.get("mmpose_angle_vertex")
        or task.get("angle_vertex")
        or task.get("vertex")
        or ""
    )
    if vertex not in {"left_shoulder", "right_shoulder"}:
        return []
    side = vertex.split("_", 1)[0]
    return [
        "left_hip",
        "right_hip",
        "left_shoulder",
        "right_shoulder",
        f"{side}_elbow",
    ]


def _front_sagittal_shoulder_angle(
    task: dict,
    points_3d: dict[str, np.ndarray],
) -> dict:
    """Measure shoulder flexion in the front-view torso sagittal plane.

    The torso landmarks define three orthogonal anatomical axes: down, left /
    right, and anterior / posterior. Projecting the shoulder-to-elbow vector
    out of the lateral axis removes abduction, leaving the sagittal component
    used for the front-view flexion reference.
    """
    required = _front_shoulder_reference_points(task)
    if not required:
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": [],
            "note": "No front-view shoulder reference is defined for this task.",
        }

    missing = [name for name in required if name not in points_3d]
    if missing:
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": missing,
            "note": "Front-view 3-D shoulder angle requires valid depth for the torso and arm landmarks.",
        }

    axes = _torso_axes(points_3d)
    if axes is None:
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": [],
            "note": "Front-view 3-D shoulder angle could not form a stable torso plane.",
        }

    shoulder_name = str(
        task.get("mmpose_angle_vertex")
        or task.get("angle_vertex")
        or task.get("vertex")
    )
    side = shoulder_name.split("_", 1)[0]
    elbow_name = f"{side}_elbow"
    arm = points_3d[elbow_name] - points_3d[shoulder_name]
    # ``lateral`` is the irrelevant left/right direction for flexion. The
    # remaining vector lies in the sagittal plane spanned by down and normal.
    projected_arm = _project_out(arm, axes["lateral"])
    projected_down = _project_out(axes["down"], axes["lateral"])
    signed = _signed_angle_3d(
        projected_down,
        projected_arm,
        axes["lateral"],
    )
    if signed is None:
        return {
            "angle": None,
            "signed_angle": None,
            "dimension": "3d",
            "missing_depth": [],
            "note": "Front-view 3-D shoulder angle could not measure a sagittal arm component.",
        }
    return {
        "angle": abs(signed),
        "signed_angle": signed,
        "dimension": "3d",
        "missing_depth": [],
        "note": "Front-view 3-D shoulder angle after removing the left/right arm component; reference is the downward torso axis in the sagittal plane.",
    }


def _mmpose_comparison(task: dict, pair: dict, video: dict, frame: int, manual_angle):
    """Calculate the same task angle from MMPose points on the same frame."""
    entry = _view_entry(pair, task.get("view", "side"))
    description = entry.get("mmpose")
    model_required = list(
        task.get("mmpose_points")
        or task.get("required_points", [])
    )
    model_task = dict(task)
    if task.get("mmpose_angle_method") == "front_hip_abduction":
        model_required = list(dict.fromkeys(
            model_required
            + [
                "left_hip",
                "right_hip",
                "left_shoulder",
                "right_shoulder",
            ]
        ))
    model_task["required_points"] = model_required
    # The front hip-abduction calculation needs the bilateral torso landmarks
    # in addition to the right hip/knee angle points. Keep those extra points
    # target-view-only so the side reference does not depend on its occluded
    # contralateral landmarks.
    pose_task = dict(task)
    pose_task["mmpose_points"] = model_required
    model = _mmpose_points(entry, pose_task, video, frame)
    model_points = model.get("points", {})
    lifted = _lift_points(model_points, video, frame)
    if task.get("mmpose_angle_method"):
        model_task["angle_method"] = task["mmpose_angle_method"]
    if task.get("mmpose_angle_points"):
        model_task["angle_points"] = list(task["mmpose_angle_points"])
    if task.get("mmpose_angle_vertex"):
        model_task["angle_vertex"] = task["mmpose_angle_vertex"]
    neutral_reference = None
    if model_task.get("angle_method") == "hip_abduction_neutral":
        neutral_reference = _hip_abduction_neutral_reference(
            task,
            pair,
            task.get("view", "side"),
        )
        lifted["hip_abduction_reference"] = neutral_reference
    elif model_task.get("angle_method") == "hip_axial_side":
        neutral_reference = _side_hip_neutral_reference(task, pair)
        lifted["hip_axial_reference"] = neutral_reference
    angle_data = _task_angle(model_task, model_points, lifted)
    missing_depth = angle_data.get("missing_depth", [])
    model_angle = _number_or_none(angle_data.get("angle"))
    model_valid = (
        bool(model.get("available"))
        and bool(model.get("frame_available"))
        and not model.get("missing_points")
        and not missing_depth
        and model_angle is not None
    )
    error = None
    if manual_angle is not None and model_valid:
        error = abs(float(manual_angle) - float(model_angle))
    return {
        "available": bool(model.get("available")),
        "frame_available": bool(model.get("frame_available")),
        "frame_id": model.get("frame_id"),
        "model": description.get("model") if isinstance(description, dict) else None,
        "reason": model.get("reason", ""),
        "points": model_points,
        "scores": model.get("scores", {}),
        "missing_points": model.get("missing_points", []),
        "relevant_points": model_required,
        "angle_deg": model_angle,
        "signed_angle_deg": _number_or_none(angle_data.get("signed_angle")),
        "angle_dimension": angle_data.get("dimension"),
        "angle_method": model_task.get("angle_method", "included"),
        "angle_points": list(model_task.get("angle_points", [])),
        "angle_vertex": model_task.get("angle_vertex"),
        "calculation_note": angle_data.get("note", ""),
        "points_3d": {
            name: _serialise_point_3d(lifted["points"].get(name))
            for name in model_required
        },
        "depth_values_m": {
            name: _number_or_none(lifted["depths"].get(name))
            for name in model_required
            if name in lifted.get("depths", {})
        },
        "missing_depth_points": missing_depth,
        "neutral_frame_index": (
            neutral_reference.get("frame_index")
            if neutral_reference is not None
            else None
        ),
        "neutral_mmpose_frame_id": (
            neutral_reference.get("frame_id")
            if neutral_reference is not None
            else None
        ),
        "neutral_reference_available": (
            bool(neutral_reference and neutral_reference.get("available"))
            if neutral_reference is not None
            else None
        ),
        "valid": model_valid,
        "error_deg": _number_or_none(error),
    }


def _reference_mmpose(task: dict, pair: dict) -> dict:
    """Calculate MMPose data for the non-target fixed reference view.

    For side-view shoulder tasks this also calculates the independent
    front-view sagittal-plane angle from the saved right/front frame.
    """
    target_view = _task_view(task.get("view", "side"))
    reference_view = "front" if target_view == "side" else "side"
    reference_entry = _view_entry(pair, reference_view)
    reference_frame = _view_frame(task, reference_view)

    reference_angle_method = task.get("reference_mmpose_angle_method")
    reference_angle_points = list(
        task.get("reference_mmpose_angle_points") or []
    )
    reference_angle_vertex = task.get("reference_mmpose_angle_vertex")
    required = list(
        task.get("reference_mmpose_points")
        or task.get("mmpose_points")
        or task.get("required_points", [])
    )
    front_reference_points = (
        _front_shoulder_reference_points(task)
        if reference_view == "front"
        else []
    )
    required = list(dict.fromkeys(required + front_reference_points))
    if reference_view == "front" and reference_angle_method == "front_hip_abduction":
        required = list(dict.fromkeys(required + [
            "left_hip",
            "right_hip",
            "left_shoulder",
            "right_shoulder",
            "right_knee",
        ]))
    model = _mmpose_points(
        reference_entry,
        {"required_points": required},
        reference_entry,
        reference_frame,
    )
    model_points = model.get("points", {})
    lifted = _lift_points(model_points, reference_entry, reference_frame)
    points_3d = lifted.get("points", {})
    description = reference_entry.get("mmpose")

    result = {
        **model,
        "view": reference_view,
        "frame_index": reference_frame,
        "model": (
            description.get("model")
            if isinstance(description, dict)
            else None
        ),
        "relevant_points": required,
        "points_3d": {
            name: _serialise_point_3d(points_3d.get(name))
            for name in required
        },
        "depth_values_m": {
            name: _number_or_none(lifted.get("depths", {}).get(name))
            for name in required
            if name in lifted.get("depths", {})
        },
        "missing_depth_points": [],
        "angle_deg": None,
        "signed_angle_deg": None,
        "angle_dimension": None,
        "angle_method": (
            reference_angle_method
            or task.get("mmpose_angle_method")
            or task.get("angle_method")
        ),
        "angle_points": list(
            reference_angle_points
            or task.get("mmpose_angle_points")
            or task.get("angle_points", [])
        ),
        "angle_vertex": (
            reference_angle_vertex
            or task.get("mmpose_angle_vertex")
            or task.get("angle_vertex")
            or task.get("vertex")
        ),
        "calculation_note": "",
        "valid": False,
    }

    if not task.get("compare_reference_view", True):
        result["calculation_note"] = (
            "Reference view only; MMPose angle is intentionally calculated "
            "from the side view for this task."
        )
        return result

    if front_reference_points:
        angle_data = _front_sagittal_shoulder_angle(task, points_3d)
        missing_depth = list(angle_data.get("missing_depth", []))
        model_angle = _number_or_none(angle_data.get("angle"))
        model_valid = (
            bool(model.get("available"))
            and bool(model.get("frame_available"))
            and not model.get("missing_points")
            and not missing_depth
            and model_angle is not None
        )
        result.update({
            "angle_deg": model_angle,
            "signed_angle_deg": _number_or_none(angle_data.get("signed_angle")),
            "angle_dimension": angle_data.get("dimension"),
            "angle_method": "front_sagittal_shoulder",
            "angle_points": list(
                task.get("mmpose_angle_points")
                or task.get("angle_points", [])
            ),
            "angle_vertex": (
                task.get("mmpose_angle_vertex")
                or task.get("angle_vertex")
                or task.get("vertex")
            ),
            "calculation_note": angle_data.get("note", ""),
            "missing_depth_points": missing_depth,
            "valid": model_valid,
        })
    elif reference_angle_method:
        # A side-target task can retain an independent anatomical computation
        # for its front-view reference. For right hip abduction this is the
        # bilateral-pelvic front-plane calculation.
        model_task = dict(task)
        model_task["required_points"] = required
        model_task["angle_method"] = reference_angle_method
        model_task["angle_points"] = reference_angle_points or list(
            task.get("mmpose_angle_points") or task.get("angle_points", [])
        )
        model_task["angle_vertex"] = (
            reference_angle_vertex
            or task.get("mmpose_angle_vertex")
            or task.get("angle_vertex")
            or task.get("vertex")
        )
        angle_data = _task_angle(model_task, model_points, lifted)
        missing_depth = list(angle_data.get("missing_depth", []))
        model_angle = _number_or_none(angle_data.get("angle"))
        model_valid = (
            bool(model.get("available"))
            and bool(model.get("frame_available"))
            and not model.get("missing_points")
            and not missing_depth
            and model_angle is not None
        )
        result.update({
            "angle_deg": model_angle,
            "signed_angle_deg": _number_or_none(angle_data.get("signed_angle")),
            "angle_dimension": angle_data.get("dimension"),
            "angle_method": model_task["angle_method"],
            "angle_points": list(model_task["angle_points"]),
            "angle_vertex": model_task["angle_vertex"],
            "calculation_note": angle_data.get("note", ""),
            "missing_depth_points": missing_depth,
            "valid": model_valid,
        })
    elif task.get("mmpose_angle_method") == "front_hip_abduction":
        model_required = list(
            task.get("mmpose_points")
            or task.get("required_points", [])
        )
        model_required = list(dict.fromkeys(model_required + [
            "right_shoulder",
            "right_hip",
            "right_knee",
        ]))
        model_task = dict(task)
        model_task["required_points"] = model_required
        # The target is the front view, where the bilateral pelvic reference
        # is available. For the non-target side view, the controlled supine
        # protocol permits the direct shoulder-hip-knee trunk/thigh angle;
        # unlike the front calculation, it does not need a neutral frame.
        model_task["angle_method"] = "side_trunk"
        model_task["angle_points"] = [
            "right_shoulder",
            "right_hip",
            "right_knee",
        ]
        model_task["angle_vertex"] = "right_hip"
        angle_data = _task_angle(model_task, model_points, lifted)
        missing_depth = list(angle_data.get("missing_depth", []))
        model_angle = _number_or_none(angle_data.get("angle"))
        model_valid = (
            bool(model.get("available"))
            and bool(model.get("frame_available"))
            and not model.get("missing_points")
            and not missing_depth
            and model_angle is not None
        )
        result.update({
            "angle_deg": model_angle,
            "signed_angle_deg": _number_or_none(angle_data.get("signed_angle")),
            "angle_dimension": angle_data.get("dimension"),
            "angle_method": model_task["angle_method"],
            "angle_points": list(model_task["angle_points"]),
            "angle_vertex": model_task["angle_vertex"],
            "calculation_note": angle_data.get("note", ""),
            "missing_depth_points": missing_depth,
            "valid": model_valid,
        })
    elif target_view == "side":
        # Side-view tasks such as elbow flexion still expose the same
        # point-cloud calculation on the saved front frame for comparison.
        model_required = list(
            task.get("mmpose_points")
            or task.get("required_points", [])
        )
        model_task = dict(task)
        model_task["required_points"] = model_required
        if task.get("mmpose_angle_method"):
            model_task["angle_method"] = task["mmpose_angle_method"]
        if task.get("mmpose_angle_points"):
            model_task["angle_points"] = list(task["mmpose_angle_points"])
        if task.get("mmpose_angle_vertex"):
            model_task["angle_vertex"] = task["mmpose_angle_vertex"]
        angle_data = _task_angle(model_task, model_points, lifted)
        missing_depth = list(angle_data.get("missing_depth", []))
        model_angle = _number_or_none(angle_data.get("angle"))
        model_valid = (
            bool(model.get("available"))
            and bool(model.get("frame_available"))
            and not model.get("missing_points")
            and not missing_depth
            and model_angle is not None
        )
        result.update({
            "angle_deg": model_angle,
            "signed_angle_deg": _number_or_none(angle_data.get("signed_angle")),
            "angle_dimension": angle_data.get("dimension"),
            "angle_method": model_task.get("angle_method", "included"),
            "angle_points": list(model_task.get("angle_points", [])),
            "angle_vertex": model_task.get("angle_vertex"),
            "calculation_note": angle_data.get("note", ""),
            "missing_depth_points": missing_depth,
            "valid": model_valid,
        })

    return result


def _invalid_depth_points_from_mmpose(result: dict) -> list[str]:
    """Return MMPose landmarks that have 2-D points but no valid 3-D depth."""
    if not isinstance(result, dict):
        return []
    if not result.get("available") or not result.get("frame_available"):
        return []
    points = result.get("points") or {}
    points_3d = result.get("points_3d") or {}
    relevant = result.get("relevant_points") or []
    return [
        str(name)
        for name in relevant
        if name in points and points_3d.get(name) is None
    ]


def _task_mmpose_depth_warnings(task: dict, pair: dict) -> list[dict]:
    """Check selected task frames for MMPose points on invalid depth."""
    warnings = []

    def add_warning(side: str, view: str, frame: int, result: dict) -> None:
        invalid_points = _invalid_depth_points_from_mmpose(result)
        if not invalid_points:
            return
        warnings.append({
            "side": side,
            "view": view,
            "view_label": _view_label(view),
            "frame_index": int(frame),
            "points": invalid_points,
        })

    try:
        target_view = _task_view(task.get("view", "side"))
        target_side = VIEW_TO_SIDE[target_view]
        target_frame = _view_frame(task, target_view)
        target_result = _mmpose_comparison(
            task,
            pair,
            _view_entry(pair, target_view),
            target_frame,
            None,
        )
        add_warning(target_side, target_view, target_frame, target_result)

        if task.get("compare_reference_view", True):
            reference_result = _reference_mmpose(task, pair)
            reference_view = _task_view(reference_result.get("view", "side"))
            add_warning(
                VIEW_TO_SIDE[reference_view],
                reference_view,
                int(reference_result.get("frame_index", _view_frame(task, reference_view))),
                reference_result,
            )
    except (FileNotFoundError, OSError, TypeError, ValueError, KeyError, IndexError):
        # This is advisory metadata. A depth-read or MMPose-read failure must
        # never prevent the admin from saving the task.
        return warnings
    return warnings


def _public_admin_task(task: dict, pair: dict) -> dict:
    result = _public_task(task, pair)
    if "mmpose_depth_warnings" not in task:
        result["mmpose_depth_warnings"] = _task_mmpose_depth_warnings(task, pair)
    return result


def _serialise_point_3d(value) -> list[float] | None:
    if value is None:
        return None
    values = np.asarray(value, dtype=float).reshape(-1)
    if values.size != 3 or not np.isfinite(values).all():
        return None
    return [float(component) for component in values]


def _number_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _preview(task: dict, points: dict, pair: dict) -> dict:
    view = _task_view(task.get("view", "side"))
    video = _view_entry(pair, view)
    frame = _view_frame(task, view)
    missing = [name for name in task["required_points"] if name not in points]
    # Calculate the model first. Front-view abduction uses the model's torso
    # point cloud as the anatomical plane for Brett's otherwise anonymous
    # A-B-C points.
    mmpose = _mmpose_comparison(task, pair, video, frame, None)
    lifted = _lift_points(
        points,
        video,
        frame,
    )
    if task.get("angle_method") in {
        "front_abduction_manual",
        "shoulder_axial_manual",
    }:
        lifted["reference_points_3d"] = {
            name: np.asarray(value, dtype=float)
            for name, value in (
                (mmpose.get("points_3d") or {})
            ).items()
            if value is not None
        }
    if task.get("angle_method") == "hip_axial_manual":
        lifted["hip_axial_reference"] = _side_hip_neutral_reference(task, pair)
    angle_data = _task_angle(task, points, lifted)
    reference_mmpose = _reference_mmpose(task, pair)
    manual_angle = _number_or_none(angle_data["angle"])
    if manual_angle is not None and mmpose.get("valid"):
        mmpose["error_deg"] = _number_or_none(
            abs(float(manual_angle) - float(mmpose["angle_deg"]))
        )

    target_model = mmpose
    reference_model = reference_mmpose
    side_model = target_model if view == "side" else reference_model
    front_model = target_model if view == "front" else reference_model
    side_mmpose_angle = _number_or_none(side_model.get("angle_deg"))
    front_mmpose_angle = _number_or_none(front_model.get("angle_deg"))
    side_mmpose_error = (
        abs(float(manual_angle) - float(side_mmpose_angle))
        if manual_angle is not None and side_mmpose_angle is not None
        else None
    )
    front_mmpose_error = (
        abs(float(manual_angle) - float(front_mmpose_angle))
        if manual_angle is not None and front_mmpose_angle is not None
        else None
    )
    return {
        "task_id": task["task_id"],
        "view": view,
        "view_label": _view_label(view),
        "frame_index": frame,
        "angle_deg": _number_or_none(angle_data["angle"]),
        "signed_angle_deg": _number_or_none(angle_data["signed_angle"]),
        "angle_dimension": angle_data["dimension"],
        "angle_method": task.get("angle_method", "included"),
        "angle_description": task.get("angle_description", angle_data["note"]),
        "calculation_note": angle_data["note"],
        "reference_frame_index": angle_data.get("reference_frame_index"),
        "angle_points": task["angle_points"],
        "missing_points": missing,
        "missing_depth_points": angle_data["missing_depth"],
        "depth_format": lifted.get("format"),
        "depth_frame_available": lifted.get("frame_available", False),
        "points_3d": {
            name: _serialise_point_3d(lifted["points"].get(name))
            for name in task["required_points"]
        },
        "depth_pixels": {
            name: lifted["pixels"].get(name)
            for name in task["required_points"]
            if name in lifted["pixels"]
        },
        "depth_values_m": {
            name: _number_or_none(lifted["depths"].get(name))
            for name in task["required_points"]
            if name in lifted["depths"]
        },
        "mmpose": mmpose,
        "reference_mmpose": reference_mmpose,
        "side_mmpose_angle_deg": side_mmpose_angle,
        "front_mmpose_angle_deg": front_mmpose_angle,
        "side_mmpose_error_deg": _number_or_none(side_mmpose_error),
        "front_mmpose_error_deg": _number_or_none(front_mmpose_error),
        "valid": (
            not missing
            and not angle_data["missing_depth"]
            and angle_data["angle"] is not None
        ),
    }


# ---------- authentication ----------
def login_required(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        if not session.get("authed") or not session.get("user"):
            if (
                request.path.startswith("/api/")
                or request.path.startswith("/media/")
                or request.path.startswith("/depth/")
            ):
                return jsonify({"error": "unauthorized"}), 401
            return redirect("/")
        return function(*args, **kwargs)

    return wrapper


def role_required(role: str):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if not session.get("authed") or not session.get("user"):
                return login_required(function)(*args, **kwargs)
            if session.get("role") != role:
                return jsonify({"error": f"{role} access required"}), 403
            return function(*args, **kwargs)

        return wrapper

    return decorator


admin_required = role_required("admin")
annotator_required = role_required("annotator")


@app.get("/")
def index():
    if session.get("role") == "admin":
        return redirect("/admin")
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/admin")
@admin_required
def admin_page():
    return send_from_directory(STATIC_DIR, "admin.html")


@app.post("/api/login")
def login():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        username = _safe_username(payload.get("username"))
    except ValueError as error:
        return jsonify({"error": str(error)}), 401
    password = str(payload.get("password") or "")
    role = None
    if username == ADMIN_USERNAME and hmac.compare_digest(password, ADMIN_PASSWORD):
        role = "admin"
    elif username in ROM2_USERS and hmac.compare_digest(password, ROM2_USERS[username]):
        role = "annotator"
    elif username != ADMIN_USERNAME and hmac.compare_digest(password, ANNOTATOR_PASSWORD):
        role = "annotator"
    if role is None:
        return jsonify({"error": "wrong username/password"}), 401
    session["authed"] = True
    session["user"] = username
    session["role"] = role
    return jsonify({"ok": True, "user": username, "role": role})


@app.get("/api/me")
def me():
    return jsonify({
        "authenticated": bool(session.get("authed") and session.get("user")),
        "user": session.get("user"),
        "role": session.get("role"),
    })


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.get("/api/config")
@login_required
def config():
    return jsonify({
        "storage_dir": str(STORAGE_DIR),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "folder_import": True,
        "allowed_folder_roots": [str(root) for root in ALLOWED_FOLDER_ROOTS],
        "templates": [
            {
                "key": key,
                "label": value["label"],
                "points": value["points"],
                "view": value.get("view", "side"),
                "angle_method": value.get("angle_method", "included"),
                "vertex": value["vertex"],
                "annotation_labels": value.get("annotation_labels", []),
                "requires_depth": bool(value.get("requires_depth", False)),
                "neutral_frame_views": list(value.get("neutral_frame_views", [])),
                "description": value.get("description", ""),
            }
            for key, value in TASK_TEMPLATES.items()
            if key in ACTIVE_TEMPLATE_KEYS
        ],
    })


# ---------- media and admin endpoints ----------
@app.get("/media/<pair_id>/<side>")
@login_required
def media_file(pair_id: str, side: str):
    pair = _find_pair(pair_id)
    if pair is None:
        return jsonify({"error": "pair not found"}), 404
    try:
        path = _browser_video_path(pair, side)
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 404
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 503
    return send_file(path, conditional=True, max_age=0)


@app.get("/frame/<pair_id>/<side>/<int:frame_index>.jpg")
@login_required
def frame_image(pair_id: str, side: str, frame_index: int):
    """Serve one exact, zero-based RGB frame for a saved task."""
    pair = _find_pair(pair_id)
    if pair is None:
        return jsonify({"error": "pair not found"}), 404
    try:
        jpeg = _exact_frame_jpeg(pair, side, frame_index)
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 404
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    return send_file(
        BytesIO(jpeg),
        mimetype="image/jpeg",
        download_name=f"{pair_id}_{side}_{frame_index}.jpg",
        max_age=3600,
    )


@app.get("/depth/<pair_id>/<side>/<int:frame_index>.png")
@login_required
def depth_image(pair_id: str, side: str, frame_index: int):
    """Serve a colorized native-resolution depth layer for one RGB frame."""
    pair = _find_pair(pair_id)
    if pair is None:
        return jsonify({"error": "pair not found"}), 404
    side = str(side).strip().lower()
    if side not in {"left", "right"}:
        return jsonify({"error": "side must be left or right"}), 400
    entry = pair.get(side)
    if not isinstance(entry, dict):
        return jsonify({"error": f"{side} video is missing"}), 404
    try:
        frame = _frame_index(frame_index, entry, f"{side} frame")
        png = _depth_color_png(entry, frame)
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 404
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    return send_file(
        BytesIO(png),
        mimetype="image/png",
        download_name=f"{pair_id}_{side}_{frame}.png",
        max_age=3600,
    )


@app.get("/api/admin/pairs/<pair_id>/browser-video-status")
@admin_required
def admin_browser_video_status(pair_id: str):
    """Return the current H.264 proxy conversion status for one video."""
    pair = _find_pair(pair_id)
    if pair is None:
        return jsonify({"error": "pair not found"}), 404
    side = str(request.args.get("side", "left")).strip().lower()
    if side not in {"left", "right"}:
        return jsonify({"error": "side must be left or right"}), 400
    try:
        return jsonify({
            "pair_id": pair_id,
            "side": side,
            **_browser_proxy_status(pair, side),
        })
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/admin/pairs/<pair_id>/mmpose")
@admin_required
def admin_mmpose_frame(pair_id: str):
    """Return body pose points for the admin's currently selected frame."""
    pair = _find_pair(pair_id)
    if pair is None:
        return jsonify({"error": "pair not found"}), 404
    side = str(request.args.get("side", "left")).strip().lower()
    if side not in {"left", "right"}:
        return jsonify({"error": "side must be left or right"}), 400
    entry = pair.get(side)
    if not isinstance(entry, dict):
        return jsonify({"error": f"{side} video is missing"}), 404
    try:
        frame = _frame_index(request.args.get("frame", 0), entry, f"{side} frame")
        pose = _mmpose_points(
            entry,
            {"required_points": list(MMPose_DISPLAY_POINTS)},
            entry,
            frame,
        )
        return jsonify({
            "pair_id": pair_id,
            "side": side,
            "frame_index": frame,
            "frame_id": pose.get("frame_id"),
            "model": entry.get("mmpose", {}).get("model")
            if isinstance(entry.get("mmpose"), dict)
            else None,
            "available": pose.get("available", False),
            "frame_available": pose.get("frame_available", False),
            "points": pose.get("points", {}),
            "scores": pose.get("scores", {}),
            "missing_points": pose.get("missing_points", []),
            "reason": pose.get("reason", ""),
            "skeleton": [list(link) for link in MMPose_DISPLAY_SKELETON],
        })
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/admin/pairs/<pair_id>/mmpose-all")
@admin_required
def admin_mmpose_all(pair_id: str):
    """Return all MMPose display landmarks for one pair member."""
    pair = _find_pair(pair_id)
    if pair is None:
        return jsonify({"error": "pair not found"}), 404
    side = str(request.args.get("side", "left")).strip().lower()
    if side not in {"left", "right"}:
        return jsonify({"error": "side must be left or right"}), 400
    entry = pair.get(side)
    if not isinstance(entry, dict):
        return jsonify({"error": f"{side} video is missing"}), 404
    try:
        result = _mmpose_all_points(entry, entry)
        return jsonify({
            "pair_id": pair_id,
            "side": side,
            "model": entry.get("mmpose", {}).get("model")
            if isinstance(entry.get("mmpose"), dict)
            else None,
            **result,
        })
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/admin/pairs")
@admin_required
def admin_pairs():
    pairs = sorted(_load_pairs(), key=lambda item: item.get("created_at", 0), reverse=True)
    return jsonify({"pairs": [_public_pair(pair) for pair in pairs]})


@app.post("/api/admin/inspect-folder")
@admin_required
def inspect_folder():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        discovery = _discover_session_folder(payload.get("path"))
        return jsonify({"ok": True, "session": _public_discovery(discovery)})
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/admin/import-folder-pair")
@admin_required
def import_folder_pair():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        left_discovery = _discover_session_folder(payload.get("left_folder"))
        right_discovery = _discover_session_folder(payload.get("right_folder"))
        if left_discovery["video_path"] == right_discovery["video_path"]:
            raise ValueError("left and right folders resolve to the same RGB video")
        pair = _new_pair(
            _folder_entry(left_discovery),
            _folder_entry(right_discovery),
        )
        pairs = _load_pairs()
        pairs.append(pair)
        _save_pairs(pairs)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    return jsonify({"ok": True, "pair": _public_pair(pair)})


@app.post("/api/admin/upload-pair")
@admin_required
def upload_pair():
    left_file = request.files.get("left_video")
    right_file = request.files.get("right_video")
    if left_file is None or right_file is None:
        return jsonify({"error": "choose both a left and a right video"}), 400

    pair_id = f"pair_{int(time.time())}_{uuid.uuid4().hex[:10]}"
    pair_dir = UPLOADS_DIR / pair_id
    pair_dir.mkdir(parents=True, exist_ok=False)
    saved_paths = []
    try:
        entries = {}
        for side, file_storage in (("left", left_file), ("right", right_file)):
            original_name = str(file_storage.filename or "").strip()
            safe_name = secure_filename(original_name)
            extension = Path(safe_name).suffix.lower()
            if not safe_name or extension not in VIDEO_EXTENSIONS:
                raise ValueError(
                    f"{side} file must use a supported video extension "
                    f"({', '.join(sorted(VIDEO_EXTENSIONS))})"
                )
            destination = pair_dir / f"{side}{extension}"
            file_storage.save(destination)
            saved_paths.append(destination)
            metadata = _video_metadata(destination)
            entries[side] = {
                "filename": destination.name,
                "original_name": original_name,
                **metadata,
            }
        pair = {
            "pair_id": pair_id,
            "left": entries["left"],
            "right": entries["right"],
            "created_at": time.time(),
        }
        pairs = _load_pairs()
        pairs.append(pair)
        _save_pairs(pairs)
    except Exception as error:
        for path in saved_paths:
            path.unlink(missing_ok=True)
        pair_dir.rmdir()
        return jsonify({"error": str(error)}), 400
    return jsonify({"ok": True, "pair": _public_pair(pair)})


@app.get("/api/admin/tasks")
@admin_required
def admin_tasks():
    pair_id = request.args.get("pair_id")
    tasks = _load_tasks()
    if pair_id:
        pair_id = _safe_id(pair_id, "pair_id")
        tasks = [task for task in tasks if task.get("pair_id") == pair_id]
    public_tasks = []
    for task in sorted(tasks, key=lambda item: item.get("created_at", 0), reverse=True):
        pair = _find_pair(task["pair_id"])
        if pair is not None:
            public_tasks.append(_public_admin_task(task, pair))
    return jsonify({"tasks": public_tasks})


def _normalise_task(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("task must be an object")
    pair_id = _safe_id(payload.get("pair_id"), "pair_id")
    pair = _find_pair(pair_id)
    if pair is None:
        raise FileNotFoundError("pair not found")
    template_key = str(payload.get("template", "shoulder_flexion")).strip()
    if template_key in TASK_TEMPLATES:
        template = TASK_TEMPLATES[template_key]
        points = list(template["points"])
        vertex = template["vertex"]
        angle_points = list(template.get("angle_points", [points[0], vertex, points[2]]))
        view = _task_view(template.get("view", "side"))
        angle_method = str(template.get("angle_method", "included")).strip()
        annotation_labels = list(template.get("annotation_labels", []))
        direction = template.get("direction")
        joint = template.get("joint")
        description = template.get("description", "")
        requires_depth = bool(template.get("requires_depth", False))
        mmpose_points = list(template.get("mmpose_points", []))
        mmpose_angle_method = template.get("mmpose_angle_method")
        mmpose_angle_points = list(template.get("mmpose_angle_points", []))
        mmpose_angle_vertex = template.get("mmpose_angle_vertex")
        reference_mmpose_points = list(template.get("reference_mmpose_points", []))
        reference_mmpose_angle_method = template.get(
            "reference_mmpose_angle_method"
        )
        reference_mmpose_angle_points = list(
            template.get("reference_mmpose_angle_points", [])
        )
        reference_mmpose_angle_vertex = template.get(
            "reference_mmpose_angle_vertex"
        )
        geometry_version = template.get("geometry_version")
        annotation_instruction = template.get("annotation_instruction", "")
        neutral_frame_index = template.get("neutral_frame_index")
        neutral_frame_views = list(template.get("neutral_frame_views", []))
        compare_reference_view = bool(template.get("compare_reference_view", True))
        default_name = template["label"]
    elif template_key == "custom":
        raw_points = payload.get("required_points", payload.get("points", []))
        if isinstance(raw_points, str):
            raw_points = [part.strip() for part in raw_points.split(",")]
        if not isinstance(raw_points, list):
            raise ValueError("custom points must be a list or comma-separated string")
        points = []
        for raw_point in raw_points:
            point = str(raw_point).strip()
            if not _POINT_RE.fullmatch(point):
                raise ValueError(f"invalid point name: {point}")
            if point.lower() not in {item.lower() for item in points}:
                points.append(point)
        if len(points) != 3:
            raise ValueError("custom angles need exactly 3 unique points")
        vertex = str(payload.get("angle_vertex", points[1])).strip()
        if vertex.lower() not in {item.lower() for item in points}:
            raise ValueError("angle vertex must be one of the 3 points")
        vertex = next(item for item in points if item.lower() == vertex.lower())
        angle_points = [points[0], vertex, points[2]]
        view = _task_view(payload.get("view", "side"))
        angle_method = "included"
        annotation_labels = []
        direction = None
        joint = None
        description = "Included angle at the selected vertex."
        requires_depth = False
        mmpose_points = []
        mmpose_angle_method = None
        mmpose_angle_points = []
        mmpose_angle_vertex = None
        reference_mmpose_points = []
        reference_mmpose_angle_method = None
        reference_mmpose_angle_points = []
        reference_mmpose_angle_vertex = None
        geometry_version = None
        annotation_instruction = ""
        neutral_frame_index = None
        neutral_frame_views = []
        compare_reference_view = True
        default_name = "Custom angle"
    else:
        raise ValueError("unknown task template")

    task_name = _task_name(payload.get("name"), default_name)
    left_frame = _frame_index(
        payload.get("left_frame_index"), pair["left"], "left_frame_index"
    )
    right_frame = _frame_index(
        payload.get("right_frame_index"), pair["right"], "right_frame_index"
    )
    if neutral_frame_index is not None:
        neutral_frame_index = _frame_index(
            payload.get("neutral_frame_index", neutral_frame_index),
            pair["left"],
            "neutral_frame_index",
        )
    neutral_frame_indices = None
    if neutral_frame_views:
        raw_neutral_frames = payload.get("neutral_frame_indices")
        if not isinstance(raw_neutral_frames, dict):
            neutral_labels = [
                "the side video" if side == "left" else "the front video"
                for side in neutral_frame_views
            ]
            raise ValueError(
                "select a neutral frame for "
                + " and ".join(neutral_labels)
                + " before saving this task"
            )
        neutral_frame_indices = {}
        for side in ("left", "right"):
            if side not in neutral_frame_views:
                continue
            raw_frame = raw_neutral_frames.get(side)
            if raw_frame is None:
                raise ValueError(
                    f"select a neutral frame for the {side} video before saving this task"
                )
            neutral_frame = _frame_index(
                raw_frame,
                pair[side],
                f"neutral_{side}_frame_index",
            )
            if requires_depth:
                if not isinstance(pair[side].get("depth"), dict):
                    raise ValueError(
                        f"{side} video needs depth for its neutral reference"
                    )
                if not _depth_frame_available(pair[side], neutral_frame):
                    raise ValueError(
                        f"neutral {side} frame {neutral_frame} has no depth data; choose another frame"
                    )
            neutral_frame_indices[side] = neutral_frame
    target_entry = _view_entry(pair, view)
    target_frame = left_frame if view == "side" else right_frame
    target_has_depth = isinstance(target_entry.get("depth"), dict)
    target_depth_available = _depth_frame_available(target_entry, target_frame)
    mmpose_frame = _mmpose_frame_info(target_entry, target_frame)
    if requires_depth and not target_has_depth:
        raise ValueError(f"{_view_label(view)} requires depth for this 3-D angle")
    if requires_depth and not target_depth_available:
        raise ValueError(
            f"{_view_label(view)} frame {target_frame} has no depth data; choose a frame with depth"
        )
    if any(point not in points for point in angle_points):
        raise ValueError("angle points must be included in required points")
    task_id = str(payload.get("task_id") or f"task_{uuid.uuid4().hex}").strip()
    task_id = _safe_id(task_id, "task_id")
    notes = str(payload.get("notes", ""))
    if len(notes) > 2000:
        raise ValueError("notes are limited to 2000 characters")
    now = time.time()
    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "task_id": task_id,
        "pair_id": pair_id,
        "template": template_key,
        "name": task_name,
        "required_points": points,
        "annotation_labels": annotation_labels if len(annotation_labels) == len(points) else [],
        "angle_points": angle_points,
        "angle_vertex": vertex,
        "view": view,
        "view_label": _view_label(view),
        "angle_method": angle_method,
        "mmpose_points": mmpose_points,
        "mmpose_angle_method": mmpose_angle_method,
        "mmpose_angle_points": mmpose_angle_points,
        "mmpose_angle_vertex": mmpose_angle_vertex,
        "reference_mmpose_points": reference_mmpose_points,
        "reference_mmpose_angle_method": reference_mmpose_angle_method,
        "reference_mmpose_angle_points": reference_mmpose_angle_points,
        "reference_mmpose_angle_vertex": reference_mmpose_angle_vertex,
        "geometry_version": geometry_version,
        "annotation_instruction": annotation_instruction,
        "neutral_frame_index": neutral_frame_index,
        "neutral_frame_indices": neutral_frame_indices,
        "neutral_frame_views": neutral_frame_views,
        "compare_reference_view": compare_reference_view,
        "requires_depth": requires_depth,
        "direction": direction,
        "joint": joint,
        "angle_description": description,
        "left_frame_index": left_frame,
        "right_frame_index": right_frame,
        "left_timestamp_sec": left_frame / pair["left"]["frame_rate"],
        "right_timestamp_sec": right_frame / pair["right"]["frame_rate"],
        "target_frame_index": target_frame,
        "mmpose_available": mmpose_frame["available"],
        "mmpose_frame_available": mmpose_frame["frame_available"],
        "mmpose_frame_id": mmpose_frame["frame_id"],
        "angle_dimension": (
            "3d"
            if target_has_depth and target_depth_available
            else "2d"
        ),
        "enabled": bool(payload.get("enabled", True)),
        "notes": notes,
        "created_at": now,
        "updated_at": now,
    }


@app.post("/api/admin/tasks")
@admin_required
def create_admin_task():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        task = _normalise_task(payload)
        pair = _find_pair(task["pair_id"])
        task["mmpose_depth_warnings"] = _task_mmpose_depth_warnings(task, pair)
        tasks = _load_tasks()
        if any(item.get("task_id") == task["task_id"] for item in tasks):
            raise ValueError("task_id already exists")
        tasks.append(task)
        _save_tasks(tasks)
        return jsonify({"ok": True, "task": _public_admin_task(task, pair)})
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.delete("/api/admin/tasks/<task_id>")
@admin_required
def delete_admin_task(task_id):
    """Remove a task from the task index without deleting its annotation file."""
    try:
        task_id = _safe_id(task_id, "task_id")
        tasks = _load_tasks()
        if not any(item.get("task_id") == task_id for item in tasks):
            return jsonify({"error": "task not found"}), 404
        _save_tasks([item for item in tasks if item.get("task_id") != task_id])
        return jsonify({"ok": True, "task_id": task_id})
    except (OSError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


# ---------- annotator endpoints ----------
@app.get("/api/tasks")
@annotator_required
def annotator_tasks():
    result = []
    for task in sorted(_load_tasks(), key=lambda item: item.get("created_at", 0), reverse=True):
        if not task.get("enabled", True):
            continue
        pair = _find_pair(task.get("pair_id"))
        if pair is None:
            continue
        annotation = _load_annotation(session["user"], task["task_id"])
        result.append({
            **_public_task(task, pair, include_neutral_metadata=False),
            "status": annotation.get("status", "not_started") if annotation else "not_started",
            "angle_deg": annotation.get("angle_deg") if annotation else None,
            "mmpose_angle_deg": annotation.get("mmpose_angle_deg") if annotation else None,
            "mmpose_error_deg": annotation.get("mmpose_error_deg") if annotation else None,
            "side_mmpose_angle_deg": annotation.get("side_mmpose_angle_deg") if annotation else None,
            "side_mmpose_error_deg": annotation.get("side_mmpose_error_deg") if annotation else None,
            "front_mmpose_angle_deg": annotation.get("front_mmpose_angle_deg") if annotation else None,
            "front_mmpose_error_deg": annotation.get("front_mmpose_error_deg") if annotation else None,
            "updated_at": annotation.get("updated_at") if annotation else None,
        })
    return jsonify({"tasks": result})


def _annotator_task(task_id: str) -> tuple[dict, dict]:
    task = _find_task(task_id)
    if task is None or not task.get("enabled", True):
        raise FileNotFoundError("task not found")
    pair = _find_pair(task["pair_id"])
    if pair is None:
        raise FileNotFoundError("task pair not found")
    return task, pair


def _annotation_points_payload(payload: dict) -> dict:
    # ``left_points`` is accepted for tasks created by the first rom_measure2
    # version. New tasks use the view-neutral ``points`` field.
    return payload.get("points", payload.get("left_points", {}))


@app.get("/api/tasks/<task_id>")
@annotator_required
def get_annotator_task(task_id: str):
    try:
        task, pair = _annotator_task(task_id)
        annotation = _load_annotation(session["user"], task["task_id"])
        return jsonify({
            "task": _public_task(task, pair, include_neutral_metadata=False),
            "annotation": _redact_neutral_metadata(annotation),
            "reference_mmpose": _redact_neutral_metadata(
                _reference_mmpose(task, pair)
            ),
        })
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"error": str(error)}), 404


@app.post("/api/tasks/<task_id>/preview")
@annotator_required
def preview_annotator_task(task_id: str):
    payload = request.get_json(force=True, silent=True) or {}
    try:
        task, pair = _annotator_task(task_id)
        view = _task_view(task.get("view", "side"))
        points = _normalise_points(
            _annotation_points_payload(payload),
            task,
            _view_entry(pair, view),
        )
        return jsonify(_redact_neutral_metadata(_preview(task, points, pair)))
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/tasks/<task_id>/annotation")
@annotator_required
def save_annotator_annotation(task_id: str):
    payload = request.get_json(force=True, silent=True) or {}
    try:
        task, pair = _annotator_task(task_id)
        existing = _load_annotation(session["user"], task["task_id"])
        action = str(payload.get("action", "submit")).strip().lower()
        if action not in {"draft", "submit"}:
            raise ValueError("action must be draft or submit")
        view = _task_view(task.get("view", "side"))
        points = _normalise_points(
            _annotation_points_payload(payload),
            task,
            _view_entry(pair, view),
        )
        result = _preview(task, points, pair)
        if action == "submit":
            if result["missing_points"]:
                raise ValueError(
                    "required points missing: " + ", ".join(result["missing_points"])
                )
            if result["angle_deg"] is None:
                raise ValueError("angle cannot be calculated; check the point placement")
        notes = str(payload.get("notes", ""))
        if len(notes) > 2000:
            raise ValueError("notes are limited to 2000 characters")
        target_mmpose = _redact_neutral_metadata(result.get("mmpose") or {})
        reference_mmpose = _redact_neutral_metadata(
            result.get("reference_mmpose") or {}
        )
        side_mmpose = target_mmpose if view == "side" else reference_mmpose
        front_mmpose = target_mmpose if view == "front" else reference_mmpose
        now = time.time()
        annotation = {
            "schema_version": ANNOTATION_SCHEMA_VERSION,
            "task_id": task["task_id"],
            "username": session["user"],
            "pair_id": task["pair_id"],
            "name": task["name"],
            "view": view,
            "view_label": task.get("view_label", _view_label(view)),
            "left_frame_index": task["left_frame_index"],
            "right_frame_index": task["right_frame_index"],
            "target_frame_index": result["frame_index"],
            "angle_points": task["angle_points"],
            "points": points,
            "points_2d": points,
            "angle_deg": result["angle_deg"],
            "signed_angle_deg": result.get("signed_angle_deg"),
            "angle_dimension": result["angle_dimension"],
            "angle_method": result.get("angle_method"),
            "angle_description": result.get("angle_description"),
            "calculation_note": result.get("calculation_note"),
            "points_3d": result["points_3d"],
            "depth_values_m": result["depth_values_m"],
            # ``mmpose`` remains the model result for the task's target
            # frame. The explicit side/front fields make the display and
            # exported annotation unambiguous for front-view tasks too.
            "mmpose": target_mmpose,
            "mmpose_angle_deg": target_mmpose.get("angle_deg"),
            "mmpose_error_deg": target_mmpose.get("error_deg"),
            "side_mmpose": side_mmpose,
            "side_mmpose_angle_deg": side_mmpose.get("angle_deg"),
            "side_mmpose_error_deg": result.get("side_mmpose_error_deg"),
            "front_mmpose": front_mmpose,
            "front_mmpose_angle_deg": front_mmpose.get("angle_deg"),
            "front_mmpose_error_deg": result.get("front_mmpose_error_deg"),
            "status": "submitted" if action == "submit" else "draft",
            "notes": notes,
            "created_at": existing.get("created_at", now) if existing else now,
            "updated_at": now,
        }
        if action == "submit":
            annotation["submitted_at"] = now
        _save_annotation(annotation)
        return jsonify({
            "ok": True,
            "task": _public_task(task, pair, include_neutral_metadata=False),
            "annotation": _redact_neutral_metadata(annotation),
            **_redact_neutral_metadata(result),
        })
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify({"error": "upload is larger than the configured limit"}), 413


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
