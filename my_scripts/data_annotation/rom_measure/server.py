#!/usr/bin/env python3
"""Flask backend for the blind RGBD ROM annotation workflow.

With role authentication configured, ``admin`` creates fixed-frame task
manifests and annotators complete those tasks without seeing MMPose until they
submit their blind points. Set ``ROM_ADMIN_PASSWORD`` and
``ROM_ANNOTATOR_PASSWORD`` before exposing the service.
"""
import argparse
import csv
import io
import json
import os
import sys
import secrets
import time
import hmac
from functools import lru_cache, wraps
from pathlib import Path

import numpy as np
from flask import (Flask, Response, jsonify, redirect, request,
                   send_file, send_from_directory, session)
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
ANNOT_DIR = os.path.join(BASE_DIR, "annotations")
STATIC_DIR = os.path.join(BASE_DIR, "static")
REPO_ROOT = Path(BASE_DIR).parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from my_scripts.data_evaluation.rgbd_geometry import (  # noqa: E402
    METRIC_LABELS,
    METRIC_TEMPLATES,
    compute_metric,
    compute_paired_metric,
    lift_points_2d,
    load_mmpose_json,
    metric_needs_reference,
    metric_pose_points,
    metric_uses_torso_frame,
    mmpose_points_for_frame,
    normalize_points,
    read_video_frame,
    video_metadata,
    DepthZipReader,
)
from my_scripts.data_evaluation.recordings import (  # noqa: E402
    discover_recordings,
    find_recording,
)


DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "NUS" / "val"
DATA_ROOT = Path(
    os.environ.get("ROM_DATA_ROOT", str(DEFAULT_DATA_ROOT))
).expanduser().resolve()
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ANNOTATION_SCHEMA_VERSION = 2
MMPose_QA_SCORE_THRESHOLD = 0.5
TASK_SCHEMA_VERSION = 3
TASK_ANNOTATION_SCHEMA_VERSION = 1
ADMIN_USERNAME = os.environ.get("ROM_ADMIN_USERNAME", "admin")
# TEMPORARY LOCAL-ONLY TESTING MODE: both roles use the same password.
# Restore environment-based passwords before exposing this service anywhere
# other than 127.0.0.1.
ADMIN_PASSWORD = "123"
ANNOTATOR_PASSWORD = "123"
ROLE_AUTH_CONFIGURED = True

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
app.secret_key = os.environ.get("ROM_SECRET", secrets.token_hex(32))
ROM_PASSWORD = os.environ.get("ROM_PASSWORD", "")

# Optional per-user passwords: ROM_USERS="alice:pw1,bob:pw2"
ROM_USERS = {}
if os.environ.get("ROM_USERS"):
    for pair in os.environ["ROM_USERS"].split(","):
        if ":" in pair:
            u, p = pair.split(":", 1)
            ROM_USERS[u.strip()] = p


# ---------- auth ----------
def login_required(f):
    @wraps(f)
    def wrapper(*a, **k):
        if not session.get("authed") or not session.get("user"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "unauthorized"}), 401
            return redirect("/")
        return f(*a, **k)
    return wrapper


def role_required(role):
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


def admin_or_legacy_required(function):
    """Keep legacy endpoints usable until the two-role auth is configured."""
    @wraps(function)
    def wrapper(*args, **kwargs):
        if not session.get("authed") or not session.get("user"):
            return login_required(function)(*args, **kwargs)
        if ROLE_AUTH_CONFIGURED and session.get("role") != "admin":
            return jsonify({"error": "admin access required"}), 403
        return function(*args, **kwargs)
    return wrapper


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/admin")
@admin_required
def admin_page():
    return send_from_directory(STATIC_DIR, "admin.html")


@app.post("/api/login")
def login():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username:
        return jsonify({"error": "username required"}), 401
    role = None
    if ROLE_AUTH_CONFIGURED:
        if username == ADMIN_USERNAME and ADMIN_PASSWORD and hmac.compare_digest(
            password, ADMIN_PASSWORD
        ):
            role = "admin"
        elif ROM_USERS and username in ROM_USERS and hmac.compare_digest(
            password, ROM_USERS[username]
        ):
            role = "annotator"
        elif ANNOTATOR_PASSWORD and hmac.compare_digest(
            password, ANNOTATOR_PASSWORD
        ) and username != ADMIN_USERNAME:
            role = "annotator"
    else:
        if ROM_USERS:
            if ROM_USERS.get(username) != password:
                return jsonify({"error": "wrong username/password"}), 401
        elif ROM_PASSWORD and password != ROM_PASSWORD:
            return jsonify({"error": "wrong password"}), 401
        role = "annotator"
    if role is None:
        return jsonify({"error": "wrong username/password"}), 401
    session["authed"] = True
    session["user"] = username
    session["role"] = role
    return jsonify({"ok": True, "user": username, "role": role})


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify({"ok": True})


def _safe_component(value, field):
    value = str(value or "").strip()
    cleaned = secure_filename(value)
    if not value or value in {".", ".."} or cleaned != value:
        raise ValueError(f"invalid {field}")
    return cleaned


def _subject_path(subject):
    subject = _safe_component(subject, "subject")
    path = (DATA_ROOT / subject).resolve()
    if path.parent != DATA_ROOT or not path.is_dir():
        raise FileNotFoundError(f"subject not found: {subject}")
    return subject, path


def _find_mmpose_json(subject_path, session_id, trial):
    root = subject_path / "mmpose"
    if not root.is_dir():
        return None
    matches = sorted(root.rglob(f"{trial}.json"))
    session_matches = [path for path in matches if session_id in path.parts]
    return (session_matches or matches)[0] if (session_matches or matches) else None


def _recording(subject, session_id, trial):
    found = find_recording(DATA_ROOT, subject, session_id, trial)
    return _recording_from_found(found)


def _recording_from_found(found):
    return {
        "subject": found.subject,
        "session_id": found.session_id,
        "trial": found.trial,
        "id": f"{found.subject}/{found.session_id}/{found.trial}",
        "video_path": found.video_path,
        "depth_path": found.depth_path,
        "calibration_path": found.calibration_path,
        "mmpose_path": found.mmpose_path,
        "video": {
            "width": found.video.width,
            "height": found.video.height,
            "frame_rate": found.video.frame_rate,
            "frame_count": found.video.frame_count,
        },
        "depth": found.depth,
    }


def _recording_response(recording, user=None):
    subject_path = DATA_ROOT / recording["subject"]

    def relative(path):
        return str(path.relative_to(subject_path)) if path else None

    annotation = _dataset_annotation_path(
        recording["subject"], recording["session_id"], recording["trial"], user
    ) if user else None
    return {
        "id": recording["id"],
        "subject": recording["subject"],
        "session_id": recording["session_id"],
        "trial": recording["trial"],
        "video": recording["video"],
        "depth": recording["depth"],
        "paths": {
            "video": relative(recording["video_path"]),
            "depth": relative(recording["depth_path"]),
            "calibration": relative(recording["calibration_path"]),
            "mmpose": relative(recording["mmpose_path"]),
        },
        "has_depth": recording["depth_path"] is not None,
        "has_calibration": recording["calibration_path"] is not None,
        "has_mmpose": recording["mmpose_path"] is not None,
        "annotated": bool(annotation and annotation.is_file()),
    }


def _request_recording(payload=None):
    values = payload or request.args
    return _recording(
        values.get("subject"),
        values.get("session_id"),
        values.get("trial"),
    )


def _dataset_annotation_path(subject, session_id, trial, user):
    if not user:
        return None
    subject, subject_path = _subject_path(subject)
    user = _safe_component(user, "username")
    session_id = _safe_component(session_id, "session")
    trial = _safe_component(trial, "trial")
    return subject_path / "annotations" / user / session_id / f"{trial}.json"


def _task_manifest_path(subject, session_id, trial):
    subject, subject_path = _subject_path(subject)
    session_id = _safe_component(session_id, "session")
    trial = _safe_component(trial, "trial")
    return subject_path / "annotation_tasks" / session_id / f"{trial}.json"


def _task_id(recording, metric, frame):
    return _safe_component(f"{recording['trial']}_{metric}_{frame}", "task_id")


def _task_frame(raw_value, recording, position, label):
    try:
        frame = int(raw_value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"task {position} has invalid {label}") from error
    if frame < 0 or frame >= recording["video"]["frame_count"]:
        raise ValueError(f"task {position} {label} out of range: {frame}")
    return frame


def _normalize_task(raw_task, recording, position=0):
    if not isinstance(raw_task, dict):
        raise ValueError(f"task {position} must be an object")
    metric = raw_task.get("metric")
    if metric not in METRIC_TEMPLATES:
        raise ValueError(f"unsupported task metric: {metric}")
    requires_pair = metric_needs_reference(metric)
    requested_mode = raw_task.get("frame_mode")
    if requested_mode not in {None, "single", "paired"}:
        raise ValueError(f"task {position} has invalid frame_mode")
    if requires_pair and requested_mode == "single":
        raise ValueError(f"{METRIC_LABELS[metric]} requires T1 and T2 frames")
    if not requires_pair and requested_mode == "paired":
        raise ValueError(f"{METRIC_LABELS[metric]} uses one frame")
    # The metric definition is authoritative. Paired segment metrics always
    # receive a neutral/baseline T1 and peak/end-range T2; hinges and axial
    # rotations always receive one fixed frame.
    is_paired = requires_pair
    if is_paired:
        t1 = _task_frame(
            raw_task.get("t1_frame_index", raw_task.get("frame_index")),
            recording,
            position,
            "t1_frame_index",
        )
        t2 = _task_frame(
            raw_task.get("t2_frame_index"),
            recording,
            position,
            "t2_frame_index",
        )
        if t1 == t2:
            raise ValueError(f"task {position} T1 and T2 must be different frames")
        default_id = f"{recording['trial']}_{metric}_{t1}_{t2}"
    else:
        frame = _task_frame(
            raw_task.get("frame_index"), recording, position, "frame_index"
        )
        default_id = f"{recording['trial']}_{metric}_{frame}"
    task_id = raw_task.get("task_id") or default_id
    task_id = _safe_component(task_id, "task_id")
    t1_points, t2_points = metric_pose_points(metric)
    required_points = list(dict.fromkeys((t1_points or []) + t2_points))
    task = {
        "task_id": task_id,
        "metric": metric,
        "metric_label": METRIC_LABELS[metric],
        "frame_mode": "paired" if is_paired else "single",
        "required_points": required_points,
        "enabled": bool(raw_task.get("enabled", True)),
        "notes": str(raw_task.get("notes", "")),
    }
    if is_paired:
        task.update({
            "frame_mode": "paired",
            "t1_frame_index": t1,
            "t2_frame_index": t2,
            "t1_timestamp_sec": t1 / recording["video"]["frame_rate"],
            "t2_timestamp_sec": t2 / recording["video"]["frame_rate"],
            "poses": {
                "t1": {
                    "frame_index": t1,
                    "timestamp_sec": t1 / recording["video"]["frame_rate"],
                    "required_points": t1_points or [],
                },
                "t2": {
                    "frame_index": t2,
                    "timestamp_sec": t2 / recording["video"]["frame_rate"],
                    "required_points": t2_points,
                },
            },
        })
    else:
        task.update({
            "frame_mode": "single",
            "frame_index": frame,
            "timestamp_sec": frame / recording["video"]["frame_rate"],
            "poses": {
                "frame": {
                    "frame_index": frame,
                    "timestamp_sec": frame / recording["video"]["frame_rate"],
                    "required_points": required_points,
                },
            },
        })
    task["annotation_hint"] = _task_annotation_hint(task)
    return task


def _task_annotation_hint(task):
    """Short instructions shown to the annotator for a fixed task."""
    metric = task["metric"]
    if task.get("frame_mode") == "paired":
        segment = "shoulder-to-elbow" if "shoulder" in metric else "hip-to-knee"
        return (
            f"Annotate the {segment} in both poses: T1 is neutral/baseline and "
            "T2 is peak/end range. The ROM is the angle between those two "
            "segment vectors; keep the trunk position as instructed by the test."
        )
    if metric_uses_torso_frame(metric):
        if "shoulder" in metric:
            return (
                "Single frame: annotate the nose, both shoulders, both hips, "
                "and the working-side elbow and wrist. Keep the upper arm and "
                "elbow in the agreed rotation-test posture."
            )
        return (
            "Single frame: annotate the nose, both shoulders, both hips, and "
            "the working-side knee and ankle. Keep the hip and knee in the "
            "agreed rotation-test posture."
        )
    return "Single frame: annotate all required joint landmarks."


def _blank_task_manifest(recording):
    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "subject": recording["subject"],
        "session_id": recording["session_id"],
        "trial": recording["trial"],
        "tasks": [],
        "created_at": time.time(),
        "updated_at": time.time(),
    }


def _load_task_manifest(recording):
    path = _task_manifest_path(
        recording["subject"], recording["session_id"], recording["trial"]
    )
    if not path.is_file():
        return _blank_task_manifest(recording)
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    tasks = [
        _normalize_task(raw_task, recording, position)
        for position, raw_task in enumerate(payload.get("tasks", []))
    ]
    manifest = _blank_task_manifest(recording)
    manifest.update({
        key: value
        for key, value in payload.items()
        if key not in {"tasks", "schema_version", "subject", "session_id", "trial"}
    })
    manifest["tasks"] = tasks
    return manifest


def _task_manifest_response(recording, manifest):
    subject_path = DATA_ROOT / recording["subject"]
    path = _task_manifest_path(
        recording["subject"], recording["session_id"], recording["trial"]
    )
    return {
        "manifest": manifest,
        "path": str(path),
        "recording": _recording_response(recording),
        "relative_path": str(path.relative_to(subject_path)),
    }


def _task_annotation_path(recording, user):
    if not user:
        return None
    _, subject_path = _subject_path(recording["subject"])
    username = _safe_component(user, "username")
    session_id = _safe_component(recording["session_id"], "session")
    trial = _safe_component(recording["trial"], "trial")
    return (
        subject_path
        / "annotations"
        / username
        / "task_reviews"
        / session_id
        / f"{trial}.json"
    )


def _blank_task_annotation(recording, user):
    return {
        "schema_version": TASK_ANNOTATION_SCHEMA_VERSION,
        "annotation_type": "rgbd_task_review",
        "subject": recording["subject"],
        "session_id": recording["session_id"],
        "trial": recording["trial"],
        "username": user,
        "tasks": {},
        "created_at": time.time(),
        "updated_at": time.time(),
    }


def _load_task_annotation(recording, user):
    path = _task_annotation_path(recording, user)
    if not path.is_file():
        return _blank_task_annotation(recording, user)
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload.get("tasks"), dict):
        return payload
    return _blank_task_annotation(recording, user)


def _task_request(payload=None):
    values = payload or request.args
    recording = _request_recording(values)
    task_id = _safe_component(values.get("task_id"), "task_id")
    manifest = _load_task_manifest(recording)
    task = next(
        (item for item in manifest["tasks"] if item["task_id"] == task_id),
        None,
    )
    if task is None or not task["enabled"]:
        raise FileNotFoundError(f"task not found: {task_id}")
    return recording, manifest, task


def _task_pose(task, pose=None):
    poses = task.get("poses", {})
    if not isinstance(poses, dict) or not poses:
        raise ValueError("task has no pose definitions")
    if task.get("frame_mode") == "paired":
        pose = pose or "t1"
        if pose not in ("t1", "t2"):
            raise ValueError("paired task pose must be t1 or t2")
    else:
        pose = "frame"
    selected = poses.get(pose)
    if not isinstance(selected, dict):
        raise ValueError(f"task has no pose: {pose}")
    return pose, selected


def _task_pose_keys(task):
    return ("t1", "t2") if task.get("frame_mode") == "paired" else ("frame",)


def _task_annotation_entry(recording, user, task_id):
    annotation = _load_task_annotation(recording, user)
    entry = annotation.get("tasks", {}).get(task_id)
    return annotation, entry


def _save_task_annotation(recording, user, annotation):
    path = _task_annotation_path(recording, user)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(annotation, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(temporary, path)
    return path


def _task_rgbd_comparison(recording, task, points, frame=None):
    if recording["depth_path"] is None:
        raise FileNotFoundError("depth archive not found")
    frame = task["frame_index"] if frame is None else int(frame)
    reader = DepthZipReader(str(recording["depth_path"]))
    try:
        manual_lifted = lift_points_2d(
            points,
            reader,
            frame,
            recording["video"]["width"],
            recording["video"]["height"],
            2,
        )
    finally:
        reader.close()
    manual_angle = compute_metric(manual_lifted.points, task["metric"])
    comparison = {
        "manual_angle_deg": _json_number(manual_angle),
        "mmpose_angle_deg": None,
        "mmpose_independent_angle_deg": None,
        "signed_error_deg": None,
        "mmpose_score_threshold": MMPose_QA_SCORE_THRESHOLD,
        "comparison_mode": "shared_3d_joint_angle",
        "mmpose_points": {},
        "missing_manual_points": [
            name
            for name in task["required_points"]
            if name not in manual_lifted.points
            or not np.isfinite(manual_lifted.points[name]).all()
        ],
        "missing_manual_depth": [
            name
            for name, value in manual_lifted.depths.items()
            if not np.isfinite(value)
        ],
        "missing_mmpose_points": [],
        "missing_mmpose_depth": [],
        "mmpose_error": None,
        "depth_timestamp": manual_lifted.depth_timestamp,
    }
    if recording["mmpose_path"] is None:
        comparison["mmpose_error"] = "MMPose JSON not found"
        return comparison
    try:
        labels, keypoints, scores = _cached_mmpose(
            str(recording["mmpose_path"]), MMPose_QA_SCORE_THRESHOLD
        )
        mmpose_2d = mmpose_points_for_frame(
            labels,
            keypoints,
            scores,
            frame,
            MMPose_QA_SCORE_THRESHOLD,
        )
        reader = DepthZipReader(str(recording["depth_path"]))
        try:
            mmpose_lifted = lift_points_2d(
                mmpose_2d,
                reader,
                frame,
                recording["video"]["width"],
                recording["video"]["height"],
                2,
            )
        finally:
            reader.close()
        comparison["mmpose_points"] = mmpose_2d
        comparison["mmpose_independent_angle_deg"] = _json_number(
            compute_metric(mmpose_lifted.points, task["metric"])
        )
        mmpose_points_for_angle = dict(mmpose_lifted.points)
        if metric_uses_torso_frame(task["metric"]):
            torso_points = (
                "nose",
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
            )
            has_manual_torso = all(
                name in manual_lifted.points
                and np.isfinite(manual_lifted.points[name]).all()
                for name in torso_points
            )
            if has_manual_torso:
                comparison["comparison_mode"] = "common_torso_frame"
                for name in torso_points:
                    mmpose_points_for_angle[name] = manual_lifted.points[name]
        mmpose_angle = compute_metric(mmpose_points_for_angle, task["metric"])
        comparison["mmpose_angle_deg"] = _json_number(mmpose_angle)
        comparison["missing_mmpose_points"] = [
            name
            for name in task["required_points"]
            if name not in mmpose_lifted.points
            or not np.isfinite(mmpose_lifted.points[name]).all()
        ]
        comparison["missing_mmpose_depth"] = [
            name
            for name, value in mmpose_lifted.depths.items()
            if not np.isfinite(value)
        ]
        if np.isfinite([manual_angle, mmpose_angle]).all():
            comparison["signed_error_deg"] = float(mmpose_angle - manual_angle)
    except (OSError, ValueError, IndexError, KeyError, TypeError) as error:
        comparison["mmpose_error"] = str(error)
    return comparison


def _lift_pose(recording, points, frame):
    reader = DepthZipReader(str(recording["depth_path"]))
    try:
        return lift_points_2d(
            points,
            reader,
            frame,
            recording["video"]["width"],
            recording["video"]["height"],
            2,
        )
    finally:
        reader.close()


def _task_rgbd_paired_comparison(recording, task, points_by_pose):
    """ROM for a T1=neutral / T2=peak segment-excursion task."""
    if task.get("frame_mode") != "paired":
        raise ValueError("paired comparison requires a paired task")
    if recording["depth_path"] is None:
        raise FileNotFoundError("depth archive not found")
    t1_def = task["poses"]["t1"]
    t2_def = task["poses"]["t2"]
    t1_points = points_by_pose.get("t1", {})
    t2_points = points_by_pose.get("t2", {})

    t1_lifted = _lift_pose(recording, t1_points, t1_def["frame_index"])
    t2_lifted = _lift_pose(recording, t2_points, t2_def["frame_index"])
    manual_rom = compute_paired_metric(
        t1_lifted.points,
        t2_lifted.points,
        task["metric"],
    )
    t1_required = task["poses"]["t1"].get("required_points", [])
    t2_required = task["poses"]["t2"].get("required_points", [])
    missing_manual_by_pose = {
        pose: [
            name
            for name in required
            if name not in lifted.points
            or not np.isfinite(lifted.points[name]).all()
        ]
        for pose, required, lifted in (
            ("t1", t1_required, t1_lifted),
            ("t2", t2_required, t2_lifted),
        )
    }

    mmpose_rom = None
    mmpose_points_t1 = {}
    mmpose_points_t2 = {}
    missing_mmpose_by_pose = {"t1": list(t1_required), "t2": list(t2_required)}
    mmpose_error = None
    if recording["mmpose_path"] is not None:
        try:
            labels, keypoints, scores = _cached_mmpose(
                str(recording["mmpose_path"]), MMPose_QA_SCORE_THRESHOLD
            )
            mmpose_points_by_pose = {
                "t1": mmpose_points_for_frame(
                    labels,
                    keypoints,
                    scores,
                    t1_def["frame_index"],
                    MMPose_QA_SCORE_THRESHOLD,
                ),
                "t2": mmpose_points_for_frame(
                    labels,
                    keypoints,
                    scores,
                    t2_def["frame_index"],
                    MMPose_QA_SCORE_THRESHOLD,
                ),
            }
            mmpose_lifted_by_pose = {
                pose: _lift_pose(
                    recording,
                    points,
                    task["poses"][pose]["frame_index"],
                )
                for pose, points in mmpose_points_by_pose.items()
            }
            mmpose_rom = compute_paired_metric(
                mmpose_lifted_by_pose["t1"].points,
                mmpose_lifted_by_pose["t2"].points,
                task["metric"],
            )
            mmpose_points_t1 = mmpose_points_by_pose["t1"]
            mmpose_points_t2 = mmpose_points_by_pose["t2"]
            missing_mmpose_by_pose = {
                pose: [
                    name
                    for name in task["poses"][pose].get("required_points", [])
                    if name not in mmpose_lifted_by_pose[pose].points
                    or not np.isfinite(
                        mmpose_lifted_by_pose[pose].points[name]
                    ).all()
                ]
                for pose in ("t1", "t2")
            }
        except (OSError, ValueError, IndexError, KeyError, TypeError) as error:
            mmpose_error = str(error)

    signed_error = None
    if all(v is not None and np.isfinite(v) for v in (manual_rom, mmpose_rom)):
        signed_error = float(mmpose_rom - manual_rom)

    return {
        "task_id": task["task_id"],
        "metric": task["metric"],
        "frame_mode": "paired",
        "manual_rom_deg": _json_number(manual_rom),
        "mmpose_rom_deg": _json_number(mmpose_rom),
        "signed_error_deg": signed_error,
        "manual_t1_angle_deg": None,
        "manual_t2_angle_deg": None,
        "manual_delta_deg": _json_number(manual_rom),
        "mmpose_t1_angle_deg": None,
        "mmpose_t2_angle_deg": None,
        "mmpose_delta_deg": _json_number(mmpose_rom),
        "manual_missing_points_by_pose": missing_manual_by_pose,
        "mmpose_missing_points_by_pose": missing_mmpose_by_pose,
        "mmpose_points_by_pose": {
            "t1": mmpose_points_t1,
            "t2": mmpose_points_t2,
        },
        "mmpose_score_threshold": MMPose_QA_SCORE_THRESHOLD,
        "comparison_mode": "paired_segment_excursion",
        "mmpose_error": mmpose_error,
        "revealed": True,
    }


def _difference(second, first):
    if second is None or first is None:
        return None
    return float(second - first)


def _blank_dataset_annotation(recording, user):
    subject_path = DATA_ROOT / recording["subject"]

    def relative(path):
        return str(path.relative_to(subject_path)) if path else None

    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "annotation_type": "rgbd_keypoints",
        "subject": recording["subject"],
        "session_id": recording["session_id"],
        "trial": recording["trial"],
        "username": user,
        "source": {
            "video": relative(recording["video_path"]),
            "depth": relative(recording["depth_path"]),
            "calibration": relative(recording["calibration_path"]),
            "mmpose": relative(recording["mmpose_path"]),
            "video_width": recording["video"]["width"],
            "video_height": recording["video"]["height"],
            "frame_rate": recording["video"]["frame_rate"],
            "frame_count": recording["video"]["frame_count"],
            "depth": recording["depth"],
        },
        "active_metric": "right_elbow_flexion",
        "metrics": {
            metric: {"t1": None, "t2": None, "frames": {}}
            for metric in METRIC_TEMPLATES
        },
        "created_at": time.time(),
        "updated_at": time.time(),
    }


def _load_dataset_annotation(recording, user):
    path = _dataset_annotation_path(
        recording["subject"], recording["session_id"], recording["trial"], user
    )
    if not path or not path.is_file():
        return None
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _annotation_metric_states(annotation):
    metrics = annotation.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    metric = annotation.get("metric")
    if metric in METRIC_TEMPLATES:
        return {
            metric: {
                "t1": annotation.get("t1"),
                "t2": annotation.get("t2"),
                "frames": annotation.get("frames", {}),
            }
        }
    return {}


def _normalized_dataset_annotation(annotation, recording, user):
    normalized = _blank_dataset_annotation(recording, user)
    if annotation:
        normalized.update({
            key: value
            for key, value in annotation.items()
            if key not in {"metrics", "metric", "t1", "t2", "frames"}
        })
        normalized["created_at"] = annotation.get(
            "created_at", normalized["created_at"]
        )
    normalized["schema_version"] = ANNOTATION_SCHEMA_VERSION
    states = _annotation_metric_states(annotation or {})
    for metric, state in states.items():
        if metric not in METRIC_TEMPLATES or not isinstance(state, dict):
            continue
        normalized["metrics"][metric] = {
            "t1": state.get("t1"),
            "t2": state.get("t2"),
            "frames": state.get("frames", {}) if isinstance(state.get("frames", {}), dict) else {},
        }
    active_metric = (annotation or {}).get(
        "active_metric", (annotation or {}).get("metric", normalized["active_metric"])
    )
    if active_metric in METRIC_TEMPLATES:
        normalized["active_metric"] = active_metric
    normalized["username"] = user
    return normalized


def _json_number(value):
    if value is None:
        return None
    value = float(value)
    return value if value == value and abs(value) != float("inf") else None


@lru_cache(maxsize=16)
def _cached_mmpose(filename, score_threshold):
    return load_mmpose_json(filename, score_threshold)


@app.get("/api/config")
@admin_or_legacy_required
def config():
    return jsonify({
        "data_root": str(DATA_ROOT),
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
        "metrics": [
            {
                "key": key,
                "label": METRIC_LABELS[key],
                "points": list(points),
                "frame_mode": "paired" if metric_needs_reference(key) else "single",
                "pose_points": {
                    "t1": metric_pose_points(key)[0] or [],
                    "t2": metric_pose_points(key)[1],
                },
            }
            for key, points in METRIC_TEMPLATES.items()
        ],
    })


@app.get("/api/admin/recordings")
@admin_required
def admin_recordings():
    return jsonify({
        "recordings": [
            _recording_response(_recording_from_found(found), session["user"])
            for found in discover_recordings(DATA_ROOT)
        ]
    })


@app.get("/api/admin/tasks")
@admin_required
def get_admin_tasks():
    try:
        recording = _request_recording()
        manifest = _load_task_manifest(recording)
        return jsonify(_task_manifest_response(recording, manifest))
    except (FileNotFoundError, OSError, ValueError) as error:
        return jsonify({"error": str(error)}), 404


@app.post("/api/admin/tasks")
@admin_required
def save_admin_tasks():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        recording = _request_recording(payload)
        raw_tasks = payload.get("tasks", [])
        if not isinstance(raw_tasks, list):
            raise ValueError("tasks must be a list")
        tasks = [
            _normalize_task(raw_task, recording, position)
            for position, raw_task in enumerate(raw_tasks)
        ]
        task_ids = [task["task_id"] for task in tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_id values must be unique")
        existing = _load_task_manifest(recording)
        manifest = _blank_task_manifest(recording)
        manifest["tasks"] = tasks
        manifest["created_at"] = existing.get("created_at", manifest["created_at"])
        manifest["updated_at"] = time.time()
        path = _task_manifest_path(
            recording["subject"], recording["session_id"], recording["trial"]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, path)
        return jsonify({
            "ok": True,
            **_task_manifest_response(recording, manifest),
        })
    except (FileNotFoundError, OSError, ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/tasks")
@annotator_required
def list_annotator_tasks():
    tasks = []
    for found in discover_recordings(DATA_ROOT):
        recording = _recording_from_found(found)
        manifest = _load_task_manifest(recording)
        annotation = _load_task_annotation(recording, session["user"])
        for task in manifest["tasks"]:
            if not task["enabled"]:
                continue
            entry = annotation.get("tasks", {}).get(task["task_id"], {})
            tasks.append({
                **task,
                "subject": recording["subject"],
                "session_id": recording["session_id"],
                "trial": recording["trial"],
                "recording_id": recording["id"],
                "status": entry.get("status", "draft"),
                "mmpose_reviewed": bool(
                    entry.get("mmpose_reviewed", False)
                ),
                "mmpose_agree": entry.get("mmpose_agree"),
            })
    return jsonify({"tasks": tasks})


def _task_client_payload(recording, task, entry):
    payload = {
        "task": task,
        "recording": {
            "id": recording["id"],
            "video": recording["video"],
            "has_depth": recording["depth_path"] is not None,
            "has_mmpose": recording["mmpose_path"] is not None,
        },
        "status": entry.get("status", "draft") if entry else "draft",
    }
    if task.get("frame_mode") == "paired":
        payload["blind_points_2d_by_pose"] = (
            entry.get("blind_points_2d_by_pose", {}) if entry else {}
        )
    else:
        payload["blind_points_2d"] = (
            entry.get("blind_points_2d", {}) if entry else {}
        )
    if entry:
        payload["entry"] = entry
    return payload


@app.get("/api/task")
@annotator_required
def get_annotator_task():
    try:
        recording, _, task = _task_request()
        _, entry = _task_annotation_entry(
            recording, session["user"], task["task_id"]
        )
        return jsonify(_task_client_payload(recording, task, entry))
    except (FileNotFoundError, OSError, ValueError) as error:
        return jsonify({"error": str(error)}), 404


def _video_frame_response(recording, frame):
    if frame < 0 or frame >= recording["video"]["frame_count"]:
        raise IndexError(f"frame out of range: {frame}")
    image = read_video_frame(str(recording["video_path"]), frame)
    import cv2
    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )
    if not ok:
        raise OSError("could not encode video frame")
    response = Response(encoded.tobytes(), mimetype="image/jpeg")
    response.headers["X-Frame-Index"] = str(frame)
    response.headers["X-Frame-Timestamp"] = (
        f"{frame / recording['video']['frame_rate']:.6f}"
    )
    return response


@app.get("/api/task-frame")
@annotator_required
def get_task_frame():
    try:
        recording, _, task = _task_request()
        pose, definition = _task_pose(task, request.args.get("pose"))
        requested = request.args.get("frame")
        if requested is not None and int(requested) != definition["frame_index"]:
            raise ValueError("annotator frames are fixed by the task manifest")
        response = _video_frame_response(recording, definition["frame_index"])
        response.headers["X-Task-Pose"] = pose
        return response
    except (FileNotFoundError, OSError, ValueError, IndexError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/task-depth-overlay")
@annotator_required
def get_task_depth_overlay():
    try:
        recording, _, task = _task_request()
        _, definition = _task_pose(task, request.args.get("pose"))
        if recording["depth_path"] is None:
            raise FileNotFoundError("depth archive not found")
        reader = DepthZipReader(str(recording["depth_path"]))
        try:
            depth = reader.read(definition["frame_index"])
        finally:
            reader.close()
        if depth is None:
            raise IndexError("depth frame unavailable")
        import cv2
        valid = np.isfinite(depth)
        normalized = np.clip((np.nan_to_num(depth, nan=6.0) - 0.2) / 5.8, 0.0, 1.0)
        bgr = cv2.applyColorMap(
            (normalized * 255).astype("uint8"), cv2.COLORMAP_TURBO
        )
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = np.where(valid, 175, 0).astype("uint8")
        ok, encoded = cv2.imencode(".png", rgba)
        if not ok:
            raise OSError("could not encode depth overlay")
        return Response(encoded.tobytes(), mimetype="image/png")
    except (FileNotFoundError, OSError, ValueError, IndexError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/api/task-preview")
@annotator_required
def preview_task_manual_rgbd():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        recording, _, task = _task_request(payload)
        pose, definition = _task_pose(task, payload.get("pose"))
        if recording["depth_path"] is None:
            raise FileNotFoundError("depth archive not found")
        if task.get("frame_mode") == "paired":
            points_by_pose = payload.get("blind_points_2d_by_pose", {})
            lifted_by_pose = {}
            for pose_name in _task_pose_keys(task):
                pose_definition = task["poses"][pose_name]
                lifted_by_pose[pose_name] = _lift_pose(
                    recording,
                    normalize_points(points_by_pose.get(pose_name, {})),
                    pose_definition["frame_index"],
                )
            lifted = lifted_by_pose[pose]
            angle = compute_paired_metric(
                lifted_by_pose["t1"].points,
                lifted_by_pose["t2"].points,
                task["metric"],
            )
            missing_points = [
                f"{pose_name}: {name}"
                for pose_name in _task_pose_keys(task)
                for name in task["poses"][pose_name].get("required_points", [])
                if name not in lifted_by_pose[pose_name].points
                or not np.isfinite(lifted_by_pose[pose_name].points[name]).all()
            ]
            missing_depth = [
                f"{pose_name}: {name}"
                for pose_name in _task_pose_keys(task)
                for name, value in lifted_by_pose[pose_name].depths.items()
                if not np.isfinite(value)
            ]
        else:
            points = normalize_points(
                payload.get("blind_points_2d", payload.get("points_2d", {}))
            )
            lifted = _lift_pose(recording, points, definition["frame_index"])
            pose_required = task["poses"][pose].get(
                "required_points", task["required_points"]
            )
            missing_points = [
                name
                for name in pose_required
                if name not in lifted.points
                or not np.isfinite(lifted.points[name]).all()
            ]
            missing_depth = [
                name
                for name, value in lifted.depths.items()
                if not np.isfinite(value)
            ]
            angle = compute_metric(lifted.points, task["metric"])
        return jsonify({
            "task_id": task["task_id"],
            "metric": task["metric"],
            "pose": pose,
            "frame_index": definition["frame_index"],
            "rgb_timestamp_sec": definition["timestamp_sec"],
            "depth_timestamp": lifted.depth_timestamp,
            "manual_angle_deg": _json_number(angle),
            "manual_rom_deg": _json_number(angle),
            "missing_points": missing_points,
            "missing_depth": missing_depth,
            "valid": bool(np.isfinite(angle) and not missing_points),
            "revealed": False,
        })
    except (FileNotFoundError, OSError, ValueError, IndexError, KeyError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/task-mmpose-frame")
@annotator_required
def get_task_mmpose_frame():
    try:
        recording, _, task = _task_request()
        pose, definition = _task_pose(task, request.args.get("pose"))
        _, entry = _task_annotation_entry(
            recording, session["user"], task["task_id"]
        )
        if not entry or entry.get("status") not in {"submitted", "reviewed"}:
            raise PermissionError("MMPose is hidden until annotation is finished")
        labels, keypoints, scores = _cached_mmpose(
            str(recording["mmpose_path"]), MMPose_QA_SCORE_THRESHOLD
        )
        points = mmpose_points_for_frame(
            labels,
            keypoints,
            scores,
            definition["frame_index"],
            MMPose_QA_SCORE_THRESHOLD,
        )
        return jsonify({
            "task_id": task["task_id"],
            "pose": pose,
            "frame_index": definition["frame_index"],
            "metric": task["metric"],
            "points": {
                name: points[name]
                for name in task["required_points"]
                if name in points
            },
            "score_threshold": MMPose_QA_SCORE_THRESHOLD,
            "revealed": True,
        })
    except PermissionError as error:
        return jsonify({"error": str(error)}), 403
    except (FileNotFoundError, OSError, ValueError, IndexError, KeyError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/task-comparison")
@annotator_required
def get_task_comparison():
    try:
        recording, _, task = _task_request()
        annotation, entry = _task_annotation_entry(
            recording, session["user"], task["task_id"]
        )
        if not entry or entry.get("status") not in {"submitted", "reviewed"}:
            raise PermissionError("comparison is hidden until annotation is finished")
        if task.get("frame_mode") == "paired":
            comparison = _task_rgbd_paired_comparison(
                recording,
                task,
                entry.get("blind_points_2d_by_pose", {}),
            )
        else:
            comparison = _task_rgbd_comparison(
                recording, task, entry.get("blind_points_2d", {})
            )
        comparison.update({
            "task_id": task["task_id"],
            "metric": task["metric"],
            "frame_mode": task.get("frame_mode", "single"),
            "revealed": True,
            "mmpose_reviewed": bool(entry.get("mmpose_reviewed", False)),
            "mmpose_agree": entry.get("mmpose_agree"),
        })
        return jsonify(comparison)
    except PermissionError as error:
        return jsonify({"error": str(error)}), 403
    except (FileNotFoundError, OSError, ValueError, IndexError, KeyError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/task-annotation")
@annotator_required
def get_task_annotation():
    try:
        recording, _, task = _task_request()
        _, entry = _task_annotation_entry(
            recording, session["user"], task["task_id"]
        )
        return jsonify(_task_client_payload(recording, task, entry))
    except (FileNotFoundError, OSError, ValueError) as error:
        return jsonify({"error": str(error)}), 404


@app.post("/api/task-annotation")
@annotator_required
def save_task_annotation():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        recording, _, task = _task_request(payload)
        action = payload.get("action", "draft")
        annotation, existing = _task_annotation_entry(
            recording, session["user"], task["task_id"]
        )
        existing_status = existing.get("status", "draft") if existing else "draft"
        if existing_status in {"submitted", "reviewed"} and action in {"draft", "submit"}:
            raise ValueError("submitted annotations are locked")
        if action in {"draft", "submit"}:
            if task.get("frame_mode") == "paired":
                raw_points_by_pose = payload.get("blind_points_2d_by_pose", {})
                if not isinstance(raw_points_by_pose, dict):
                    raise ValueError("blind_points_2d_by_pose must be an object")
                points_by_pose = {
                    pose: normalize_points(raw_points_by_pose.get(pose, {}))
                    for pose in _task_pose_keys(task)
                }
                raw_quality_by_pose = payload.get("quality_by_pose", {})
                if not isinstance(raw_quality_by_pose, dict):
                    raise ValueError("quality_by_pose must be an object")
                quality_by_pose = {
                    pose: dict(raw_quality_by_pose.get(pose, {}))
                    for pose in _task_pose_keys(task)
                }
                for quality in quality_by_pose.values():
                    quality["usable"] = bool(quality.get("usable", False))
                raw_notes_by_pose = payload.get("notes_by_pose", {})
                if not isinstance(raw_notes_by_pose, dict):
                    raise ValueError("notes_by_pose must be an object")
                notes_by_pose = {
                    pose: str(raw_notes_by_pose.get(pose, ""))
                    for pose in _task_pose_keys(task)
                }
                for pose, points in points_by_pose.items():
                    for name, point in points.items():
                        if not (
                            0 <= point["x"] <= recording["video"]["width"]
                            and 0 <= point["y"] <= recording["video"]["height"]
                        ):
                            raise ValueError(
                                f"point outside RGB frame ({pose}): {name}"
                            )
                if action == "draft":
                    entry = {
                        "task_id": task["task_id"],
                        "metric": task["metric"],
                        "frame_mode": "paired",
                        "poses": task["poses"],
                        "status": "draft",
                        "blind_points_2d_by_pose": points_by_pose,
                        "quality_by_pose": quality_by_pose,
                        "notes_by_pose": notes_by_pose,
                        "updated_at": time.time(),
                    }
                else:
                    missing_by_pose = {
                        pose: [
                            name
                            for name in task["poses"][pose].get("required_points", [])
                            if name not in points_by_pose[pose]
                        ]
                        for pose in _task_pose_keys(task)
                    }
                    missing_by_pose = {
                        pose: missing
                        for pose, missing in missing_by_pose.items()
                        if missing
                    }
                    if missing_by_pose:
                        details = "; ".join(
                            f"{pose}: {', '.join(missing)}"
                            for pose, missing in missing_by_pose.items()
                        )
                        raise ValueError("required points missing: " + details)
                    comparison = _task_rgbd_paired_comparison(
                        recording, task, points_by_pose
                    )
                    entry = {
                        "task_id": task["task_id"],
                        "metric": task["metric"],
                        "frame_mode": "paired",
                        "poses": task["poses"],
                        "status": "submitted",
                        "blind_points_2d_by_pose": points_by_pose,
                        "quality_by_pose": quality_by_pose,
                        "notes_by_pose": notes_by_pose,
                        "submitted_at": time.time(),
                        "revealed_at": time.time(),
                        "manual_t1_angle_deg": comparison["manual_t1_angle_deg"],
                        "manual_t2_angle_deg": comparison["manual_t2_angle_deg"],
                        "manual_delta_deg": comparison["manual_delta_deg"],
                        "mmpose_t1_angle_deg": comparison["mmpose_t1_angle_deg"],
                        "mmpose_t2_angle_deg": comparison["mmpose_t2_angle_deg"],
                        "mmpose_delta_deg": comparison["mmpose_delta_deg"],
                        "signed_error_deg": comparison["signed_error_deg"],
                        "comparison_mode": comparison["comparison_mode"],
                    }
            else:
                points = normalize_points(
                    payload.get("blind_points_2d", payload.get("points_2d", {}))
                )
                for name, point in points.items():
                    if not (
                        0 <= point["x"] <= recording["video"]["width"]
                        and 0 <= point["y"] <= recording["video"]["height"]
                    ):
                        raise ValueError(f"point outside RGB frame: {name}")
                quality = payload.get("quality", {})
                if not isinstance(quality, dict):
                    raise ValueError("quality must be an object")
                quality = dict(quality)
                quality["usable"] = bool(quality.get("usable", False))
                if action == "draft":
                    entry = {
                        "task_id": task["task_id"],
                        "metric": task["metric"],
                        "frame_mode": "single",
                        "poses": task["poses"],
                        "frame_index": task["frame_index"],
                        "timestamp_sec": task["timestamp_sec"],
                        "status": "draft",
                        "blind_points_2d": points,
                        "quality": quality,
                        "notes": str(payload.get("notes", "")),
                        "updated_at": time.time(),
                    }
                else:
                    missing = [name for name in task["required_points"] if name not in points]
                    if missing:
                        raise ValueError(
                            "required points missing: " + ", ".join(missing)
                        )
                    comparison = _task_rgbd_comparison(recording, task, points)
                    entry = {
                        "task_id": task["task_id"],
                        "metric": task["metric"],
                        "frame_mode": "single",
                        "poses": task["poses"],
                        "frame_index": task["frame_index"],
                        "timestamp_sec": task["timestamp_sec"],
                        "status": "submitted",
                        "blind_points_2d": points,
                        "quality": quality,
                        "notes": str(payload.get("notes", "")),
                        "submitted_at": time.time(),
                        "revealed_at": time.time(),
                        "manual_angle_deg": comparison["manual_angle_deg"],
                        "mmpose_angle_deg": comparison["mmpose_angle_deg"],
                        "mmpose_independent_angle_deg": comparison["mmpose_independent_angle_deg"],
                        "signed_error_deg": comparison["signed_error_deg"],
                        "comparison_mode": comparison["comparison_mode"],
                    }
        elif action == "review":
            if existing_status not in {"submitted", "reviewed"}:
                raise ValueError("finish the blind annotation before reviewing")
            decision = payload.get("review_decision")
            if decision not in {"agree", "disagree"}:
                raise ValueError("review_decision must be agree or disagree")
            entry = dict(existing)
            entry["status"] = "reviewed"
            entry["mmpose_reviewed"] = True
            entry["mmpose_agree"] = decision == "agree"
            entry["disagreement_reason"] = str(payload.get("disagreement_reason", ""))
            entry["reviewed_at"] = time.time()
        else:
            raise ValueError("action must be draft, submit, or review")
        annotation["tasks"][task["task_id"]] = entry
        annotation["updated_at"] = time.time()
        path = _save_task_annotation(recording, session["user"], annotation)
        response = {
            "ok": True,
            "path": str(path),
            "task": task,
            "status": entry["status"],
            "entry": entry,
        }
        if entry["status"] in {"submitted", "reviewed"}:
            if task.get("frame_mode") == "paired":
                response["comparison"] = _task_rgbd_paired_comparison(
                    recording,
                    task,
                    entry["blind_points_2d_by_pose"],
                )
            else:
                response["comparison"] = _task_rgbd_comparison(
                    recording, task, entry["blind_points_2d"]
                )
            response["revealed"] = True
        else:
            response["revealed"] = False
        return jsonify(response)
    except (FileNotFoundError, OSError, ValueError, IndexError, KeyError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/recordings")
@admin_or_legacy_required
def list_recordings():
    recordings = []
    if DATA_ROOT.is_dir():
        for subject_path in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
            video_root = subject_path / "videos"
            if not video_root.is_dir():
                continue
            for session_path in sorted(path for path in video_root.iterdir() if path.is_dir()):
                for video in sorted(
                    path for path in session_path.iterdir()
                    if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
                ):
                    try:
                        recording = _recording(
                            subject_path.name, session_path.name, video.stem
                        )
                    except (FileNotFoundError, OSError, ValueError):
                        continue
                    recordings.append(_recording_response(recording, session["user"]))
    return jsonify({"recordings": recordings})


@app.get("/api/recording")
@admin_or_legacy_required
def get_recording():
    try:
        recording = _request_recording()
    except (FileNotFoundError, OSError, ValueError) as error:
        return jsonify({"error": str(error)}), 404
    return jsonify(_recording_response(recording, session["user"]))


@app.get("/api/frame")
@admin_or_legacy_required
def get_video_frame():
    try:
        recording = _request_recording()
        frame = int(request.args.get("frame", "0"))
        if frame < 0 or frame >= recording["video"]["frame_count"]:
            raise IndexError(f"frame out of range: {frame}")
        image = read_video_frame(str(recording["video_path"]), frame)
        import cv2
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])
        if not ok:
            raise OSError("could not encode video frame")
        response = Response(encoded.tobytes(), mimetype="image/jpeg")
        response.headers["X-Frame-Index"] = str(frame)
        response.headers["X-Frame-Timestamp"] = f"{frame / recording['video']['frame_rate']:.6f}"
        return response
    except (FileNotFoundError, OSError, ValueError, IndexError) as error:
        return jsonify({"error": str(error)}), 404


@app.get("/api/depth-overlay")
@admin_or_legacy_required
def get_depth_overlay():
    try:
        recording = _request_recording()
        if recording["depth_path"] is None:
            raise FileNotFoundError("depth archive not found")
        frame = int(request.args.get("frame", "0"))
        minimum = float(request.args.get("min", "0.2"))
        maximum = float(request.args.get("max", "6.0"))
        if maximum <= minimum:
            raise ValueError("depth max must be greater than depth min")
        reader = DepthZipReader(str(recording["depth_path"]))
        try:
            depth = reader.read(frame)
        finally:
            reader.close()
        if depth is None:
            raise IndexError(f"depth frame out of range: {frame}")
        import cv2
        valid = np.isfinite(depth)
        normalized = np.clip((np.nan_to_num(depth, nan=maximum) - minimum) / (maximum - minimum), 0.0, 1.0)
        bgr = cv2.applyColorMap((normalized * 255).astype("uint8"), cv2.COLORMAP_TURBO)
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = np.where(valid, 175, 0).astype("uint8")
        ok, encoded = cv2.imencode(".png", rgba)
        if not ok:
            raise OSError("could not encode depth overlay")
        return Response(encoded.tobytes(), mimetype="image/png")
    except (FileNotFoundError, OSError, ValueError, IndexError) as error:
        return jsonify({"error": str(error)}), 404


@app.get("/api/mmpose-frame")
@admin_or_legacy_required
def get_mmpose_frame():
    try:
        recording = _request_recording()
        if recording["mmpose_path"] is None:
            raise FileNotFoundError("MMPose JSON not found")
        metric = request.args.get("metric", "right_elbow_flexion")
        if metric not in METRIC_TEMPLATES:
            raise ValueError(f"unsupported metric: {metric}")
        frame = int(request.args.get("frame", "0"))
        labels, keypoints, scores = _cached_mmpose(
            str(recording["mmpose_path"]), MMPose_QA_SCORE_THRESHOLD
        )
        if frame < 0 or frame >= len(keypoints):
            raise IndexError(f"MMPose frame out of range: {frame}")
        all_points = mmpose_points_for_frame(
            labels,
            keypoints,
            scores,
            frame,
            MMPose_QA_SCORE_THRESHOLD,
        )
        return jsonify({
            "frame_index": frame,
            "metric": metric,
            "points": {
                name: all_points[name]
                for name in METRIC_TEMPLATES[metric]
                if name in all_points
            },
            "score_threshold": MMPose_QA_SCORE_THRESHOLD,
        })
    except (FileNotFoundError, OSError, ValueError, IndexError) as error:
        return jsonify({"error": str(error)}), 404


@app.post("/api/preview")
@admin_or_legacy_required
def preview_rgbd_metric():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        recording = _request_recording(payload)
        if recording["depth_path"] is None:
            raise FileNotFoundError("depth archive not found")
        metric = payload.get("metric", "right_elbow_flexion")
        if metric not in METRIC_TEMPLATES:
            raise ValueError(f"unsupported metric: {metric}")
        frame = int(payload.get("frame", 0))
        points = normalize_points(payload.get("points", {}))
        reader = DepthZipReader(str(recording["depth_path"]))
        try:
            lifted = lift_points_2d(
                points,
                reader,
                frame,
                recording["video"]["width"],
                recording["video"]["height"],
                int(payload.get("depth_sample_radius", 2)),
            )
        finally:
            reader.close()
        manual_angle = compute_metric(lifted.points, metric)
        missing_depth = [
            name for name, value in lifted.depths.items() if not np.isfinite(value)
        ]
        missing_manual_points = [
            name
            for name in METRIC_TEMPLATES[metric]
            if name not in lifted.points or not np.isfinite(lifted.points[name]).all()
        ]
        mmpose_angle = None
        mmpose_independent_angle = None
        mmpose_missing_depth = []
        missing_mmpose_points = []
        mmpose_error = None
        comparison_mode = "shared_3d_joint_angle"
        if recording["mmpose_path"] is not None:
            try:
                labels, keypoints, scores = _cached_mmpose(
                    str(recording["mmpose_path"]), MMPose_QA_SCORE_THRESHOLD
                )
                if frame >= len(keypoints):
                    raise IndexError(f"MMPose frame out of range: {frame}")
                mmpose_2d = mmpose_points_for_frame(
                    labels,
                    keypoints,
                    scores,
                    frame,
                    MMPose_QA_SCORE_THRESHOLD,
                )
                mmpose_reader = DepthZipReader(str(recording["depth_path"]))
                try:
                    mmpose_lifted = lift_points_2d(
                        mmpose_2d,
                        mmpose_reader,
                        frame,
                        recording["video"]["width"],
                        recording["video"]["height"],
                        int(payload.get("depth_sample_radius", 2)),
                    )
                finally:
                    mmpose_reader.close()
                mmpose_independent_angle = compute_metric(
                    mmpose_lifted.points, metric
                )
                mmpose_points_for_angle = dict(mmpose_lifted.points)
                if metric_uses_torso_frame(metric):
                    manual_torso_points = (
                        "nose",
                        "left_shoulder",
                        "right_shoulder",
                        "left_hip",
                        "right_hip",
                    )
                    has_manual_torso = all(
                        name in lifted.points and np.isfinite(lifted.points[name]).all()
                        for name in manual_torso_points
                    )
                    comparison_mode = (
                        "common_torso_frame" if has_manual_torso else "independent"
                    )
                    if has_manual_torso:
                        for name in manual_torso_points:
                            mmpose_points_for_angle[name] = lifted.points[name]
                mmpose_angle = compute_metric(mmpose_points_for_angle, metric)
                mmpose_missing_depth = [
                    name
                    for name, value in mmpose_lifted.depths.items()
                    if not np.isfinite(value)
                ]
                missing_mmpose_points = [
                    name
                    for name in METRIC_TEMPLATES[metric]
                    if name not in mmpose_lifted.points
                    or not np.isfinite(mmpose_lifted.points[name]).all()
                ]
            except (OSError, ValueError, IndexError, KeyError, TypeError) as error:
                mmpose_error = str(error)
        signed_error = (
            mmpose_angle - manual_angle
            if mmpose_angle is not None
            and np.isfinite([mmpose_angle, manual_angle]).all()
            else np.nan
        )
        return jsonify({
            "metric": metric,
            "frame_index": frame,
            "rgb_timestamp_sec": frame / recording["video"]["frame_rate"],
            "depth_timestamp": lifted.depth_timestamp,
            "angle_deg": _json_number(manual_angle),
            "manual_angle_deg": _json_number(manual_angle),
            "mmpose_angle_deg": _json_number(mmpose_angle),
            "mmpose_independent_angle_deg": _json_number(mmpose_independent_angle),
            "signed_error_deg": _json_number(signed_error),
            "points_3d": {
                name: [_json_number(component) for component in point]
                for name, point in lifted.points.items()
            },
            "depths": {name: _json_number(value) for name, value in lifted.depths.items()},
            "missing_depth": missing_depth,
            "missing_manual_points": missing_manual_points,
            "missing_mmpose_depth": mmpose_missing_depth,
            "missing_mmpose_points": missing_mmpose_points,
            "mmpose_error": mmpose_error,
            "mmpose_score_threshold": MMPose_QA_SCORE_THRESHOLD,
            "comparison_mode": comparison_mode,
            "required_points": METRIC_TEMPLATES[metric],
            "valid": bool(np.isfinite(manual_angle)),
            "mmpose_valid": bool(
                mmpose_angle is not None and np.isfinite(mmpose_angle)
            ),
            "rectification_applied": False,
        })
    except (FileNotFoundError, OSError, ValueError, IndexError, KeyError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


@app.get("/api/annotation")
@admin_or_legacy_required
def get_dataset_annotation():
    try:
        recording = _request_recording()
        annotation = _load_dataset_annotation(recording, session["user"])
        return jsonify(_normalized_dataset_annotation(
            annotation, recording, session["user"]
        ))
    except (FileNotFoundError, OSError, ValueError) as error:
        return jsonify({"error": str(error)}), 404


@app.post("/api/annotation")
@admin_or_legacy_required
def save_dataset_annotation():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        recording = _request_recording(payload)
        incoming = payload.get("annotation", payload)
        if not isinstance(incoming, dict):
            raise ValueError("annotation must be an object")
        existing = _load_dataset_annotation(recording, session["user"])
        annotation = _normalized_dataset_annotation(
            existing, recording, session["user"]
        )
        incoming_states = _annotation_metric_states(incoming)
        if not incoming_states:
            raise ValueError("annotation has no supported metric state")
        for metric, state in incoming_states.items():
            if metric not in METRIC_TEMPLATES:
                raise ValueError(f"unsupported metric: {metric}")
            if not isinstance(state, dict) or not isinstance(state.get("frames", {}), dict):
                raise ValueError("frames must be an object keyed by frame index")
            normalized_state = {
                "t1": state.get("t1"),
                "t2": state.get("t2"),
                "frames": {},
            }
            for frame_key, item in state.get("frames", {}).items():
                if not isinstance(item, dict):
                    raise ValueError("each frame annotation must be an object")
                frame = int(item.get("frame_index", frame_key))
                if frame < 0 or frame >= recording["video"]["frame_count"]:
                    raise ValueError(f"frame out of range: {frame}")
                points = normalize_points(item.get("points_2d", {}))
                quality = item.get("quality", {})
                if not isinstance(quality, dict):
                    raise ValueError("frame quality must be an object")
                quality = dict(quality)
                quality["usable"] = bool(quality.get("usable", False))
                quality["mmpose_reviewed"] = bool(
                    quality.get("mmpose_reviewed", False)
                )
                quality["mmpose_agree"] = bool(
                    quality.get("mmpose_agree", False)
                )
                if quality["mmpose_agree"]:
                    quality["mmpose_reviewed"] = True
                normalized_state["frames"][str(frame)] = {
                    "frame_index": frame,
                    "rgb_timestamp_sec": frame / recording["video"]["frame_rate"],
                    "depth_timestamp": item.get("depth_timestamp"),
                    "points_2d": points,
                    "quality": quality,
                    "notes": str(item.get("notes", "")),
                }
            annotation["metrics"][metric] = normalized_state
        active_metric = incoming.get(
            "active_metric", incoming.get("metric", annotation["active_metric"])
        )
        if active_metric not in METRIC_TEMPLATES:
            raise ValueError(f"unsupported active metric: {active_metric}")
        annotation["active_metric"] = active_metric
        annotation["updated_at"] = time.time()
        path = _dataset_annotation_path(
            recording["subject"], recording["session_id"], recording["trial"], session["user"]
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(annotation, file, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary, path)
        return jsonify({"ok": True, "path": str(path), "annotation": annotation})
    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError) as error:
        return jsonify({"error": str(error)}), 400


# ---------- images ----------
@app.get("/api/images")
@admin_or_legacy_required
def list_images():
    user = session["user"]
    out = []
    for name in sorted(os.listdir(IMAGES_DIR)):
        if os.path.splitext(name)[1].lower() in ALLOWED_EXT:
            anno = _load_annotation(user, name)
            out.append({"name": name,
                        "annotated": bool(anno and (anno.get("angles") or anno.get("points"))),
                        "url": f"/api/image/{name}"})
    return jsonify({"images": out})


@app.get("/api/image/<path:name>")
@admin_or_legacy_required
def serve_image(name):
    return send_from_directory(IMAGES_DIR, name)


@app.post("/api/upload")
@admin_or_legacy_required
def upload():
    saved = []
    for f in request.files.getlist("files"):
        name = secure_filename(f.filename or "")
        if not name or os.path.splitext(name)[1].lower() not in ALLOWED_EXT:
            continue
        dest = os.path.join(IMAGES_DIR, name)
        if os.path.exists(dest):  # avoid clobbering
            stem, ext = os.path.splitext(name)
            name = f"{stem}_{int(time.time())}{ext}"
            dest = os.path.join(IMAGES_DIR, name)
        f.save(dest)
        saved.append(name)
    return jsonify({"saved": saved})


# ---------- annotations (per-user) ----------
def _user_annot_dir(user):
    d = os.path.join(ANNOT_DIR, secure_filename(user))
    os.makedirs(d, exist_ok=True)
    return d


def _anno_path(user, name):
    return os.path.join(_user_annot_dir(user), secure_filename(name) + ".json")


def _load_annotation(user, name):
    p = _anno_path(user, name)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return None


@app.get("/api/annotation/<path:name>")
@admin_or_legacy_required
def get_annotation(name):
    return jsonify(_load_annotation(session["user"], name) or {})


@app.post("/api/annotation/<path:name>")
@admin_or_legacy_required
def save_annotation(name):
    data = request.get_json(force=True, silent=True) or {}
    data["image"] = name
    data["user"] = session["user"]
    data["updated_at"] = time.time()
    path = _anno_path(session["user"], name)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    return jsonify({"ok": True, "path": path})


# ---------- export ----------
@app.get("/api/export.csv")
@admin_or_legacy_required
def export_csv():
    subject_name = request.args.get("subject")
    if subject_name:
        try:
            subject_name, subject_path = _subject_path(subject_name)
            requested_session = request.args.get("session_id")
            requested_trial = request.args.get("trial")
            annotation_root = subject_path / "annotations" / _safe_component(
                session["user"], "username"
            )
        except (FileNotFoundError, ValueError) as error:
            return jsonify({"error": str(error)}), 404
        point_names = sorted({
            point
            for points in METRIC_TEMPLATES.values()
            for point in points
        })
        buf = io.StringIO()
        fieldnames = [
            "username", "subject", "session_id", "trial", "metric",
            "frame_index", "rgb_timestamp_sec", "depth_timestamp", "usable",
            "mmpose_reviewed", "mmpose_agree", "notes",
        ] + [coordinate for point in point_names for coordinate in (f"{point}_x", f"{point}_y")]
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        if annotation_root.is_dir():
            paths = sorted(annotation_root.rglob("*.json"))
            for path in paths:
                annotation = json.loads(path.read_text(encoding="utf-8"))
                annotation_session = annotation.get("session_id")
                annotation_trial = annotation.get("trial")
                if requested_session and annotation_session != requested_session:
                    continue
                if requested_trial and annotation_trial != requested_trial:
                    continue
                metric_states = _annotation_metric_states(annotation)
                for metric, metric_state in metric_states.items():
                    for frame in sorted(
                        metric_state.get("frames", {}).values(),
                        key=lambda item: int(item.get("frame_index", 0)),
                    ):
                        row = {
                            "username": session["user"],
                            "subject": subject_name,
                            "session_id": annotation_session,
                            "trial": annotation_trial,
                            "metric": metric,
                            "frame_index": frame.get("frame_index"),
                            "rgb_timestamp_sec": frame.get("rgb_timestamp_sec"),
                            "depth_timestamp": frame.get("depth_timestamp"),
                            "usable": frame.get("quality", {}).get("usable", False),
                            "mmpose_reviewed": frame.get("quality", {}).get(
                                "mmpose_reviewed", False
                            ),
                            "mmpose_agree": frame.get("quality", {}).get(
                                "mmpose_agree", False
                            ),
                            "notes": frame.get("notes", ""),
                        }
                        for point, value in frame.get("points_2d", {}).items():
                            if point in point_names:
                                row[f"{point}_x"] = value.get("x")
                                row[f"{point}_y"] = value.get("y")
                        writer.writerow(row)
        return Response(
            buf.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=rgbd_annotations.csv"},
        )

    user = session["user"]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["user", "image", "angle_name", "vertex", "prox", "dist",
                "interior_deg", "reported_deg", "updated_at"])
    for fn in sorted(os.listdir(_user_annot_dir(user))):
        if not fn.endswith(".json"):
            continue
        anno = _load_annotation(user, fn[:-5])
        if not anno:
            continue
        for a in anno.get("angles", []):
            w.writerow([user, anno.get("image"), a.get("name"), a.get("vertex"),
                        a.get("prox"), a.get("dist"), a.get("interior_deg"),
                        a.get("reported_deg"), anno.get("updated_at")])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=rom_export.csv"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)
