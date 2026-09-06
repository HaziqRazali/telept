#!/usr/bin/env python3
"""Evaluate human RGBD keypoint annotations against MMPose RGBD points."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[3]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

if __package__ in {None, ""}:
    from recordings import (
        annotation_path,
        find_recording,
        task_annotation_path,
    )
    from rgbd_geometry import (
        METRIC_LABELS,
        METRIC_TEMPLATES,
        DepthZipReader,
        compute_metric,
        compute_paired_metric,
        lift_points_2d,
        load_mmpose_json,
        metric_uses_torso_frame,
        mmpose_points_for_frame,
        normalize_points,
    )
else:
    from .recordings import (
        annotation_path,
        find_recording,
        task_annotation_path,
    )
    from .rgbd_geometry import (
        METRIC_LABELS,
        METRIC_TEMPLATES,
        DepthZipReader,
        compute_metric,
        compute_paired_metric,
        lift_points_2d,
        load_mmpose_json,
        metric_uses_torso_frame,
        mmpose_points_for_frame,
        normalize_points,
    )


def evaluate_annotation(
    annotation: dict,
    recording,
    mode: str = "common_plane",
    depth_sample_radius: int = 2,
    score_threshold: float = 0.5,
    metric: str | None = None,
    task_id: str | None = None,
) -> tuple[list[dict], dict]:
    if isinstance(annotation.get("tasks"), dict):
        return evaluate_task_document(
            annotation,
            recording,
            mode=mode,
            depth_sample_radius=depth_sample_radius,
            score_threshold=score_threshold,
            task_id=task_id,
            metric=metric,
        )
    metric, metric_state = annotation_metric_state(annotation, metric)
    if mode not in {"common_plane", "independent"}:
        raise ValueError(f"unsupported evaluation mode: {mode}")
    if recording.mmpose_path is None:
        raise FileNotFoundError("MMPose JSON was not found for this recording")
    labels, keypoints, scores = load_mmpose_json(str(recording.mmpose_path), score_threshold)
    depth_reader = DepthZipReader(str(recording.depth_path)) if recording.depth_path else None
    if depth_reader is None:
        raise FileNotFoundError("depth archive was not found for this recording")
    rows = []
    try:
        for frame_key, frame_annotation in sorted(
            metric_state.get("frames", {}).items(),
            key=lambda item: int(item[1].get("frame_index", item[0])),
        ):
            frame = int(frame_annotation.get("frame_index", frame_key))
            if frame < 0 or frame >= recording.video.frame_count:
                continue
            manual_2d = normalize_points(frame_annotation.get("points_2d", {}))
            mmpose_2d = mmpose_points_for_frame(
                labels, keypoints, scores, frame, score_threshold
            )
            manual_3d = lift_points_2d(
                manual_2d,
                depth_reader,
                frame,
                recording.video.width,
                recording.video.height,
                depth_sample_radius,
            )
            mmpose_3d = lift_points_2d(
                mmpose_2d,
                depth_reader,
                frame,
                recording.video.width,
                recording.video.height,
                depth_sample_radius,
            )
            manual_angle = compute_metric(manual_3d.points, metric)
            if mode == "common_plane":
                mmpose_points_3d = dict(mmpose_3d.points)
                for name in _torso_points(metric):
                    if name in manual_3d.points:
                        mmpose_points_3d[name] = manual_3d.points[name]
                mmpose_angle = compute_metric(mmpose_points_3d, metric)
            else:
                mmpose_angle = compute_metric(mmpose_3d.points, metric)
            missing_manual = _missing_points(manual_3d.points, metric)
            missing_mmpose = _missing_points(mmpose_3d.points, metric)
            signed_error = (
                float(mmpose_angle - manual_angle)
                if np.isfinite([mmpose_angle, manual_angle]).all()
                else np.nan
            )
            rows.append({
                "subject": recording.subject,
                "session_id": recording.session_id,
                "trial": recording.trial,
                "username": annotation.get("username", ""),
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "mode": mode,
                "frame_index": frame,
                "rgb_timestamp_sec": frame / recording.video.frame_rate,
                "depth_timestamp": manual_3d.depth_timestamp,
                "manual_angle_deg": _csv_number(manual_angle),
                "mmpose_angle_deg": _csv_number(mmpose_angle),
                "signed_error_deg": _csv_number(signed_error),
                "absolute_error_deg": _csv_number(abs(signed_error)),
                "manual_valid": not missing_manual and np.isfinite(manual_angle),
                "mmpose_valid": not missing_mmpose and np.isfinite(mmpose_angle),
                "missing_manual_points": ",".join(missing_manual),
                "missing_mmpose_points": ",".join(missing_mmpose),
                "missing_manual_depth": ",".join(_missing_depth(manual_3d.depths)),
                "missing_mmpose_depth": ",".join(_missing_depth(mmpose_3d.depths)),
                "usable": bool(frame_annotation.get("quality", {}).get("usable", False)),
                "mmpose_reviewed": bool(
                    frame_annotation.get("quality", {}).get("mmpose_reviewed", False)
                ),
                "mmpose_agree": bool(
                    frame_annotation.get("quality", {}).get("mmpose_agree", False)
                ),
                "notes": frame_annotation.get("notes", ""),
            })
    finally:
        depth_reader.close()
    summary = summarize(rows, annotation, metric_state)
    return rows, summary


def _paired_segment_rom(
    recording,
    metric: str,
    entry: dict,
    depth_sample_radius: int = 2,
    score_threshold: float = 0.5,
) -> tuple[float | None, float | None]:
    """ROM for a T1=neutral / T2=peak segment-excursion task."""
    t1 = entry.get("poses", {}).get("t1", {})
    t2 = entry.get("poses", {}).get("t2", {})
    t1_frame = t1.get("frame_index")
    if t1_frame is None:
        t1_frame = entry.get("t1_frame_index")
    t2_frame = t2.get("frame_index")
    if t2_frame is None:
        t2_frame = entry.get("t2_frame_index")
    points_by_pose = entry.get("blind_points_2d_by_pose", {})
    if t1_frame is None or t2_frame is None:
        return None, None
    if recording.depth_path is None:
        raise FileNotFoundError("depth archive was not found for this recording")
    depth_reader = DepthZipReader(str(recording.depth_path))
    try:
        t1_lifted = lift_points_2d(
            normalize_points(points_by_pose.get("t1", {})),
            depth_reader,
            int(t1_frame),
            recording.video.width,
            recording.video.height,
            depth_sample_radius,
        )
        t2_lifted = lift_points_2d(
            normalize_points(points_by_pose.get("t2", {})),
            depth_reader,
            int(t2_frame),
            recording.video.width,
            recording.video.height,
            depth_sample_radius,
        )
        manual_rom = compute_paired_metric(
            t1_lifted.points,
            t2_lifted.points,
            metric,
        )
        mmpose_rom = None
        if recording.mmpose_path is not None:
            labels, keypoints, scores = load_mmpose_json(
                str(recording.mmpose_path), score_threshold
            )
            mmpose_t1_2d = mmpose_points_for_frame(
                labels, keypoints, scores, int(t1_frame), score_threshold
            )
            mmpose_t2_2d = mmpose_points_for_frame(
                labels, keypoints, scores, int(t2_frame), score_threshold
            )
            mmpose_t1_lifted = lift_points_2d(
                mmpose_t1_2d,
                depth_reader,
                int(t1_frame),
                recording.video.width,
                recording.video.height,
                depth_sample_radius,
            )
            mmpose_t2_lifted = lift_points_2d(
                mmpose_t2_2d,
                depth_reader,
                int(t2_frame),
                recording.video.width,
                recording.video.height,
                depth_sample_radius,
            )
            mmpose_rom = compute_paired_metric(
                mmpose_t1_lifted.points,
                mmpose_t2_lifted.points,
                metric,
            )
    finally:
        depth_reader.close()
    return _csv_number(manual_rom), _csv_number(mmpose_rom)


def evaluate_task_document(
    annotation: dict,
    recording,
    mode: str = "common_plane",
    depth_sample_radius: int = 2,
    score_threshold: float = 0.5,
    task_id: str | None = None,
    metric: str | None = None,
) -> tuple[list[dict], dict]:
    """Evaluate task submissions, optionally selecting one task ID."""
    tasks = annotation.get("tasks", {})
    if not isinstance(tasks, dict):
        raise ValueError("task annotation tasks must be an object")
    metric_filter = metric
    selected = []
    for current_id, entry in tasks.items():
        if task_id and current_id != task_id:
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("status", "draft") not in {"submitted", "reviewed"}:
            continue
        task_metric = entry.get("metric")
        if task_metric not in METRIC_TEMPLATES or (
            metric_filter and task_metric != metric_filter
        ):
            continue
        if entry.get("frame_mode") == "paired" or "blind_points_2d_by_pose" in entry:
            selected.append((current_id, entry, task_metric, None))
        else:
            frame = entry.get("frame_index")
            if frame is None:
                continue
            selected.append((current_id, entry, task_metric, int(frame)))
    if not selected:
        requested = f" {task_id}" if task_id else ""
        raise ValueError(f"no evaluable task entries found{requested}")

    rows = []
    paired_deltas = []
    for current_id, entry, metric, frame in selected:
        if frame is None:
            pose_definitions = entry.get("poses", {})
            t1_definition = pose_definitions.get("t1", {})
            t2_definition = pose_definitions.get("t2", {})
            t1_frame = t1_definition.get("frame_index")
            t2_frame = t2_definition.get("frame_index")
            if t1_frame is None:
                t1_frame = entry.get("t1_frame_index")
            if t2_frame is None:
                t2_frame = entry.get("t2_frame_index")
            if t1_frame is None or t2_frame is None:
                raise ValueError(f"task {current_id} has no T1/T2 frames")
            manual_rom, mmpose_rom = _paired_segment_rom(
                recording,
                metric,
                entry,
                depth_sample_radius=depth_sample_radius,
                score_threshold=score_threshold,
            )
            rom_error = _difference(mmpose_rom, manual_rom)
            quality_by_pose = entry.get("quality_by_pose", {})
            notes_by_pose = entry.get("notes_by_pose", {})
            rows.append({
                "subject": recording.subject,
                "session_id": recording.session_id,
                "trial": recording.trial,
                "username": annotation.get("username", ""),
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "mode": "paired_segment_excursion",
                "frame_mode": "paired",
                "frame_index": int(t2_frame),
                "t1_frame_index": int(t1_frame),
                "t2_frame_index": int(t2_frame),
                "rgb_timestamp_sec": int(t2_frame) / recording.video.frame_rate,
                "depth_timestamp": None,
                "manual_angle_deg": manual_rom,
                "mmpose_angle_deg": mmpose_rom,
                "signed_error_deg": rom_error,
                "absolute_error_deg": (
                    abs(rom_error) if rom_error is not None else None
                ),
                "manual_valid": manual_rom is not None,
                "mmpose_valid": mmpose_rom is not None,
                "missing_manual_points": "",
                "missing_mmpose_points": "",
                "missing_manual_depth": "",
                "missing_mmpose_depth": "",
                "usable": bool(
                    quality_by_pose.get("t1", {}).get("usable", False)
                    and quality_by_pose.get("t2", {}).get("usable", False)
                ),
                "mmpose_reviewed": bool(entry.get("mmpose_reviewed", False)),
                "mmpose_agree": bool(entry.get("mmpose_agree", False)),
                "notes": " / ".join(
                    note for note in (
                        notes_by_pose.get("t1", ""),
                        notes_by_pose.get("t2", ""),
                    ) if note
                ),
            })
            paired_deltas.append({
                "task_id": current_id,
                "metric": metric,
                "t1_frame_index": int(t1_frame),
                "t2_frame_index": int(t2_frame),
                "manual_rom_deg": manual_rom,
                "mmpose_rom_deg": mmpose_rom,
                "manual_t1_angle_deg": None,
                "manual_t2_angle_deg": None,
                "manual_delta_deg": manual_rom,
                "mmpose_t1_angle_deg": None,
                "mmpose_t2_angle_deg": None,
                "mmpose_delta_deg": mmpose_rom,
                "signed_error_deg": rom_error,
                "absolute_error_deg": abs(rom_error) if rom_error is not None else None,
                "mmpose_reviewed": bool(entry.get("mmpose_reviewed", False)),
                "mmpose_agree": bool(entry.get("mmpose_agree", False)),
            })
            continue
        quality = dict(entry.get("quality", {}))
        quality["mmpose_reviewed"] = bool(entry.get("mmpose_reviewed", False))
        quality["mmpose_agree"] = bool(entry.get("mmpose_agree", False))
        frame_annotation = {
            "frame_index": frame,
            "points_2d": entry.get("blind_points_2d", {}),
            "quality": quality,
            "notes": entry.get("notes", ""),
        }
        metric_annotation = {
            "username": annotation.get("username", ""),
            "metric": metric,
            "frames": {str(frame): frame_annotation},
        }
        task_rows, _ = evaluate_annotation(
            metric_annotation,
            recording,
            mode=mode,
            depth_sample_radius=depth_sample_radius,
            score_threshold=score_threshold,
            metric=metric,
        )
        for row in task_rows:
            row["task_id"] = current_id
        rows.extend(task_rows)
    summary = summarize(rows, annotation)
    summary["task_count"] = len(selected)
    summary["task_ids"] = [current_id for current_id, *_ in selected]
    summary["paired_task_deltas"] = paired_deltas
    return rows, summary


def summarize(
    rows: list[dict],
    annotation: dict | None = None,
    metric_state: dict | None = None,
) -> dict:
    errors = np.asarray(
        [row["absolute_error_deg"] for row in rows if row["absolute_error_deg"] is not None],
        dtype=float,
    )
    signed = np.asarray(
        [row["signed_error_deg"] for row in rows if row["signed_error_deg"] is not None],
        dtype=float,
    )
    manual_valid = sum(bool(row["manual_valid"]) for row in rows)
    mmpose_valid = sum(bool(row["mmpose_valid"]) for row in rows)
    reviewed = sum(bool(row["mmpose_reviewed"]) for row in rows)
    agreed = sum(bool(row["mmpose_agree"]) for row in rows)
    summary = {
        "metric": rows[0]["metric"] if rows else (annotation or {}).get("metric"),
        "metric_label": rows[0]["metric_label"] if rows else None,
        "mode": rows[0]["mode"] if rows else None,
        "frames_evaluated": len(rows),
        "frames_with_valid_manual_angle": manual_valid,
        "frames_with_valid_mmpose_angle": mmpose_valid,
        "frames_with_valid_comparison": int(errors.size),
        "frames_mmpose_reviewed": reviewed,
        "frames_mmpose_agreed": agreed,
        "mmpose_agreement_rate_reviewed": (
            _json_number(agreed / reviewed) if reviewed else None
        ),
        "mae_deg": _json_number(np.mean(errors)) if errors.size else None,
        "median_absolute_error_deg": _json_number(np.median(errors)) if errors.size else None,
        "rmse_deg": _json_number(np.sqrt(np.mean(errors ** 2))) if errors.size else None,
        "bias_deg": _json_number(np.mean(signed)) if signed.size else None,
        "p95_absolute_error_deg": _json_number(np.percentile(errors, 95)) if errors.size else None,
        "max_absolute_error_deg": _json_number(np.max(errors)) if errors.size else None,
        "correlation": _correlation(rows),
        "t1": (metric_state or {}).get("t1", (annotation or {}).get("t1")),
        "t2": (metric_state or {}).get("t2", (annotation or {}).get("t2")),
    }
    return summary


def load_annotation(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as file:
        annotation = json.load(file)
    required = ("subject", "session_id", "trial")
    missing = [name for name in required if name not in annotation]
    if missing:
        raise ValueError(f"annotation missing fields: {', '.join(missing)}")
    if "tasks" in annotation:
        if not isinstance(annotation["tasks"], dict):
            raise ValueError("task annotation tasks must be an object")
    elif "metrics" in annotation:
        if not isinstance(annotation["metrics"], dict):
            raise ValueError("annotation metrics must be an object")
    elif "metric" not in annotation or not isinstance(annotation.get("frames"), dict):
        raise ValueError("annotation must contain metrics or metric and frames")
    return annotation


def annotation_metric_state(
    annotation: dict,
    metric: str | None = None,
) -> tuple[str, dict]:
    metric = metric or annotation.get("active_metric") or annotation.get("metric")
    metrics = annotation.get("metrics")
    if isinstance(metrics, dict):
        if metric not in METRIC_TEMPLATES:
            raise ValueError(f"annotation has unsupported metric: {metric}")
        state = metrics.get(metric)
        if not isinstance(state, dict):
            raise ValueError(f"annotation has no state for metric: {metric}")
        if not isinstance(state.get("frames", {}), dict):
            raise ValueError("annotation frames must be an object")
        return metric, state
    if metric not in METRIC_TEMPLATES:
        raise ValueError(f"annotation has unsupported metric: {metric}")
    return metric, {
        "t1": annotation.get("t1"),
        "t2": annotation.get("t2"),
        "frames": annotation.get("frames", {}),
    }


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(
        key for row in rows for key in row
    )) if rows else [
        "subject", "session_id", "trial", "username", "metric", "metric_label",
        "mode", "frame_index", "rgb_timestamp_sec", "depth_timestamp",
        "manual_angle_deg", "mmpose_angle_deg", "signed_error_deg",
        "absolute_error_deg", "manual_valid", "mmpose_valid",
        "missing_manual_points", "missing_mmpose_points", "missing_manual_depth",
        "missing_mmpose_depth", "usable", "mmpose_reviewed", "mmpose_agree",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _torso_points(metric: str) -> tuple[str, ...]:
    if not metric_uses_torso_frame(metric):
        return ()
    return (
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    )


def _missing_points(points: dict[str, np.ndarray], metric: str) -> list[str]:
    return [
        name for name in METRIC_TEMPLATES[metric]
        if name not in points or not np.isfinite(points[name]).all()
    ]


def _missing_depth(depths: dict[str, float]) -> list[str]:
    return [name for name, value in depths.items() if not np.isfinite(value)]


def _correlation(rows: list[dict]) -> float | None:
    valid = [
        row for row in rows
        if row["manual_angle_deg"] is not None and row["mmpose_angle_deg"] is not None
    ]
    if len(valid) < 2:
        return None
    manual = np.asarray([row["manual_angle_deg"] for row in valid], dtype=float)
    mmpose = np.asarray([row["mmpose_angle_deg"] for row in valid], dtype=float)
    if np.ptp(manual) < 1e-12 or np.ptp(mmpose) < 1e-12:
        return None
    return _json_number(np.corrcoef(manual, mmpose)[0, 1])


def _json_number(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _difference(second, first):
    if second is None or first is None:
        return None
    return float(second - first)


def _csv_number(value) -> float | None:
    return _json_number(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("ROM_DATA_ROOT", "data/NUS/val"))
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--annotation", default=None, help="Override the dataset-local annotation path")
    parser.add_argument("--metric", default=None, choices=tuple(METRIC_TEMPLATES), help="Metric state to evaluate; defaults to active_metric")
    parser.add_argument("--task-id", default=None, help="Task ID to evaluate from a task-review annotation file")
    parser.add_argument("--task-review", action="store_true", help="Load the task-review annotation path instead of the legacy free-form path")
    parser.add_argument("--mode", choices=("common_plane", "independent"), default="common_plane")
    parser.add_argument("--depth-sample-radius", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="Per-frame CSV path")
    parser.add_argument("--summary-output", default=None, help="Summary JSON path")
    args = parser.parse_args()

    recording = find_recording(args.data_root, args.subject, args.session_id, args.trial)
    annotation_file = Path(args.annotation) if args.annotation else (
        task_annotation_path(recording, args.username)
        if args.task_review or args.task_id
        else annotation_path(recording, args.username)
    )
    if not annotation_file.is_file():
        raise FileNotFoundError(f"annotation file not found: {annotation_file}")
    annotation = load_annotation(annotation_file)
    rows, summary = evaluate_annotation(
        annotation,
        recording,
        mode=args.mode,
        depth_sample_radius=args.depth_sample_radius,
        score_threshold=args.score_threshold,
        metric=args.metric,
        task_id=args.task_id,
    )
    if args.output:
        write_csv(args.output, rows)
    if args.summary_output:
        path = Path(args.summary_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
