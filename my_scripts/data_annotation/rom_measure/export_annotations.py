#!/usr/bin/env python3
"""Export ROM Measure 2 annotations as web-app-style comparison images.

The script is intended to be copied into the ROM Measure 2 application
directory, next to ``server.py``.  It deliberately reuses the application's
saved frame extraction and saved comparison payloads instead of running a
second MMPose pipeline.  This keeps the exported image aligned with what the
annotator saw in the browser.

Example (on the ROM2 server)::

    python export_annotations.py --username haziq

By default one PNG is written for every annotation whose task is still in
``tasks.json``.  The left panel is the side-view frame and the right panel is
the front-view frame.  MMPose landmarks are red, the annotator's landmarks
are green on the target view, and a comparison card is rendered below the
images.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ``server.py`` is intentionally imported from the ROM2 directory.  It owns
# the depth transform, exact-frame extraction, storage location, and schema
# used by the web application.
try:
    import server
except ImportError as error:  # pragma: no cover - gives a useful CLI error
    raise SystemExit(
        "Run this script from the ROM Measure 2 directory, or put that "
        "directory on PYTHONPATH so server.py can be imported."
    ) from error


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

# OpenCV colours are BGR.
BACKGROUND = (13, 23, 37)
PANEL_BACKGROUND = (22, 38, 60)
MUTED = (169, 184, 201)
WHITE = (245, 248, 252)
RED = (45, 62, 242)
GREEN = (55, 220, 95)
YELLOW = (90, 214, 248)
CYAN = (235, 208, 74)
BAD = (80, 80, 240)
GOOD = (70, 205, 125)
GUTTER = 18
DEPTH_CSS_OPACITY = 0.48


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("._") or "annotation"


def _point_xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    x = _finite_number(value.get("x"))
    y = _finite_number(value.get("y"))
    if x is None or y is None:
        return None
    return x, y


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _annotation_files(username: str) -> list[Path]:
    username_dir = server.secure_filename(username)
    if not username_dir:
        return []
    directory = server.ANNOTATIONS_DIR / username_dir
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.json"), key=lambda path: path.name.lower())


def _load_context(username: str) -> tuple[dict[str, dict], dict[str, dict], list[dict]]:
    tasks = {
        str(task.get("task_id")): task
        for task in server._load_tasks()
        if isinstance(task, dict) and task.get("task_id")
    }
    pairs = {
        str(pair.get("pair_id")): pair
        for pair in server._load_pairs()
        if isinstance(pair, dict) and pair.get("pair_id")
    }
    annotations = []
    for path in _annotation_files(username):
        annotation = _read_json(path)
        if annotation is not None:
            annotations.append({"path": path, "data": annotation})
    return tasks, pairs, annotations


def _decode_jpeg(payload: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("the application returned an unreadable frame")
    return image


def _apply_depth_overlay(
    image: np.ndarray,
    pair_entry: dict,
    frame_index: int,
    enabled: bool,
) -> tuple[np.ndarray, bool, str]:
    if not enabled:
        return image, False, "disabled"
    try:
        encoded = server._depth_color_png(pair_entry, int(frame_index))
        overlay = cv2.imdecode(
            np.frombuffer(encoded, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        return image, False, str(error)
    if overlay is None or overlay.ndim != 3 or overlay.shape[2] != 4:
        return image, False, "depth overlay is not RGBA"

    height, width = image.shape[:2]
    overlay = cv2.resize(overlay, (width, height), interpolation=cv2.INTER_NEAREST)
    alpha = overlay[:, :, 3].astype(np.float32) / 255.0
    alpha *= DEPTH_CSS_OPACITY
    alpha = alpha[:, :, None]
    foreground = overlay[:, :, :3].astype(np.float32)
    background = image.astype(np.float32)
    blended = np.clip(foreground * alpha + background * (1.0 - alpha), 0, 255)
    return blended.astype(np.uint8), True, "shown"


def _resize_panel(image: np.ndarray, panel_width: int) -> tuple[np.ndarray, float, float]:
    source_height, source_width = image.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise RuntimeError("frame has no image dimensions")
    scale = float(panel_width) / float(source_width)
    target_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(image, (int(panel_width), target_height), interpolation=cv2.INTER_AREA)
    return resized, scale, scale


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 2,
    outline: bool = True,
) -> None:
    if outline:
        cv2.putText(image, text, origin, FONT, scale, BACKGROUND, thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def _display_name(name: str) -> str:
    words = str(name).split("_")
    if len(words) > 1 and words[0] in {"left", "right"}:
        return f"{words[0][0].upper()} " + " ".join(words[1:])
    return " ".join(words)


def _scaled_xy(
    point: Any,
    scale_x: float,
    scale_y: float,
) -> tuple[int, int] | None:
    xy = _point_xy(point)
    if xy is None:
        return None
    return int(round(xy[0] * scale_x)), int(round(xy[1] * scale_y))


def _draw_polyline(
    image: np.ndarray,
    points: dict[str, Any],
    names: list[str],
    scale_x: float,
    scale_y: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    locations = [
        _scaled_xy(points.get(name), scale_x, scale_y)
        for name in names
    ]
    for first, second in zip(locations, locations[1:]):
        if first is not None and second is not None:
            cv2.line(image, first, second, BACKGROUND, thickness + 4, cv2.LINE_AA)
            cv2.line(image, first, second, color, thickness, cv2.LINE_AA)


def _draw_torso_quadrilateral(
    image: np.ndarray,
    points: dict[str, Any],
    scale_x: float,
    scale_y: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    names = ["left_shoulder", "right_shoulder", "right_hip", "left_hip", "left_shoulder"]
    locations = [
        _scaled_xy(points.get(name), scale_x, scale_y)
        for name in names
    ]
    for first, second in zip(locations, locations[1:]):
        if first is not None and second is not None:
            cv2.line(image, first, second, BACKGROUND, thickness + 4, cv2.LINE_AA)
            cv2.line(image, first, second, color, thickness, cv2.LINE_AA)


def _draw_pose(
    image: np.ndarray,
    pose: dict[str, Any] | None,
    scale_x: float,
    scale_y: float,
    color: tuple[int, int, int],
    label_prefix: str,
    draw_torso: bool = True,
) -> None:
    if not isinstance(pose, dict):
        return
    points = pose.get("points")
    if not isinstance(points, dict):
        return

    angle_points = pose.get("angle_points")
    if isinstance(angle_points, list):
        _draw_polyline(
            image,
            points,
            [str(name) for name in angle_points],
            scale_x,
            scale_y,
            color,
            max(2, int(round(image.shape[1] / 700))),
        )
    if draw_torso and all(name in points for name in (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    )):
        _draw_torso_quadrilateral(
            image,
            points,
            scale_x,
            scale_y,
            color,
            max(2, int(round(image.shape[1] / 1000))),
        )

    radius = max(6, int(round(image.shape[1] / 145)))
    label_scale = max(0.42, min(0.9, image.shape[1] / 1800.0))
    for name, value in points.items():
        location = _scaled_xy(value, scale_x, scale_y)
        if location is None:
            continue
        cv2.circle(image, location, radius + 3, BACKGROUND, -1, cv2.LINE_AA)
        cv2.circle(image, location, radius, color, -1, cv2.LINE_AA)
        label = f"{label_prefix} {_display_name(str(name))}"
        _put_text(
            image,
            label,
            (location[0] + radius + 5, location[1] - radius - 3),
            label_scale,
            color,
            max(1, int(round(label_scale * 2))),
        )


def _user_order(task: dict, annotation: dict) -> list[str]:
    required = task.get("required_points")
    if isinstance(required, list) and required:
        return [str(name) for name in required]
    angle_points = annotation.get("angle_points") or task.get("angle_points")
    if isinstance(angle_points, list):
        return [str(name) for name in angle_points]
    points = annotation.get("points")
    return list(points) if isinstance(points, dict) else []


def _user_labels(task: dict, order: list[str]) -> dict[str, str]:
    labels = task.get("annotation_labels")
    if isinstance(labels, list) and len(labels) == len(order):
        return {name: str(label) for name, label in zip(order, labels)}
    return {name: _display_name(name) for name in order}


def _draw_user_points(
    image: np.ndarray,
    task: dict,
    annotation: dict,
    scale_x: float,
    scale_y: float,
) -> None:
    points = annotation.get("points") or annotation.get("points_2d")
    if not isinstance(points, dict):
        return
    order = _user_order(task, annotation)
    labels = _user_labels(task, order)
    _draw_polyline(
        image,
        points,
        order,
        scale_x,
        scale_y,
        GREEN,
        max(3, int(round(image.shape[1] / 520))),
    )
    radius = max(8, int(round(image.shape[1] / 120)))
    label_scale = max(0.55, min(1.05, image.shape[1] / 1500.0))
    for name, value in points.items():
        location = _scaled_xy(value, scale_x, scale_y)
        if location is None:
            continue
        cv2.circle(image, location, radius + 4, BACKGROUND, -1, cv2.LINE_AA)
        cv2.circle(image, location, radius, GREEN, -1, cv2.LINE_AA)
        _put_text(
            image,
            labels.get(str(name), _display_name(str(name))),
            (location[0] + radius + 7, location[1] - radius - 5),
            label_scale,
            GREEN,
            max(2, int(round(label_scale * 2))),
        )


def _pose_for_view(annotation: dict, view: str) -> dict[str, Any] | None:
    direct_key = "side_mmpose" if view == "side" else "front_mmpose"
    direct = annotation.get(direct_key)
    if isinstance(direct, dict) and (direct.get("points") or direct.get("angle_deg") is not None):
        return direct
    target_view = str(annotation.get("view") or "side").lower()
    target = annotation.get("mmpose")
    if target_view == view and isinstance(target, dict):
        return target
    return direct if isinstance(direct, dict) else None


def _angle_for_view(annotation: dict, view: str, pose: dict | None) -> float | None:
    key = "side_mmpose_angle_deg" if view == "side" else "front_mmpose_angle_deg"
    angle = _finite_number(annotation.get(key))
    if angle is not None:
        return angle
    return _finite_number(pose.get("angle_deg")) if isinstance(pose, dict) else None


def _error_for_view(
    annotation: dict,
    view: str,
    manual_angle: float | None,
    mmpose_angle: float | None,
) -> float | None:
    key = "side_mmpose_error_deg" if view == "side" else "front_mmpose_error_deg"
    saved = _finite_number(annotation.get(key))
    if saved is not None:
        return saved
    if manual_angle is None or mmpose_angle is None:
        return None
    return abs(manual_angle - mmpose_angle)


def _format_angle(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}°"


def _format_pose_status(pose: dict | None, angle: float | None) -> str:
    if angle is not None:
        return _format_angle(angle)
    if not isinstance(pose, dict) or not pose:
        return "—"
    if pose.get("available") is False:
        return "unavailable"
    return "INVALID"


def _frame_timestamp(task: dict, view: str, frame_index: int) -> float | None:
    key = "left_timestamp_sec" if view == "side" else "right_timestamp_sec"
    value = _finite_number(task.get(key))
    if value is not None:
        return value
    video = task.get("pair", {}).get("left" if view == "side" else "right", {})
    fps = _finite_number(video.get("fps")) if isinstance(video, dict) else None
    return frame_index / fps if fps and fps > 0 else None


def _render_panel(
    image: np.ndarray,
    title: str,
    subtitle: str,
    pose: dict | None,
    task: dict,
    annotation: dict,
    show_user_points: bool,
    panel_width: int,
    depth_state: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    image, scale_x, scale_y = _resize_panel(image, panel_width)
    _draw_pose(image, pose, scale_x, scale_y, RED, "M", draw_torso=True)
    if show_user_points:
        _draw_user_points(image, task, annotation, scale_x, scale_y)

    header_height = 72
    footer_height = 74
    panel = np.full(
        (header_height + image.shape[0] + footer_height, image.shape[1], 3),
        PANEL_BACKGROUND,
        dtype=np.uint8,
    )
    _put_text(panel, title, (18, 30), 0.82, CYAN, 2)
    _put_text(panel, subtitle, (18, 57), 0.52, MUTED, 1)
    panel[header_height:header_height + image.shape[0], :] = image
    footer_y = header_height + image.shape[0]
    pose_angle = _finite_number(pose.get("angle_deg")) if isinstance(pose, dict) else None
    mmpose_text = f"MMPose 3D angle: {_format_pose_status(pose, pose_angle)}"
    _put_text(panel, mmpose_text, (18, footer_y + 29), 0.62, RED, 2)
    depth_text = f"Depth overlay: {depth_state}"
    _put_text(panel, depth_text, (18, footer_y + 57), 0.48, MUTED, 1)
    return panel, {
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "mmpose_angle_deg": pose_angle,
        "depth_overlay": depth_state,
    }


def _draw_card(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    value: str,
    label: str,
    value_color: tuple[int, int, int],
) -> None:
    cv2.rectangle(image, (x, y), (x + width, y + height), PANEL_BACKGROUND, -1)
    cv2.rectangle(image, (x, y), (x + width, y + height), (44, 68, 96), 2)
    _put_text(image, value, (x + 18, y + 48), 1.02, value_color, 2)
    _put_text(image, label, (x + 18, y + 78), 0.48, MUTED, 1)


def _render_comparison(
    task: dict,
    annotation: dict,
    left_panel: np.ndarray,
    right_panel: np.ndarray,
    left_meta: dict,
    right_meta: dict,
) -> tuple[np.ndarray, dict[str, Any]]:
    panel_height = max(left_panel.shape[0], right_panel.shape[0])
    def pad_panel(panel: np.ndarray) -> np.ndarray:
        if panel.shape[0] == panel_height:
            return panel
        padded = np.full((panel_height, panel.shape[1], 3), PANEL_BACKGROUND, dtype=np.uint8)
        padded[:panel.shape[0], :] = panel
        return padded

    left_panel = pad_panel(left_panel)
    right_panel = pad_panel(right_panel)
    content_width = left_panel.shape[1] + GUTTER + right_panel.shape[1]
    title_height = 92
    metrics_height = 190
    output = np.full(
        (title_height + panel_height + metrics_height, content_width, 3),
        BACKGROUND,
        dtype=np.uint8,
    )
    name = str(task.get("name") or annotation.get("name") or task.get("task_id") or "ROM annotation")
    status = str(annotation.get("status") or "saved").capitalize()
    pair_id = str(task.get("pair_id") or annotation.get("pair_id") or "")
    _put_text(output, name, (18, 36), 1.0, WHITE, 2)
    _put_text(output, f"Haziq · {status} · {pair_id}", (18, 68), 0.55, MUTED, 1)
    legend_x = max(18, content_width - 680)
    _put_text(output, "MMPose", (legend_x, 36), 0.55, RED, 2)
    _put_text(output, "Haziq annotation", (legend_x + 120, 36), 0.55, GREEN, 2)
    _put_text(output, "depth: blue near → red far · black = invalid", (legend_x, 68), 0.45, MUTED, 1)

    x_right = left_panel.shape[1] + GUTTER
    output[title_height:title_height + panel_height, :left_panel.shape[1]] = left_panel
    output[title_height:title_height + panel_height, x_right:x_right + right_panel.shape[1]] = right_panel

    manual_angle = _finite_number(annotation.get("angle_deg"))
    side_pose = _pose_for_view(annotation, "side")
    front_pose = _pose_for_view(annotation, "front")
    side_angle = _angle_for_view(annotation, "side", side_pose)
    front_angle = _angle_for_view(annotation, "front", front_pose)
    side_error = _error_for_view(annotation, "side", manual_angle, side_angle)
    front_error = _error_for_view(annotation, "front", manual_angle, front_angle)

    metrics_top = title_height + panel_height + 20
    gap = 10
    card_width = (content_width - 5 * gap) // 5
    cards = [
        (_format_angle(manual_angle), "Haziq annotated 3D angle", GREEN),
        (_format_pose_status(side_pose, side_angle), "MMPose 3D angle · side view", RED),
        (_format_pose_status(front_pose, front_angle), "MMPose 3D angle · front view", RED),
        (_format_angle(side_error), "absolute difference · annotated vs side", YELLOW),
        (_format_angle(front_error), "absolute difference · annotated vs front", YELLOW),
    ]
    for index, (value, label, color) in enumerate(cards):
        _draw_card(
            output,
            index * (card_width + gap),
            metrics_top,
            card_width,
            112,
            value,
            label,
            color,
        )

    metadata = {
        "task_id": task.get("task_id") or annotation.get("task_id"),
        "name": name,
        "status": annotation.get("status"),
        "pair_id": pair_id,
        "manual_angle_deg": manual_angle,
        "side_mmpose_angle_deg": side_angle,
        "front_mmpose_angle_deg": front_angle,
        "side_error_deg": side_error,
        "front_error_deg": front_error,
        "left_panel": left_meta,
        "right_panel": right_meta,
    }
    return output, metadata


def _task_for_annotation(annotation: dict, tasks: dict[str, dict]) -> dict | None:
    task_id = str(annotation.get("task_id") or "")
    return tasks.get(task_id)


def _render_one(
    task: dict,
    annotation: dict,
    pair: dict,
    panel_width: int,
    include_depth: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    left_frame = int(task["left_frame_index"])
    right_frame = int(task["right_frame_index"])
    left_image = _decode_jpeg(server._exact_frame_jpeg(pair, "left", left_frame))
    right_image = _decode_jpeg(server._exact_frame_jpeg(pair, "right", right_frame))
    left_image, left_depth, left_depth_message = _apply_depth_overlay(
        left_image,
        pair["left"],
        left_frame,
        include_depth,
    )
    right_image, right_depth, right_depth_message = _apply_depth_overlay(
        right_image,
        pair["right"],
        right_frame,
        include_depth,
    )

    target_view = str(task.get("view") or annotation.get("view") or "side").lower()
    left_pose = _pose_for_view(annotation, "side")
    right_pose = _pose_for_view(annotation, "front")
    left_user = target_view == "side"
    right_user = target_view == "front"
    left_timestamp = _frame_timestamp(task, "side", left_frame)
    right_timestamp = _frame_timestamp(task, "front", right_frame)
    left_subtitle = f"frame {left_frame}"
    right_subtitle = f"frame {right_frame}"
    if left_timestamp is not None:
        left_subtitle += f" · {left_timestamp:.3f} s"
    if right_timestamp is not None:
        right_subtitle += f" · {right_timestamp:.3f} s"
    left_panel, left_meta = _render_panel(
        left_image,
        "LEFT · SIDE VIEW",
        left_subtitle,
        left_pose,
        task,
        annotation,
        left_user,
        panel_width,
        "shown" if left_depth else left_depth_message,
    )
    right_panel, right_meta = _render_panel(
        right_image,
        "RIGHT · FRONT VIEW",
        right_subtitle,
        right_pose,
        task,
        annotation,
        right_user,
        panel_width,
        "shown" if right_depth else right_depth_message,
    )
    return _render_comparison(task, annotation, left_panel, right_panel, left_meta, right_meta)


def _default_output_dir(username: str) -> Path:
    return Path(server.STORAGE_DIR) / "exports" / _safe_filename(username)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ROM Measure 2 annotations as side-by-side PNG comparisons."
    )
    parser.add_argument("--username", default="haziq", help="annotation directory/user to export")
    parser.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="export only this task id; may be supplied more than once",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="output directory (default: <ROM2 storage>/exports/<username>)",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=1200,
        help="width of each rendered image panel in pixels (default: 1200)",
    )
    parser.add_argument(
        "--no-depth-overlay",
        action="store_true",
        help="do not include the browser-style depth colour overlay",
    )
    parser.add_argument(
        "--include-orphans",
        action="store_true",
        help="also try to export annotations whose task was deleted",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.panel_width < 240:
        raise SystemExit("--panel-width must be at least 240 pixels")

    tasks, pairs, annotations = _load_context(args.username)
    wanted = {str(task_id) for task_id in (args.task_ids or [])}
    output_dir = (args.output_dir or _default_output_dir(args.username)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    exported = 0
    skipped = 0
    for item in annotations:
        annotation = item["data"]
        task_id = str(annotation.get("task_id") or "")
        if wanted and task_id not in wanted:
            continue
        task = _task_for_annotation(annotation, tasks)
        if task is None:
            skipped += 1
            manifest.append({
                "task_id": task_id,
                "source_annotation": str(item["path"]),
                "status": "skipped",
                "reason": "task is no longer present in tasks.json",
            })
            if not args.include_orphans:
                continue
            print(f"SKIP {task_id}: task is no longer present in tasks.json")
            continue
        pair_id = str(task.get("pair_id") or annotation.get("pair_id") or "")
        pair = pairs.get(pair_id)
        if pair is None:
            skipped += 1
            print(f"SKIP {task_id}: pair {pair_id} is missing")
            manifest.append({
                "task_id": task_id,
                "source_annotation": str(item["path"]),
                "status": "skipped",
                "reason": f"pair {pair_id} is missing",
            })
            continue

        try:
            rendered, metadata = _render_one(
                task,
                annotation,
                pair,
                args.panel_width,
                not args.no_depth_overlay,
            )
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            skipped += 1
            print(f"SKIP {task_id}: {error}")
            manifest.append({
                "task_id": task_id,
                "source_annotation": str(item["path"]),
                "status": "skipped",
                "reason": str(error),
            })
            continue

        name = _safe_filename(str(task.get("name") or annotation.get("name") or task_id))
        output_path = output_dir / f"{_safe_filename(pair_id)}__{_safe_filename(task_id)}__{name}.png"
        if not cv2.imwrite(str(output_path), rendered):
            raise RuntimeError(f"could not write {output_path}")
        metadata.update({
            "status": "exported",
            "source_annotation": str(item["path"]),
            "output": str(output_path),
        })
        manifest.append(metadata)
        exported += 1
        print(
            f"OK {task_id}: {metadata['manual_angle_deg']}° manual -> "
            f"{output_path}"
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "username": args.username,
                "storage_dir": str(server.STORAGE_DIR),
                "depth_overlay": not args.no_depth_overlay,
                "panel_width": args.panel_width,
                "exported": exported,
                "skipped": skipped,
                "items": manifest,
            },
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")
    print(f"Exported {exported} annotation image(s) to {output_dir}")
    if skipped:
        print(f"Skipped {skipped} annotation(s); see {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
