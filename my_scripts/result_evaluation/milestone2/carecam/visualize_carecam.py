#!/usr/bin/env python3
"""Render one admin-selected timestamp from all repaired CareCam runs.

The script scans the six repaired iPad2 capture folders and the four CareCam
test directories in each folder (24 runs total).  For every run it:

1. seeks the RGB video to the selected timestamp,
2. loads the matching row from ``out.pkl`` or ``out.csv``, and
3. draws the 28-point CareCam 2-D skeleton on the RGB frame.

The output is one PNG per run.  By default the PNGs are written to this
directory, with names such as::

    030926_16-11_006_1061AD10__10m__t10.000s.png

Use one timestamp for every run::

    python visualize_carecam.py --timestamp 10.0

If the administrator selected a different timestamp for each run, pass a JSON
map instead::

    python visualize_carecam.py --timestamp-map admin_timestamps.json

The map may contain a common value, a folder value, or a folder/test value::

    {
      "timestamp_sec": 10.0,
      "030926_16-11_006_1061AD10": 8.5,
      "030926_16-15_007_4474C658/10m": 12.0
    }

With no timestamp option, the script automatically reads the admin selections
from ``/data/haziq/telept/data/milestone2/rom_measure/tasks.json`` and joins
them to the right-side videos through ``pairs.json``::

    python visualize_carecam.py

The current task file has 22 admin tasks.  Admin-task mode renders all four
CareCam test outputs for every task timestamp, so it writes 88 PNGs.

The default capture root is the repaired dataset used for the CareCam runs.
The script does not modify the capture folders or their downloaded results.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_CAPTURE_ROOT = Path(
    "/data/haziq/telept/data/milestone2/ipad2_repaired"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/haziq/telept/my_scripts/result_evaluation/milestone2/carecam"
)
DEFAULT_TASKS_JSON = Path(
    "/data/haziq/telept/data/milestone2/rom_measure/tasks.json"
)
DEFAULT_PAIRS_JSON = Path(
    "/data/haziq/telept/data/milestone2/rom_measure/pairs.json"
)
TESTS: tuple[tuple[str, str], ...] = (
    ("TUG", "TUG"),
    ("Walk-turn-walk", "Walk-turn-walk"),
    ("10m", "10m"),
    ("TUG3m", "TUG3m"),
)


# CareCam's 28-point pose convention.  This is the same convention used by
# the existing project visualizer in telept-ec2025026/Py_Haziq.
KEYPOINT_NAMES = (
    "R ear/eye", "L ear/eye", "nose", "neck", "chest",
    "R shoulder", "L shoulder", "R elbow", "L elbow",
    "R wrist", "L wrist", "R hip", "L hip", "hip center",
    "R knee (out)", "L knee (out)", "R knee (in)", "L knee (in)",
    "R ankle", "L ankle", "R midfoot", "L midfoot", "R heel", "L heel",
    "R toe tip", "L toe tip", "R outer toe", "L outer toe",
)

SKELETON_CONNECTIONS = (
    (0, 2), (1, 2), (2, 3), (3, 4), (4, 13),
    (3, 5), (3, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 13), (12, 13),
    (11, 14), (12, 15), (13, 16), (13, 17),
    (14, 16), (15, 17),
    (14, 18), (15, 19), (16, 20), (17, 21),
    (18, 20), (19, 21),
    (18, 22), (22, 24), (22, 26), (24, 26),
    (19, 23), (23, 25), (23, 27), (25, 27),
)

LEFT_INDICES = {1, 6, 8, 10, 12, 15, 17, 19, 21, 23, 25, 27}
RIGHT_INDICES = {0, 5, 7, 9, 11, 14, 16, 18, 20, 22, 24, 26}

# OpenCV uses BGR.  The two sides are intentionally distinct.
LEFT_BONE = (219, 144, 74)
RIGHT_BONE = (144, 219, 74)
CENTRE_BONE = (190, 190, 190)
LEFT_POINT = (255, 170, 70)
RIGHT_POINT = (80, 235, 110)
CENTRE_POINT = (100, 220, 245)
WHITE = (248, 248, 248)
BLACK = (15, 15, 15)
WARNING = (40, 175, 245)


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned.strip("._") or "run"


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _parse_timestamp(value: Any) -> float | None:
    number = _finite(value)
    if number is None or number < 0:
        return None
    return number


def _parse_array_text(value: Any) -> np.ndarray | None:
    """Parse pandas' NumPy-array string representation from out.csv."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"[]", "nan", "NaN", "None"}:
        return None

    # NumPy's string representation separates values with whitespace rather
    # than commas.  Removing brackets lets np.fromstring handle both regular
    # arrays and arrays containing NaN values.
    numbers = np.fromstring(text.replace("[", " ").replace("]", " "), sep=" ")
    if numbers.size == 0:
        return None
    if numbers.size % 3 == 0:
        return numbers.reshape((-1, 3))
    return numbers


def _load_rows_from_csv(path: Path, pose_column: str) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            frame_num = _finite(raw.get("frame_num"))
            if frame_num is None:
                continue
            frame = int(round(frame_num))
            rows[frame] = {
                "pose2d": _parse_array_text(raw.get(pose_column)),
                "pose3d": _parse_array_text(raw.get("person1_pose3D")),
            }
    return rows


def _load_rows(path: Path, pose_column: str) -> dict[int, dict[str, Any]]:
    """Load pose rows, preferring PKL when pandas is available.

    The normal workstation environment may not have pandas on PATH.  The CSV
    fallback keeps this visualization script usable without requiring a new
    environment installation.
    """
    pkl_path = path / "out.pkl"
    csv_path = path / "out.csv"

    try:
        import pandas as pd  # type: ignore

        dataframe = pd.read_pickle(pkl_path)
        if pose_column not in dataframe.columns:
            raise KeyError(f"missing {pose_column!r}")
        rows: dict[int, dict[str, Any]] = {}
        for _, record in dataframe.iterrows():
            frame_num = _finite(record.get("frame_num"))
            if frame_num is None:
                continue
            rows[int(round(frame_num))] = {
                "pose2d": record.get(pose_column),
                "pose3d": record.get("person1_pose3D"),
            }
        if rows:
            return rows
    except Exception as error:  # noqa: BLE001 - CSV is the intended fallback.
        print(f"  note: using out.csv for {path}: {error}")

    if not csv_path.is_file():
        raise FileNotFoundError(f"neither out.pkl nor out.csv exists in {path}")
    return _load_rows_from_csv(csv_path, pose_column)


def _select_row(rows: dict[int, dict[str, Any]], frame_num: int) -> dict[str, Any] | None:
    if frame_num in rows:
        return rows[frame_num]
    if not rows:
        return None
    closest = min(rows, key=lambda candidate: abs(candidate - frame_num))
    return rows[closest]


def _valid_xy(points: Any) -> np.ndarray | None:
    if points is None:
        return None
    try:
        array = np.asarray(points, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 2 or array.shape[1] < 2 or array.shape[0] == 0:
        return None
    if not np.isfinite(array[:, :2]).any():
        return None
    return array


def _bone_color(a: int, b: int) -> tuple[int, int, int]:
    if a in LEFT_INDICES and b in LEFT_INDICES:
        return LEFT_BONE
    if a in RIGHT_INDICES and b in RIGHT_INDICES:
        return RIGHT_BONE
    return CENTRE_BONE


def _point_color(index: int) -> tuple[int, int, int]:
    if index in LEFT_INDICES:
        return LEFT_POINT
    if index in RIGHT_INDICES:
        return RIGHT_POINT
    return CENTRE_POINT


def _draw_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float,
               color: tuple[int, int, int] = WHITE, thickness: int = 2) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                BLACK, thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def draw_keypoints(
    frame: np.ndarray,
    points: Any,
    *,
    output_height: int,
    labels: bool,
) -> tuple[np.ndarray, int]:
    """Resize a frame and draw the CareCam skeleton over it."""
    source_height, source_width = frame.shape[:2]
    if output_height > 0 and source_height != output_height:
        scale = output_height / float(source_height)
        output_width = max(1, int(round(source_width * scale)))
        image = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        image = frame.copy()

    pose = _valid_xy(points)
    if pose is None:
        return image, 0

    locations: list[tuple[int, int] | None] = []
    for point in pose:
        x = _finite(point[0])
        y = _finite(point[1])
        if x is None or y is None:
            locations.append(None)
        else:
            locations.append((int(round(x * scale)), int(round(y * scale))))

    thickness = max(2, int(round(image.shape[1] / 900)))
    radius = max(5, int(round(image.shape[1] / 180)))
    for first, second in SKELETON_CONNECTIONS:
        if first >= len(locations) or second >= len(locations):
            continue
        a, b = locations[first], locations[second]
        if a is None or b is None:
            continue
        cv2.line(image, a, b, BLACK, thickness + 4, cv2.LINE_AA)
        cv2.line(image, a, b, _bone_color(first, second), thickness, cv2.LINE_AA)

    drawn = 0
    for index, location in enumerate(locations):
        if location is None:
            continue
        drawn += 1
        color = _point_color(index)
        cv2.circle(image, location, radius + 3, BLACK, -1, cv2.LINE_AA)
        cv2.circle(image, location, radius, color, -1, cv2.LINE_AA)
        if labels:
            _draw_text(
                image,
                str(index),
                (location[0] + radius + 4, location[1] - radius - 2),
                max(0.45, min(0.9, image.shape[1] / 2200.0)),
                color,
                max(1, thickness),
            )
    return image, drawn


def _add_header(
    image: np.ndarray,
    *,
    folder: str,
    test: str,
    timestamp: float,
    video_frame: int,
    frame_num: int,
    drawn: int,
    pose_column: str,
) -> np.ndarray:
    header_height = max(100, int(round(image.shape[0] * 0.12)))
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], header_height), (8, 18, 30), -1)
    image = cv2.addWeighted(overlay, 0.78, image, 0.22, 0)
    title_scale = max(0.50, min(1.0, image.shape[1] / 2200.0))
    detail_scale = max(0.42, min(0.78, image.shape[1] / 2800.0))
    _draw_text(
        image,
        f"{folder}  |  {test}",
        (18, max(25, int(header_height * 0.27))),
        title_scale,
        WHITE,
        2,
    )
    _draw_text(
        image,
        f"t={timestamp:.3f}s  |  video frame={video_frame}  |  out frame_num={frame_num}",
        (18, max(52, int(header_height * 0.56))),
        detail_scale,
        WHITE,
        1,
    )
    status = (
        f"{pose_column} keypoints drawn={drawn}/28"
        if drawn
        else f"out frame_num={frame_num}  |  no valid {pose_column} keypoints at this timestamp"
    )
    _draw_text(
        image,
        status,
        (18, max(80, int(header_height * 0.84))),
        detail_scale,
        (160, 230, 255) if drawn else WARNING,
        1,
    )
    return image


def _load_timestamp_map(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("timestamp map must be a JSON object")
    return payload


def _load_admin_timestamp_records(
    tasks_path: Path,
    pairs_path: Path,
    capture_root: Path,
) -> list[dict[str, Any]]:
    """Resolve admin task timestamps to repaired iPad2 capture folders."""
    with tasks_path.open("r", encoding="utf-8") as handle:
        tasks_payload = json.load(handle)
    with pairs_path.open("r", encoding="utf-8") as handle:
        pairs_payload = json.load(handle)

    tasks = tasks_payload.get("tasks", []) if isinstance(tasks_payload, dict) else []
    pairs = pairs_payload.get("pairs", []) if isinstance(pairs_payload, dict) else []
    pairs_by_id = {
        str(pair.get("pair_id")): pair
        for pair in pairs
        if isinstance(pair, dict) and pair.get("pair_id")
    }

    records: list[dict[str, Any]] = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or "task")
        pair_id = str(task.get("pair_id") or "")
        pair = pairs_by_id.get(pair_id)
        right = pair.get("right", {}) if isinstance(pair, dict) else {}
        if not isinstance(right, dict):
            right = {}

        source_folder = str(right.get("source_folder") or "").strip()
        video_path = str(right.get("video_path") or "").strip()
        folder_name = (
            Path(source_folder).name
            if source_folder
            else Path(video_path).parent.name
        )
        if not folder_name:
            print(f"  warning: {task_id} has no right source folder", file=sys.stderr)
            continue

        timestamp = _parse_timestamp(task.get("right_timestamp_sec"))
        if timestamp is None:
            frame_index = _finite(task.get("right_frame_index"))
            frame_rate = _finite(right.get("frame_rate"))
            if frame_index is not None and frame_rate and frame_rate > 0:
                timestamp = float(frame_index) / frame_rate
        if timestamp is None:
            print(f"  warning: {task_id} has no usable right timestamp", file=sys.stderr)
            continue

        capture_folder = capture_root / folder_name
        if not capture_folder.is_dir():
            print(
                f"  warning: {task_id} maps to missing repaired folder "
                f"{capture_folder}",
                file=sys.stderr,
            )
            continue

        records.append({
            "task_id": task_id,
            "task_name": str(task.get("name") or "task"),
            "pair_id": pair_id,
            "folder_name": folder_name,
            "capture_folder": capture_folder,
            "timestamp": timestamp,
        })
    return records


def _resolve_timestamp(
    common_timestamp: float | None,
    timestamp_map: dict[str, Any] | None,
    folder: str,
    test: str,
) -> float | None:
    if timestamp_map is None:
        return common_timestamp

    candidates = (
        f"{folder}/{test}",
        f"{folder}__{test}",
        folder,
        test,
        "timestamp_sec",
        "timestamp",
        "*",
    )
    for key in candidates:
        if key not in timestamp_map:
            continue
        value = timestamp_map[key]
        if isinstance(value, dict):
            value = value.get("timestamp_sec", value.get("timestamp"))
        timestamp = _parse_timestamp(value)
        if timestamp is not None:
            return timestamp
    return common_timestamp


def _find_video(folder: Path) -> Path | None:
    videos = sorted(folder.glob("rgb_video_*.mp4"))
    return videos[0] if videos else None


def _render_run(
    *,
    folder: Path,
    test: str,
    timestamp: float,
    output_root: Path,
    pose_column: str,
    output_height: int,
    labels: bool,
    output_tag: str | None = None,
) -> tuple[bool, str]:
    run_dir = folder / "carecam" / test
    pkl_path = run_dir / "out.pkl"
    csv_path = run_dir / "out.csv"
    video_path = _find_video(folder)
    if video_path is None:
        return False, "missing rgb_video_*.mp4"
    if not pkl_path.is_file() and not csv_path.is_file():
        return False, "missing out.pkl and out.csv"

    rows = _load_rows(run_dir, pose_column)
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            return False, f"cannot open {video_path}"
        fps = _finite(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        video_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if video_count <= 0:
            return False, "video has no frames"
        requested_index = int(round(timestamp * fps))
        video_index = max(0, min(video_count - 1, requested_index))
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, f"cannot read video frame {video_index}"
    finally:
        cap.release()

    frame_num = video_index + 1
    row = _select_row(rows, frame_num) or {}
    pose = row.get("pose2d")
    image, drawn = draw_keypoints(
        frame,
        pose,
        output_height=output_height,
        labels=labels,
    )
    image = _add_header(
        image,
        folder=folder.name,
        test=test,
        timestamp=timestamp,
        video_frame=video_index,
        frame_num=frame_num,
        drawn=drawn,
        pose_column=pose_column,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    tag = f"{_safe_filename(output_tag)}__" if output_tag else ""
    output_name = (
        f"{tag}{_safe_filename(folder.name)}__{_safe_filename(test)}__"
        f"t{timestamp:.3f}s.png"
    )
    output_path = output_root / output_name
    if not cv2.imwrite(str(output_path), image):
        return False, f"could not write {output_path}"
    return True, f"{output_path} ({drawn}/28 keypoints)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--timestamp",
        type=float,
        help="common timestamp in seconds selected by the administrator",
    )
    group.add_argument(
        "--timestamp-map",
        type=Path,
        help="JSON file containing a common or per-run timestamp map",
    )
    group.add_argument(
        "--tasks-json",
        type=Path,
        nargs="?",
        const=DEFAULT_TASKS_JSON,
        help=(
            "use admin-selected right_timestamp_sec values from tasks.json; "
            "if omitted, this is the default mode"
        ),
    )
    parser.add_argument("--capture-root", type=Path, default=DEFAULT_CAPTURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pairs-json", type=Path, default=DEFAULT_PAIRS_JSON)
    parser.add_argument(
        "--pose-column",
        choices=("person1_pose2D", "person1_pose2D_raw"),
        default="person1_pose2D",
        help="pose field to draw (default: person1_pose2D)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1200,
        help="output image height in pixels; use 0 to keep source resolution",
    )
    parser.add_argument(
        "--labels",
        action="store_true",
        help="draw numeric keypoint labels in addition to the skeleton",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if (
        args.timestamp is None
        and args.timestamp_map is None
        and args.tasks_json is None
    ):
        args.tasks_json = DEFAULT_TASKS_JSON
    if args.timestamp is not None and _parse_timestamp(args.timestamp) is None:
        raise SystemExit("--timestamp must be a non-negative number of seconds")
    if args.height < 0:
        raise SystemExit("--height must be zero or a positive number")
    if not args.capture_root.is_dir():
        raise SystemExit(f"capture root does not exist: {args.capture_root}")

    timestamp_map = None
    if args.timestamp_map is not None:
        try:
            timestamp_map = _load_timestamp_map(args.timestamp_map)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"could not read timestamp map: {error}") from error

    admin_records = None
    if args.tasks_json is not None:
        try:
            admin_records = _load_admin_timestamp_records(
                args.tasks_json,
                args.pairs_json,
                args.capture_root,
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SystemExit(f"could not read admin task metadata: {error}") from error
        if not admin_records:
            raise SystemExit("no admin task timestamps could be resolved")
    else:
        folders = sorted(path for path in args.capture_root.iterdir() if path.is_dir())
        if not folders:
            raise SystemExit(f"no capture folders found in {args.capture_root}")

    success = 0
    failures = 0
    skipped = 0
    if admin_records is not None:
        expected = len(admin_records) * len(TESTS)
        print(
            f"Found {len(admin_records)} admin tasks and {expected} "
            "task/test images."
        )
    else:
        expected = len(folders) * len(TESTS)
        print(f"Found {len(folders)} capture folders and {expected} test runs.")
    print(f"Writing images to {args.output_root}")

    if admin_records is not None:
        run_specs = (
            (
                record["capture_folder"],
                test,
                record["timestamp"],
                f"{record['task_id']}__{record['task_name']}",
            )
            for record in admin_records
            for test, _ in TESTS
        )
    else:
        run_specs = (
            (
                folder,
                test,
                _resolve_timestamp(args.timestamp, timestamp_map, folder.name, test),
                None,
            )
            for folder in folders
            for test, _ in TESTS
        )

    for folder, test, timestamp, output_tag in run_specs:
        if timestamp is None:
            skipped += 1
            print(f"SKIP {folder.name}/{test}: no timestamp configured")
            continue
        try:
            ok, detail = _render_run(
                folder=folder,
                test=test,
                timestamp=timestamp,
                output_root=args.output_root,
                pose_column=args.pose_column,
                output_height=args.height,
                labels=args.labels,
                output_tag=output_tag,
            )
        except (OSError, ValueError, TypeError, KeyError) as error:
            ok, detail = False, str(error)
        if ok:
            success += 1
            print(f"OK   {folder.name}/{test}: {detail}")
        else:
            failures += 1
            print(f"FAIL {folder.name}/{test}: {detail}", file=sys.stderr)

    print(
        f"Finished: {success} images written, {failures} failures, "
        f"{skipped} skipped (expected {expected})."
    )
    return 0 if failures == 0 and skipped == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
