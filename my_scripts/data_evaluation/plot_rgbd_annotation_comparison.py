#!/usr/bin/env python3
"""Plot manual RGBD and MMPose RGBD values on annotated frames."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

from my_scripts.data_evaluation.evaluate_rgbd_annotations import (
    evaluate_annotation,
    load_annotation,
    write_csv,
)
from my_scripts.data_evaluation.recordings import (
    annotation_path,
    find_recording,
    task_annotation_path,
)


def plot_comparison(rows: list[dict], summary: dict, output: str | None = None, show: bool = False) -> None:
    if output and not show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        raise ValueError("annotation contains no frames for the selected metric")
    time_values = np.asarray([row["rgb_timestamp_sec"] for row in rows], dtype=float)
    manual = np.asarray([
        row["manual_angle_deg"] if row["manual_angle_deg"] is not None else np.nan
        for row in rows
    ])
    mmpose = np.asarray([
        row["mmpose_angle_deg"] if row["mmpose_angle_deg"] is not None else np.nan
        for row in rows
    ])
    absolute_error = np.asarray([
        row["absolute_error_deg"] if row["absolute_error_deg"] is not None else np.nan
        for row in rows
    ])
    row_times = {
        int(row["frame_index"]): row["rgb_timestamp_sec"]
        for row in rows
    }

    figure, (angle_axis, error_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": (2.2, 1.0)},
    )
    angle_axis.plot(
        time_values,
        manual,
        "o-",
        color="#e76f51",
        linewidth=1.5,
        markersize=5,
        label="Manual RGBD",
    )
    angle_axis.plot(
        time_values,
        mmpose,
        "s--",
        color="#457b9d",
        linewidth=1.5,
        markersize=4,
        label="MMPose RGBD",
    )
    error_axis.plot(
        time_values,
        absolute_error,
        "o-",
        color="#2a9d8f",
        linewidth=1.5,
        markersize=5,
        label="Absolute difference",
    )
    for axis in (angle_axis, error_axis):
        axis.grid(alpha=0.3)
        for key in ("t1", "t2"):
            value = summary.get(key)
            x_value = row_times.get(int(value)) if value is not None else None
            if x_value is not None:
                axis.axvline(
                    x_value,
                    color="#6a4c93" if key == "t1" else "#1982c4",
                    linestyle="--",
                    linewidth=1.0,
                    label=key.upper(),
                )
    angle_axis.set_title(
        f"{summary.get('metric_label', summary.get('metric'))} - "
        f"{summary.get('mode')} - annotated frames only"
    )
    angle_axis.set_ylabel("degrees")
    angle_axis.legend(loc="best", fontsize=8)
    error_axis.set_xlabel("RGB timestamp (s)")
    error_axis.set_ylabel("absolute error")
    error_axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    if output:
        figure.savefig(output, dpi=150)
        print(f"plot written: {output}")
    if show or not output:
        plt.show()
    else:
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.environ.get("ROM_DATA_ROOT", "data/NUS/val"))
    parser.add_argument("--subject", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--annotation", default=None)
    parser.add_argument("--metric", default=None)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--task-review", action="store_true", help="Load the task-review annotation path")
    parser.add_argument("--mode", choices=("common_plane", "independent"), default="common_plane")
    parser.add_argument("--depth-sample-radius", type=int, default=2)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None, help="PNG output; omit to open an interactive plot")
    parser.add_argument("--csv-output", default=None, help="Optional per-frame CSV output")
    args = parser.parse_args()

    recording = find_recording(args.data_root, args.subject, args.session_id, args.trial)
    annotation_file = Path(args.annotation) if args.annotation else (
        task_annotation_path(recording, args.username)
        if args.task_review or args.task_id
        else annotation_path(recording, args.username)
    )
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
    if args.csv_output:
        write_csv(args.csv_output, rows)
    plot_comparison(rows, summary, output=args.output)
    print(f"frames: {summary['frames_with_valid_comparison']}/{summary['frames_evaluated']}")
    print(f"MAE: {summary['mae_deg']}")
    print(f"maximum absolute error: {summary['max_absolute_error_deg']}")


if __name__ == "__main__":
    main()
