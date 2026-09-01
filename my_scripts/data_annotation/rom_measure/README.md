# rom_measure - RGBD keypoint annotation webapp

Annotate semantic 2D keypoints on exact RGB video frames, pair them with the
recording depth archive, and preview the shared RGBD angle calculation. Raw
annotations are saved per user beside the NUS dataset. The Python evaluator
then compares manual RGBD points against MMPose RGBD points.

The original still-image endpoints and demo files in `images/` remain available
for compatibility, but production NUS work uses the task browser.

## Roles and workflow

Configure two passwords on the server:

```bash
ROM_ADMIN_PASSWORD=use-a-private-admin-password
ROM_ANNOTATOR_PASSWORD=use-a-private-annotator-password
```

The username `admin` (or `ROM_ADMIN_USERNAME`) with the admin password opens
`/admin`, where fixed annotation tasks can be created. Any other username with
the annotator password opens `/`, where only assigned tasks are visible.

For backwards compatibility, `ROM_PASSWORD` remains an annotator-password
fallback when the new role variables are not configured. Do not use that legacy
fallback for a public deployment where task authoring must be protected.

The blind workflow is enforced by the server:

1. An annotator receives a predefined metric and exact frame.
2. MMPose data is hidden while semantic RGB points are placed.
3. `Finish annotation` locks the blind points and reveals the MMPose frame and
  RGBD comparison.
4. The annotator records agree or disagree, with an optional reason.

MMPose values are not returned by the task endpoints before step 3.

## Annotator workflow

- Log in with an annotator username and the annotator password.
- Choose an assigned task from the task list. The task already fixes the
  recording, exact frame, and metric type.
- Place the required semantic points in the RGB image with Space. Drag with the
  left or middle mouse button to pan a zoomed frame; `Reset all keypoints`
  clears the current task's manual placements, including both poses in a
  paired task. Depth is read from the paired depth frame.
- Save a draft if needed. MMPose remains hidden.
- Click `Finish annotation` when the blind points are ready. The server locks
  those points and reveals the synchronized MMPose frame and RGBD comparison.
- Select `Agree` or `Disagree`; add a reason when useful.
- Use the zoom controls to inspect either synchronized frame.

MMPose values are never returned by task endpoints before `Finish annotation`.
The webapp stores the blind points and review decision; the offline evaluator
recomputes the authoritative angles.

## Admin workflow

- Log in as `admin` with the admin password, then open `/admin`.
- Select a recording, scrub to the peak/exact frame, choose a metric, and add a
  task.
- **Plane-angle ROM tests** (shoulder/hip) use a **T1=neutral / T2=peak task**.
  `T1` is the neutral pose and only needs the working-side shoulder + hip; it
  defines the trunk down-axis (`down = side_hip - side_shoulder`). `T2` is the
  peak pose and supplies the moving limb. The ROM is the limb's elevation from
  the neutral down-axis, so the reference stays flat even if the person leans at
  the peak (a same-frame down-axis would tilt by 4-12°).
- **Hinge tests** (elbow/knee/ankle) are intrinsic included angles and
  single-frame; they need no neutral reference.
- Save the task manifest. External annotators then see only those predefined
  tasks and cannot choose another metric or arbitrary frame.

Supported ROM metrics (both sides). Plane-angle metrics need the neutral T1
reference (three working-side points split across two poses); hinge metrics are
single-frame:

| Region | Metric (right/left) | T1 (neutral) points | T2 (peak) points |
|---|---|---|---|
| Shoulder | flexion, extension, abduction, adduction, flexion/extension | shoulder, hip | shoulder, elbow |
| Elbow | flexion, extension | — (single frame) | shoulder, elbow, wrist |
| Hip | flexion | shoulder, hip | hip, knee |
| Knee | flexion, extension | — (single frame) | hip, knee, ankle |
| Ankle | dorsiflexion, plantarflexion | — (single frame) | knee, ankle, big-toe |

The test label (flexion vs extension vs abduction) plus the peak frame the
annotator picks determines the direction; the value is the limb elevation from
the neutral down axis (0 deg = hanging/neutral).

## Dataset layout

Set the root to the directory containing NUS subjects:

```text
<data-root>/
  <subject>/
    videos/<session_id>/<trial>.mp4
    depth/<session_id>/depth.zip
    camera_parameters/<session_id>/calibration.json
    mmpose/<model>/<session_id>/<trial>.json
    annotation_tasks/<session_id>/<trial>.json
    annotations/<username>/<session_id>/<trial>.json
```

For the current recording, annotations are stored at:

```text
/home/haziq/datasets/telept/data/NUS/val/
haziq_upperlimb_right_24082026/annotations/Haziq/
session_1787566918897/dynamic_03.json
```

The server equivalent is under `/data/haziq/telept/data/NUS/val/`.

## Run

From this directory:

```bash
python3 -m pip install -r requirements.txt
ROM_DATA_ROOT=/data/haziq/telept/data/NUS/val \
ROM_ADMIN_PASSWORD=admin-secret \
ROM_ANNOTATOR_PASSWORD=annotator-secret \
python3 server.py --host 0.0.0.0 --port 8090
```

Open `http://localhost:8090` for annotator tasks or `http://localhost:8090/admin`
for task authoring. Expose that port through the appropriate SSH tunnel when
needed. Remote port `8080` is occupied by another Dart service.

The default local root is inferred as `../../data/NUS/val` from the repository.
Set `ROM_DATA_ROOT` explicitly when running from another checkout.

## Annotation schema

Legacy free-form annotations use the schema below. Current task submissions use
the task-review shape described after the evaluator command.

```json
{
  "schema_version": 2,
  "annotation_type": "rgbd_keypoints",
  "subject": "haziq_upperlimb_right_24082026",
  "session_id": "session_1787566918897",
  "trial": "dynamic_03",
  "username": "Haziq",
  "active_metric": "right_shoulder_adduction",
  "metrics": {
    "right_shoulder_adduction": {
      "t1": 310,
      "t2": 394,
      "frames": {
        "310": {
          "frame_index": 310,
          "rgb_timestamp_sec": 10.33,
          "depth_timestamp": 106930.47,
          "points_2d": {
            "right_shoulder": {"x": 328.2, "y": 497.3}
          },
          "quality": {"usable": true},
          "notes": ""
        }
      }
    }
  }
}
```

Coordinates are original RGB pixels, not displayed CSS coordinates. The
annotation file is the source of truth. Depth-lifted 3D points and angles are
recomputed by `my_scripts/data_evaluation/evaluate_rgbd_annotations.py`.

## Evaluation

From the repository root:

```bash
python3 my_scripts/data_evaluation/evaluate_rgbd_annotations.py \
  --data-root /home/haziq/datasets/telept/data/NUS/val \
  --subject haziq_upperlimb_right_24082026 \
  --session-id session_1787566918897 \
  --trial dynamic_03 \
  --username Haziq \
  --mode common_plane \
  --output /tmp/rgbd_annotations.csv \
  --summary-output /tmp/rgbd_summary.json
```

`common_plane` uses manually annotated torso points for both manual and MMPose
arm points, isolating detector error. `independent` lets each point source
construct its own torso frame. Mocap comparison remains a separate
external-validity analysis.

For task-review annotations, evaluate one task or all submitted tasks:

```bash
python3 my_scripts/data_evaluation/evaluate_rgbd_annotations.py \
  --data-root /home/haziq/datasets/telept/data/NUS/val \
  --subject haziq_upperlimb_right_24082026 \
  --session-id session_1787566918897 \
  --trial dynamic_03 \
  --username Haziq \
  --task-id dynamic_03_right_elbow_flexion_394 \
  --mode common_plane \
  --output /tmp/rgbd_task.csv
```

The task annotation file stores the blind points separately from review data:

```text
<subject>/annotations/<username>/task_reviews/<session_id>/<trial>.json
```

Each single-frame task entry includes `blind_points_2d`, `status`,
`manual_angle_deg`, `mmpose_angle_deg`, `signed_error_deg`,
`mmpose_reviewed`, `mmpose_agree`, and an optional `disagreement_reason`.
Shoulder task entries are paired and instead contain `blind_points_2d_by_pose`
for `t1` and `t2`, plus `manual_delta_deg`, `mmpose_delta_deg`, and the signed
T1-to-T2 error. The original blind points are not replaced after MMPose is
revealed.

Admin task manifests are stored separately:

```json
{
  "schema_version": 1,
  "subject": "haziq_upperlimb_right_24082026",
  "session_id": "session_1787566918897",
  "trial": "dynamic_03",
  "tasks": [
    {
      "task_id": "dynamic_03_right_elbow_flexion_394",
      "metric": "right_elbow_flexion",
      "frame_index": 394,
      "enabled": true,
      "notes": ""
    }
  ]
}
```

They live at `<subject>/annotation_tasks/<session_id>/<trial>.json` and are
written only by the admin role.

For an annotated-versus-MMPose plot, use the separate report visualizer:

```bash
python3 my_scripts/data_evaluation/plot_rgbd_annotation_comparison.py \
  --data-root /home/haziq/datasets/telept/data/NUS/val \
  --subject haziq_upperlimb_right_24082026 \
  --session-id session_1787566918897 \
  --trial dynamic_03 \
  --username Haziq \
  --metric right_shoulder_adduction \
  --mode common_plane \
  --output /tmp/right_shoulder_adduction_comparison.png
```

The annotation UI is for frame-level QA. The report visualizer is for plotting
all annotated frames. The C3D viewer remains the external mocap comparison
tool.

## Multi-user and storage

Each username has its own namespace:
`annotations/<username>/<session_id>/<trial>.json`.

- Admin password: `ROM_ADMIN_PASSWORD=...`.
- Annotator password: `ROM_ANNOTATOR_PASSWORD=...`.
- Legacy shared annotator password: `ROM_PASSWORD=changeme`.
- Per-user passwords: `ROM_USERS="alice:pw1,bob:pw2"`.
- No password: set neither variable.

Do not commit raw videos, depth ZIPs, or production annotations. The repository
already ignores `data/`; use `rsync` or `scp` to transfer annotations between
the server and a local dataset copy.

## Limitations

- The current browser annotates RGB keypoints; it does not manually drag 3D
  points.
- Depth lens-distortion rectification is recorded as a pending calibration
  step and must be enabled before final accuracy claims.
- A 3-degree target should be reported only after checking annotation
  repeatability and the manual RGBD error floor.
