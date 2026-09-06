# rom_measure - RGBD keypoint annotation webapp

Annotate semantic 2D keypoints on exact RGB video frames, pair them with the
recording depth archive, and preview the shared RGBD angle calculation. Raw
annotations are saved per user beside the NUS dataset. The Python evaluator
then compares manual RGBD points against MMPose RGBD points.

The original still-image endpoints and demo files in `images/` remain available
for compatibility, but production NUS work uses the task browser. The server
and offline report tools in this directory include their own recording,
geometry, and evaluation helpers.

## Roles and workflow

Configure two passwords on the server:

```bash
ROM_ADMIN_PASSWORD=use-a-private-admin-password
ROM_ANNOTATOR_PASSWORD=use-a-private-annotator-password
```

For temporary localhost testing, this checkout currently uses `123` for both
admin and annotator passwords. Do not expose the service outside
`127.0.0.1` while this testing mode is enabled; restore the environment-based
passwords before deployment.

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
- Select a semantic point in the point list, then press Space over the RGB
  image to place it. Selecting a point does not advance automatically after
  placement. Drag with the left or middle mouse button to pan a zoomed frame;
  `Reset all keypoints` clears the current task's manual placements, including
  both poses in a paired task. Depth is read from the paired depth frame.
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
- Select a recording, scrub to the desired frame, choose a metric, and add a
  task. The frame index is the authoritative value; the matching video
  timestamp is shown beside it.
- **Paired segment-excursion tests** (shoulder and hip flexion, extension,
  abduction, and adduction) use a **T1=neutral/baseline / T2=peak task**. The
  same segment endpoints are annotated in both frames: shoulder-to-elbow for
  the shoulder and hip-to-knee for the hip. ROM is the angle between those two
  segment vectors.
- **Single-frame tests** (shoulder/hip internal or external rotation,
  elbow/knee flexion or extension, and ankle dorsiflexion/plantarflexion) need
  one fixed frame. Axial-rotation tests additionally use the nose, both
  shoulders, and both hips to construct a torso reference frame; the upper-arm,
  elbow, hip, and knee test postures should be standardized before annotating.
- Annotators see both T1 and T2 images side by side for paired tasks. The
  active panel is outlined and controls which pose the point list, notes, and
  quality fields refer to; point placements remain separate for T1 and T2.
- Save the task manifest. External annotators then see only those predefined
  tasks and cannot choose another metric or arbitrary frame.

Supported ROM metrics (both sides):

| Region | Metric (right/left) | Task mode | Required points |
|---|---|---|---|
| Shoulder | flexion, extension, abduction, adduction | T1/T2 | shoulder, elbow in each frame |
| Shoulder | internal/external rotation | single | nose, both shoulders, both hips, working elbow, wrist |
| Elbow | flexion, extension | single | shoulder, elbow, wrist |
| Hip | flexion, extension, abduction, adduction | T1/T2 | hip, knee in each frame |
| Hip | internal/external rotation | single | nose, both shoulders, both hips, working knee, ankle |
| Knee | flexion, extension | single | hip, knee, ankle |
| Ankle | dorsiflexion, plantarflexion | single | knee, ankle, big-toe |

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

### Current local milestone2 setup

From this directory, stop any older server with `Ctrl+C`, then run:

```bash
ROM_DATA_ROOT=/home/haziq/datasets/telept/data/milestone2/val \
python3 server.py --host 127.0.0.1 --port 8091
```

Open `http://127.0.0.1:8091/admin` for admin task authoring or
`http://127.0.0.1:8091` for annotator tasks. In the current temporary
localhost testing mode, the admin username is `admin` and both admin and
annotator passwords are `123`.

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
recomputed by `my_scripts/data_annotation/rom_measure/evaluate_rgbd_annotations.py`.

## Evaluation

From the repository root:

```bash
python3 my_scripts/data_annotation/rom_measure/evaluate_rgbd_annotations.py \
  --data-root /home/haziq/datasets/telept/data/NUS/val \
  --subject haziq_upperlimb_right_24082026 \
  --session-id session_1787566918897 \
  --trial dynamic_03 \
  --username Haziq \
  --mode common_plane \
  --output /tmp/rgbd_annotations.csv \
  --summary-output /tmp/rgbd_summary.json
```

For single-frame axial-rotation tasks, `common_plane` uses manually annotated
torso points for both manual and MMPose calculations, isolating limb-detector
error. `independent` lets each point source construct its own torso frame.
Mocap comparison remains a separate external-validity analysis.

For task-review annotations, evaluate one task or all submitted tasks:

```bash
python3 my_scripts/data_annotation/rom_measure/evaluate_rgbd_annotations.py \
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
Paired shoulder/hip segment entries instead contain
`blind_points_2d_by_pose` for `t1` and `t2`, plus `manual_delta_deg`,
`mmpose_delta_deg`, and the signed T1-to-T2 error. The original blind points
are not replaced after MMPose is revealed.

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
python3 my_scripts/data_annotation/rom_measure/plot_rgbd_annotation_comparison.py \
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
