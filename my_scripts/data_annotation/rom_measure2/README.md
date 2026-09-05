# rom_measure2

`rom_measure2` is a small paired-video ROM annotation app. It is separate from
the original `rom_measure` app and keeps its own uploads, task index, and
annotations by default under:

```text
/home/haziq/datasets/telept/data/milestone2/rom_measure2/
```

On the first restart after this location change, an existing legacy
`rom_measure2/data/` directory is moved there automatically.

## Workflow

The admin signs in at `/admin` and:

1. imports one left session folder and one right/reference session folder (or
   uploads a pair of videos as a fallback);
2. scrubs the two videos independently with their sliders;
3. chooses **Right shoulder flexion**, **Left shoulder flexion**, **Right
   shoulder extension**, **Right elbow flexion**, **Right knee flexion**,
   **Right knee extension**,
   **Right shoulder abduction**, **Right shoulder internal rotation**,
   **Right shoulder external rotation**, **Right hip abduction**, **Right hip internal rotation**,
   **Right hip external rotation**, **Right hip flexion**, **Right ankle
   dorsiflexion**, or **Right ankle plantarflexion**;
4. saves the task for annotators.

While scrubbing an imported pair, the admin page requests the MMPose result
for the selected frame and draws its body skeleton over each video. A
semi-transparent depth-color layer is also shown when that RGB frame has a
depth frame: blue is nearer, red is farther, and black marks no-depth areas
that should not be used for annotation. The relevant hip, shoulder, elbow, knee,
and ankle landmarks are highlighted, and the frame status reports the exact MMPose `frame_id` or
explains when no result exists.
When an admin saves a task, the server also checks the task's relevant MMPose
landmarks against the original floating-point depth. If a landmark falls in a
black/no-depth region, the task is still saved but a red warning lists
the affected view, frame, and keypoint on the saved-task card. This makes it
easy to restore the task, choose a different frame, and delete the invalid
task if necessary.

If an iPad video needs conversion for Chrome, a progress bar appears beneath
that video's controls while its H.264 proxy is created. MMPose results are
preloaded for the admin pair and then selected from memory during playback, so
the overlay does not wait for a request on every video update.

The annotator signs in at `/` and receives the saved pair as exact source-frame
images. This avoids browser video-seeking differences: the displayed image,
the saved zero-based frame index, and its MMPose coordinates always refer to
the same original RGB frame.
For shoulder flexion, both views are shown, but only the left side-view image
accepts input. Brett places anonymous points **A**, **B**, and **C** in order;
the 3-D angle is calculated at **B**. The front frame is reference-only. Neutral
is treated as 0°, so this is a single-frame/T1 comparison and does not use a
T1/T2 difference.
For right shoulder abduction, only the right/front image accepts input. Brett
again places anonymous **A**, **B**, and **C** freely according to the clinical
reference line; the left/side image is reference-only. The server samples the
true 3-D point-cloud coordinates for those clicks, forms the torso frontal plane
from the MMPose left/right hip and shoulder landmarks, projects both Brett
vectors (A→B and B→C) into that plane, and measures the angle at B. Brett's
points are not replaced by or forced onto MMPose landmarks.
Right shoulder extension uses the same side-view instantaneous trunk-to-humerus
angle as shoulder flexion, with the selected task identifying the movement. The
right elbow-flexion task uses anonymous A-B-C points on the side view and
calculates the 3-D included shoulder–elbow–wrist angle as flexion from full
extension. Both tasks show the target side-view MMPose angle, the saved
front-view MMPose angle, and both absolute differences.
Right knee flexion follows the same side-view A-B-C workflow. Brett's points
are compared with the 3-D `right_hip`, `right_knee`, and `right_ankle` MMPose
angle on the saved side frame. The front frame is displayed for reference, but
its knee-flexion result is marked invalid when the knee is occluded by the
thigh.

Right knee extension uses the same side-view A-B-C workflow and the same
`right_hip`, `right_knee`, and `right_ankle` 3-D point-cloud geometry. Brett
places **A** at the right hip, **B** at the right knee, and **C** at the right
ankle. Full knee extension is reported as 0°, and MMPose supplies comparisons
from both the saved side and front frames.

Right hip flexion is a side-view A-B-C task. Brett places **A** at the right
shoulder, **B** at the right hip, and **C** at the right knee; the angle is
measured at **B**. MMPose uses the corresponding `right_shoulder`, `right_hip`,
and `right_knee` point-cloud coordinates on both saved frames. At the hip, the
included shoulder–hip–knee angle is converted to flexion as `180° − included
angle`, so the standing neutral configuration is approximately 0°. No hidden
neutral frame is required because the trunk is the anatomical reference. The
front-view hip-flexion result is marked invalid when the shoulder is occluded
by the thigh.

Right ankle dorsiflexion and plantarflexion are side-view A-B-C tasks. Brett
places **A** at the right knee, **B** at the right ankle, and **C** along the
right foot toward the midpoint of the toes. MMPose uses the 3-D knee and ankle
points plus the right heel and the midpoint of the right big and small toes to
form the tibia and foot axes. The tibia–foot included angle is approximately
90° in ankle neutral; the reported ROM is the absolute excursion from 90°, so
neutral is 0°. These are side-view measurements; the front image remains
reference-only and no front ankle angle is reported.

For right hip abduction, no neutral frame is required. The selected peak frames
remain the frames shown to Brett. Brett annotates the left/side image, while
MMPose evaluates both saved views using their respective anatomical references.

Brett places **A** at the right shoulder, **B** at the right hip, and **C** at
the right knee. The 3-D angle is measured at **B** from the shoulder-to-hip
trunk direction to the right thigh. Under the controlled supine protocol, the
patient keeps the pelvis still, the knee extended, and the leg sliding along
the floor, so this side-view trunk-to-thigh angle is used as the hip-abduction
estimate. The side result uses only the reliable right shoulder, right hip,
and right knee; it does not require the occluded contralateral landmarks. The
front-view MMPose result remains available as an independent comparison: it
forms the torso frontal plane from the bilateral shoulders and hips, projects
the right hip-to-knee thigh vector into that plane, and measures it from the
downward midpoint-shoulder-to-midpoint-hip pelvic reference. The result shows
Brett's side-view angle, MMPose's side- and front-view angles, and both
available absolute differences.

For right shoulder internal and external rotation, only the right/front image
accepts input. Brett places **A** at the shoulder, **B** at the elbow, and
**C** at the wrist. A→B defines the humerus axis and B→C defines the forearm;
the torso/chest normal from the MMPose hip/shoulder plane is projected into the
plane perpendicular to the humerus, and the forearm is compared with that
reference. These tasks show the front-view MMPose axial angle only. The selected
task identifies internal versus external rotation; the displayed value is the
unsigned excursion from the torso reference.

For right hip internal and external rotation, only the left/side image accepts
input and the MMPose comparison is also calculated from that same side frame.
Brett places anonymous **A**, **B**, and **C** at the right hip, right knee,
and right ankle. A→B defines the femur axis and B→C defines the lower-leg
direction. This is an axial-rotation calculation around the femur, not the
ordinary included angle at the knee. The admin selects a neutral frame in the
left/side video before saving the task. The selected frame is compared with
that hidden neutral frame, using the neutral lower-leg direction as the
reference. This avoids using a one-sided shoulder-to-hip line as the body
centreline; the front image remains reference-only and is not used for the
hip-rotation angle or comparison. Existing hip-rotation tasks created before
this selector was added continue to fall back to side frame 0.

The neutral-frame selector is currently enabled only for right hip
internal/external rotation. Hip-abduction tasks use anatomical references in
both views; hip-rotation tasks require only a side neutral frame because their
front image is reference-only.

For imported sessions containing `mmpose/*/rgb/rgb.json`, the server maps the
task's zero-based RGB frame to the JSON's one-based `frame_id`, reads the
appropriate left/right hip, shoulder, and elbow keypoints, and runs the same
angle calculation. For a side-view shoulder-flexion task, the annotator sees
four results: Brett's annotated 3-D angle on the left frame, the side-view
MMPose 3-D angle on that same frame, the independent front-view MMPose 3-D
angle from the saved right frame, plus absolute differences from Brett's angle
to each available MMPose result. MMPose is a validation reference; it does not
replace Brett's saved annotation.
RGBD data is used for the side-view trunk angles, 3-D plane projection, axial
rotation, and included joint angles. The active 3-D templates require a valid
depth frame; upload-only pairs cannot produce these true point-cloud angles.

## Importing the milestone2 folders

Use two paths in the admin page because each milestone2 session folder contains
one RGB video. For example, the supplied folder can be entered as the left
folder:

```text
/home/haziq/datasets/telept/data/milestone2/ipad1/030926_16-12_513_8EDDA9FF
```

The app discovers `rgb_video_*.mp4`, `depth_data_*`,
`instr_matrix_*.json`, `rgb_instr_matrix_*.json`, and
`mmpose/*/rgb/rgb.json` without copying the large files. The milestone2 depth
stream is read as raw DEFLATE containing
`320×240` little-endian `float32` depth frames. Its depth frame IDs are matched
to RGB frame IDs through `instr_matrix_*.json`, and the stored landscape depth
map is rotated 90° clockwise when mapping portrait RGB pixels. Existing pair
indexes written by the earlier counter-clockwise implementation are normalized
to this corrected transform when loaded.

For safety, folder imports are limited to `ROM2_ALLOWED_FOLDER_ROOTS`, which
defaults to `/home/haziq/datasets/telept/data`. Set it to a colon-separated
list when the source folders live elsewhere.

The iPad RGB videos are HEVC/H.265, which Chrome on Linux may not play. When
an imported or uploaded video uses an unsupported codec, the server creates a
browser-compatible H.264 proxy on first access under
`/home/haziq/datasets/telept/data/milestone2/rom_measure2/browser_videos/`.
This requires the system `ffmpeg` executable; the first access can take some
time for a large 4K video. The proxy is used for browser playback on the admin
page. Annotator frames are decoded directly from the original RGB file, so
their coordinates are independent of proxy timing.

For right shoulder flexion, Brett's A-B-C points are internally compared with:

```text
right_hip, right_shoulder, right_elbow
```

For left shoulder flexion, the corresponding model points are
`left_hip, left_shoulder, left_elbow`. In either case, the side-view angle is
the angle between the downward trunk reference (`shoulder → hip`) and the
humerus (`shoulder → elbow`). The elbow is intentional: it represents the
humerus endpoint used by the goniometer convention; the wrist is not used.

The admin chooses the side and front frame independently because the cameras
are not synchronized. The saved task stores both frame indices and timestamps;
clicking a saved task in the admin list restores that exact frame pair. The
admin can delete a saved task from this list; this removes it from the task
index and annotator queue while retaining any existing annotation JSON.
The target-view MMPose comparison is made only when the selected target frame
has a matching MMPose record and valid depth at the model landmarks; if it does
not, the comparison is marked unavailable.

The shoulder angle is measured in camera-space 3-D between the downward trunk
reference (`shoulder → hip`) and the humerus (`shoulder → elbow`), using the
depth value sampled around each clicked pixel and the depth-camera intrinsics.
The supplemental front-view shoulder-flexion angle forms the torso axes from
both hips and both shoulders, removes the left/right component from the
shoulder-to-elbow vector, and measures the remaining sagittal-plane vector
against the downward torso axis. Its overlay is drawn from the midpoint of the
two shoulders to the midpoint of the two hips, plus the target shoulder to
elbow segment. It is intentionally displayed as a separate reference because
the side and front frames are not synchronized and do not measure the same
projected motion.
For right shoulder abduction, MMPose is evaluated only on the right/front frame:
its shoulder-to-elbow vector is projected into the same torso frontal plane and
measured against the downward midpoint-shoulder-to-midpoint-hip reference.
The annotator page therefore shows the front-view MMPose angle and its absolute
difference from Brett's front-view 3-D angle; the side-view comparison remains
blank for this task because it is not part of the measurement.
For right hip abduction, MMPose is evaluated on both saved frames. On the front
frame it forms a torso frontal plane from the bilateral shoulders and hips,
projects the true 3-D right-hip-to-right-knee vector into that plane, and
measures it from the downward midpoint-shoulder-to-midpoint-hip pelvic
reference. On the side frame it uses the true 3-D right shoulder–right
hip–right knee trunk/thigh angle. This side estimate relies on the controlled
supine leg-slide protocol to isolate abduction and does not use a neutral frame.
Elbow and knee flexion use the joint included angle converted to flexion from
full extension. Ankle dorsiflexion/plantarflexion uses the tibia–foot included
angle relative to the 90° neutral ankle position. Axial shoulder/hip rotation
requires valid depth and a
standardized rotation posture. The side-view hip axial angle is relative to the
neutral frame and should be validated against clinical goniometer readings
before being treated as a clinical measure.

## Run

```bash
cd /home/haziq/datasets/telept/my_scripts/data_annotation/rom_measure2
python3 -m pip install -r requirements.txt
ROM2_ADMIN_PASSWORD=change-me \
ROM2_ANNOTATOR_PASSWORD=annotator-password \
python3 server.py --host 127.0.0.1 --port 8092
```

Open `http://127.0.0.1:8092/admin` for task setup or
`http://127.0.0.1:8092` for annotation. The local defaults are `admin` / `123`
for the admin and any non-admin username / `123` for an annotator. Change the
passwords before exposing the service.

Optional settings:

```bash
ROM2_STORAGE_DIR=/some/private/folder
ROM2_ADMIN_USERNAME=admin
ROM2_USERS="alice:alice-password,bob:bob-password"
ROM2_MAX_UPLOAD_MB=2048
ROM2_SECRET=use-a-long-random-secret
ROM2_ALLOWED_FOLDER_ROOTS=/home/haziq/datasets/telept/data
```

The storage directory contains:

```text
rom_measure2/
  uploads/<pair_id>/left.<ext>
  uploads/<pair_id>/right.<ext>
  browser_videos/<pair_id>/left.mp4
  browser_videos/<pair_id>/right.mp4
  pairs.json
  tasks.json
  annotations/<username>/<task_id>.json
```

Imported pairs keep references to the original session files; they are not
copied into `uploads`. The saved annotation contains points on the task's
target view, the derived 3-D points when depth is available, and the calculated
angle. It does not contain annotation points from the non-target view.
