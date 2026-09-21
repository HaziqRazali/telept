# ROM Measure launch guide

This directory contains the paired-video ROM annotation app.

## 1. Activate the environment

```bash
conda activate base
```

The required packages are already installed in this environment. If needed,
verify them with:

```bash
python3 -c "import flask, numpy, cv2; print('environment OK')"
```

## 2. Start the server

```bash
cd /data/haziq/telept/my_scripts/result_evaluation/milestone2/manual_vs_mmpose
python3 server.py --host 127.0.0.1 --port 8092
```

Leave this terminal running. Press `Ctrl+C` to stop the server.

## 3. Open the app

- Annotator page: <http://127.0.0.1:8092/>
- Admin page: <http://127.0.0.1:8092/admin>

The server is bound to the workstation's loopback interface. If your browser
is on your local PC rather than the workstation, create the tunnel from the
local PC and keep it running:

```bash
ssh -N -L 8092:127.0.0.1:8092 haziq@100.83.137.120
```

Then open the same URLs on the local PC. Here, `127.0.0.1:8092` in the
browser is forwarded to port 8092 on the workstation.

## 4. Log in as Haziq

On the annotator page, use:

```text
Username: haziq
Password: 123
```

This will load Haziq's saved drafts and submitted annotations. The current
annotation storage is:

```text
/data/haziq/telept/data/milestone2/rom_measure/
```

Do not set `ROM2_STORAGE_DIR` to a different location unless you intend to
use a different annotation database.

## Optional admin login

```text
Username: admin
Password: 123
```

Use the admin page to import video pairs, choose frames, and create tasks for
annotators.

## How the app works

The app has two workflows:

1. An admin imports a pair of sessions and creates measurement tasks.
2. An annotator opens the saved task, places the required points, and saves a
   draft or submits the annotation.

For the milestone2 convention, the **left video is the side view** and the
**right video is the front/reference view**. The two cameras are not assumed
to be synchronized.

### How the admin chooses frames

1. Open `/admin` and sign in as `admin`.
2. Enter the left/side session folder and the right/front session folder.
3. Use **Inspect folders** to check that the RGB video, depth stream, camera
   intrinsics, and MMPose output were discovered.
4. Select **Import folder pair**. The pair then appears in the imported-pair
   list.
5. Select the pair and scrub the left and right frame sliders independently.
   The page shows the exact zero-based frame number, timestamp, MMPose
   overlay, and depth overlay when available.
6. Choose the movement template and task name, then select **Save task for
   users**.

The saved task records both selected frame indices and timestamps. Clicking a
saved task later restores those exact frames. Hip internal/external rotation
tasks also require the admin to mark the selected neutral frame before saving.
The admin task card reports whether the exact MMPose frame and valid depth are
available, and warns about relevant landmarks that fall in invalid/black depth.

The admin may also upload a pair of videos as a fallback. Folder imports do
not copy the large source files: the server reads the original session data
and records its paths. Uploaded videos are copied into the app's storage.

## Data sources and metadata storage

By default, the app reads and writes under:

```text
/data/haziq/telept/data/milestone2/rom_measure/
```

The default folder-import safety root is:

```text
/data/haziq/telept/data
```

For the current dataset, source sessions are under paths such as
`/data/haziq/telept/data/milestone2/ipad1/` and
`/data/haziq/telept/data/milestone2/ipad2/`. Each imported session can contain
the RGB video, `depth_data_*`, `instr_matrix_*.json`,
`rgb_instr_matrix_*.json`, and `mmpose/*/rgb/rgb.json`. Those source files
remain in their session folders; the pair index stores references to them.

The app storage has this layout:

```text
/data/haziq/telept/data/milestone2/rom_measure/
├── pairs.json
├── tasks.json
├── annotations/
│   └── <username>/<task_id>.json
├── uploads/<pair_id>/
└── browser_videos/<pair_id>/
```

- `pairs.json` stores imported-pair metadata, including the original left and
  right file paths, video properties, depth metadata, camera information, and
  MMPose availability.
- `tasks.json` stores the admin's task definitions: movement template, task
  name, selected left/right frame indices, timestamps, neutral-frame indices
  when required, and MMPose/depth availability or warnings.
- `annotations/<username>/<task_id>.json` stores each annotator's draft or
  submitted A/B/C points, calculated 3-D angle, sampled depth values, frame
  indices, notes, status, and MMPose comparison results. For example, Haziq's
  annotations are in `annotations/haziq/`.
- `uploads/` contains videos uploaded through the fallback upload form.
- `browser_videos/` contains H.264 browser proxies generated from source
  videos when Chrome cannot play the original codec. These proxies can be
  regenerated; they are not the annotation source of truth.

The storage location can be overridden with `ROM2_STORAGE_DIR`. The allowed
folder-import roots can be overridden with `ROM2_ALLOWED_FOLDER_ROOTS`, using
colon-separated paths. The measurement-specific geometry and task templates
are documented in
[`manual_vs_mmpose/README.md`](manual_vs_mmpose/README.md).
