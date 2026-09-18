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

## 4. Log in as Haziq

On the annotator page, use:

```text
Username: haziq
Password: 123
```

This will load Haziq's saved drafts and submitted annotations. The current
annotation storage is:

```text
/home/haziq/datasets/telept/data/milestone2/rom_measure/
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
