# Default-pose examples

These examples render the copied MHR body without a JSON pose or video.  The
native MHR model receives zero model/pose parameters, which gives its default
pose.  The geometry remains the current lean identity plus the standard
`Lean_Abs_Definition` layer; only the appearance changes.

Run either wrapper from any directory:

```bash
bash /home/haziq/datasets/telept/data/body_models/mhr/example_scripts/render_default_blue.sh
bash /home/haziq/datasets/telept/data/body_models/mhr/example_scripts/render_default_reskinned.sh
```

The wrappers default to the local `mhr_new` interpreter used for validation.
On another checkout or host, override it without editing the scripts:

```bash
MHR_PYTHON=/path/to/python ./render_default_blue.sh
```

Or call the common renderer directly:

```bash
/home/haziq/anaconda3/envs/mhr_new/bin/python \
  /home/haziq/datasets/telept/data/body_models/mhr/example_scripts/render_default_mhr.py \
  --appearance blue

/home/haziq/anaconda3/envs/mhr_new/bin/python \
  /home/haziq/datasets/telept/data/body_models/mhr/example_scripts/render_default_mhr.py \
  --appearance reskinned
```

Outputs are written to:

```text
/home/haziq/datasets/telept/data/body_models/mhr/previews/mhr_default_pose_blue_dark_fresnel.png
/home/haziq/datasets/telept/data/body_models/mhr/previews/mhr_default_pose_reskinned_dark_gray.png
```

Each command also writes separate `_front.png` and `_side.png` images.  The
colored skeleton and white joint dots are included for inspection.  The
central chain now has a marker at `c_head` and stops there; the short
`c_head`-to-`c_jaw` facial segment is omitted.  The renderer scripts
themselves remain in the ROM visualization directory; these files are only
small body-model preview examples.
