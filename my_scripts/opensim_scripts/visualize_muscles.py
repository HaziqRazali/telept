"""
Visualize OpenSim skeleton + muscle/tendon paths using the OpenSim Python API
and aitviewer for rendering.

Each muscle is drawn as a polyline through its via-points (origin → insertion).

Usage:
    conda activate addbiomechanics
    cd ~/code/SMPL2AddBiomechanics

    # Interactive GUI:
    python ~/datasets/telept/my_scripts/opensim_scripts/visualize_muscles.py \
        --osim_path output/01/osim_results/Models/match_markers_but_ignore_physics.osim \
        --mot_path  output/01/osim_results/IK/01_01_poses_segment_0_ik.mot \
        --gui

    # Headless video export:
    python ~/datasets/telept/my_scripts/opensim_scripts/visualize_muscles.py \
        --osim_path output/01/osim_results/Models/match_markers_but_ignore_physics.osim \
        --mot_path  output/01/osim_results/IK/01_01_poses_segment_0_ik.mot \
        --output muscles.mp4
"""

import argparse
import os
import sys
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Visualize OpenSim skeleton + muscle paths")
parser.add_argument("--osim_path", required=True, help="Path to .osim model file")
parser.add_argument("--mot_path",  required=True, help="Path to .mot IK result file")
parser.add_argument("--gui",        action="store_true", help="Open interactive viewer")
parser.add_argument("--output",     default="muscles.mp4", help="Output video (headless mode)")
parser.add_argument("--downsample", type=int, default=1,
                    help="Use every Nth frame (e.g. 4 = 4× faster, lower res). Default: 1 (all frames)")
parser.add_argument("--cache",      default=None,
                    help="Path to cache file (.npz). Saves muscle paths on first run, loads on subsequent runs.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# OpenSim Python API
# ---------------------------------------------------------------------------
try:
    import opensim as osim_api
except ImportError:
    sys.exit(
        "ERROR: opensim Python package not found.\n"
        "Install with:  conda install -c opensim-org opensim=4.5"
    )

# ---------------------------------------------------------------------------
# aitviewer imports (set window type before importing Viewer)
# ---------------------------------------------------------------------------
from aitviewer.configuration import CONFIG as C
if args.gui:
    C.window_type = "glfw"

from aitviewer.renderables.osim import OSIMSequence, load_osim
from aitviewer.renderables.lines import Lines
from aitviewer.viewer import Viewer

# ---------------------------------------------------------------------------
# Load .mot file
# ---------------------------------------------------------------------------
def load_mot(path):
    """Parse an OpenSim .mot file → (times, column_names, full_data_array)."""
    with open(path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader") + 1
    col_names = lines[start].split()
    data = np.array(
        [[float(v) for v in l.split()] for l in lines[start + 1:] if l.strip()],
        dtype=np.float64,
    )
    return data[:, 0], col_names, data


times, col_names, mot_data = load_mot(args.mot_path)

# Apply downsampling
if args.downsample > 1:
    times    = times[::args.downsample]
    mot_data = mot_data[::args.downsample]

n_frames = len(times)
fps = 1.0 / float(np.mean(np.diff(times))) if n_frames > 1 else 30.0
print(f"Loaded {n_frames} frames @ {fps:.1f} fps from {args.mot_path}")

# ---------------------------------------------------------------------------
# OpenSim model: compute world-space muscle path points for each frame
# (or load from cache if available)
# ---------------------------------------------------------------------------
cache_hit = False
if args.cache and os.path.exists(args.cache):
    print(f"Loading muscle paths from cache: {args.cache}")
    cache = np.load(args.cache, allow_pickle=True)
    muscle_names = list(cache["muscle_names"])
    n_muscles = len(muscle_names)
    muscle_paths_per_frame = [
        [cache[f"m{mi}_f{fi}"] for fi in range(n_frames)]
        for mi in range(n_muscles)
    ]
    cache_hit = True
    print("Cache loaded.")

if not cache_hit:
    model = osim_api.Model(args.osim_path)
    model.initSystem()

    coords     = model.getCoordinateSet()
    coord_map  = {coords.get(i).getName(): i for i in range(coords.getSize())}

    muscles    = model.getMuscles()
    n_muscles  = muscles.getSize()
    muscle_names = [muscles.get(i).getName() for i in range(n_muscles)]

    # muscle_paths_per_frame[muscle_idx][frame_idx] = ndarray (N_viapoints, 3)
    muscle_paths_per_frame = [[] for _ in range(n_muscles)]

    print(f"Computing muscle path points: {n_frames} frames × {n_muscles} muscles …")
    state = model.initSystem()

    try:
        from tqdm import tqdm as _tqdm
        frame_iter = _tqdm(range(n_frames), desc="Frames", unit="fr")
    except ImportError:
        frame_iter = range(n_frames)

    for fi in frame_iter:
        for col_idx, col_name in enumerate(col_names):
            if col_name not in coord_map:
                continue
            coord = coords.get(coord_map[col_name])
            val   = mot_data[fi, col_idx]
            if coord.getMotionType() == osim_api.Coordinate.Rotational:
                val = np.deg2rad(val)
            coord.setValue(state, val)

        model.realizePosition(state)

        for mi in range(n_muscles):
            pp_set = muscles.get(mi).getGeometryPath().getCurrentPath(state)
            pts = []
            for p in range(pp_set.getSize()):
                loc = pp_set.get(p).getLocationInGround(state)
                pts.append([loc[0], loc[1], loc[2]])
            muscle_paths_per_frame[mi].append(np.array(pts, dtype=np.float32))

    print("Done.")

    # Save to cache if requested
    if args.cache:
        print(f"Saving muscle paths to cache: {args.cache}")
        save_dict = {"muscle_names": np.array(muscle_names)}
        for mi in range(n_muscles):
            for fi in range(n_frames):
                save_dict[f"m{mi}_f{fi}"] = muscle_paths_per_frame[mi][fi]
        np.savez_compressed(args.cache, **save_dict)
        print("Cache saved.")

# ---------------------------------------------------------------------------
# OSIMSequence — skeleton rendered from bone meshes.
# Use from_files (identical to show_ab_results.py) so nimblephysics handles
# degree→radian conversion and world-space FK correctly.
# fps_out drives downsampling: e.g. 120 fps .mot → 30 fps with --downsample 4.
# ignore_geometry uses the aitviewer config Geometry folder (better meshes).
# ---------------------------------------------------------------------------
osim_seq = OSIMSequence.from_files(
    osim_path=os.path.abspath(args.osim_path),
    mot_file=os.path.abspath(args.mot_path),
    fps_out=int(fps),
    ignore_geometry=True,
    name="Skeleton",
)

# Align frame counts between skeleton and cached muscle paths
n_frames = min(n_frames, osim_seq.n_frames)
muscle_paths_per_frame = [frames[:n_frames] for frames in muscle_paths_per_frame]

# ---------------------------------------------------------------------------
# Build aitviewer Lines renderables — one per muscle
# ---------------------------------------------------------------------------
# Lines() expects shape (n_frames, n_points, 3).
# Muscles can have variable via-point counts; pad shorter frames by repeating
# the last point so the array is rectangular.

PALETTE = [
    (0.85, 0.15, 0.15, 1.0),   # red
    (0.15, 0.65, 0.85, 1.0),   # cyan
    (0.95, 0.55, 0.10, 1.0),   # orange
    (0.20, 0.80, 0.30, 1.0),   # green
    (0.70, 0.30, 0.90, 1.0),   # purple
    (0.95, 0.85, 0.10, 1.0),   # yellow
]

muscle_renderables = []
for mi, name in enumerate(muscle_names):
    frames   = muscle_paths_per_frame[mi]
    max_pts  = max(f.shape[0] for f in frames)
    if max_pts < 2:
        continue   # can't draw a line with fewer than 2 points

    padded = np.zeros((n_frames, max_pts, 3), dtype=np.float32)
    for fi, f in enumerate(frames):
        n = f.shape[0]
        padded[fi, :n] = f
        if n < max_pts:
            padded[fi, n:] = f[-1]   # pad by repeating last point

    colour = PALETTE[mi % len(PALETTE)]
    muscle_renderables.append(
        Lines(lines=padded, r_base=0.003, color=colour, name=name)
    )

print(f"Created {len(muscle_renderables)} muscle line renderables.")

# ---------------------------------------------------------------------------
# Launch aitviewer
# ---------------------------------------------------------------------------
v = Viewer()
v.run_animations = True
v.playback_fps   = fps
v.scene.camera.position = np.array([3.0, 1.5, 0.0])
v.scene.add(osim_seq, *muscle_renderables)

if args.gui:
    # Alias all event callbacks (moderngl_window expects on_* names;
    # aitviewer Viewer defines bare names)
    v.on_render                = v.render
    v.on_resize                = v.resize
    v.on_key_event             = v.key_event
    v.on_mouse_position_event  = v.mouse_position_event
    v.on_mouse_press_event     = v.mouse_press_event
    v.on_mouse_release_event   = v.mouse_release_event
    v.on_mouse_drag_event      = v.mouse_drag_event
    v.on_mouse_scroll_event    = v.mouse_scroll_event
    v.on_unicode_char_entered  = v.unicode_char_entered
    v.on_files_dropped_event   = v.files_dropped_event
    v.window.config = v
    v.run()
else:
    v._init_scene()
    v.export_video(args.output, output_fps=fps)
    print(f"Saved to {args.output}")
