"""
Interactive 3D viewer: OpenSim skeleton + colour-coded muscle paths.

Requires:  conda activate addbiomechanics

Usage
-----
# Static default pose (no motion needed):
python view_muscles_3d.py --osim_path ~/code/SMPL2AddBiomechanics/models/bsm/bsm.osim

# With animated motion:
python view_muscles_3d.py \
    --osim_path /path/to/model.osim \
    --mot_path  /path/to/ik.mot \
    --downsample 4 \
    --cache /tmp/muscles.npz

Controls (aitviewer):
  Left-drag   → rotate
  Right-drag  → pan
  Scroll      → zoom
  Space       → play / pause animation
  Q           → quit
"""

import argparse
import os
import sys
import tempfile
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Interactive 3D OpenSim muscle viewer")
parser.add_argument("--osim_path",  required=True, help="Path to .osim model file")
parser.add_argument("--mot_path",   default=None,  help="Optional .mot for animated playback")
parser.add_argument("--downsample", type=int, default=1, help="Use every Nth frame (default 1 = all)")
parser.add_argument("--cache",      default=None,  help="Cache muscle paths to/from .npz file")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# OpenSim
# ---------------------------------------------------------------------------
try:
    import opensim as osim_api
except ImportError:
    sys.exit("ERROR: opensim not found — run: conda activate addbiomechanics")

# ---------------------------------------------------------------------------
# aitviewer
# ---------------------------------------------------------------------------
from aitviewer.configuration import CONFIG as C
C.window_type = "glfw"

from aitviewer.renderables.osim import OSIMSequence
from aitviewer.renderables.lines import Lines
from aitviewer.viewer import Viewer
import imgui

# ---------------------------------------------------------------------------
# Muscle group colours  (hex → float RGBA, consistent with render_muscles_static.py)
# ---------------------------------------------------------------------------
_GROUP_TABLE = [
    # (substrings,          hex colour,   label)
    (["biclong", "bicshort", "bra"],   "#1F77B4", "Biceps / brachialis"),
    (["trilong", "trilat", "trimed"],  "#D62728", "Triceps"),
    (["delt", "subscap", "infra", "supra", "teres"], "#9467BD", "Shoulder"),
    (["glmax", "glmed", "glmin", "glut_max", "glut_med", "glut_min",
      "piri", "tfl", "quad_fem", "gem", "peri"],        "#E63946", "Gluteal / deep hip"),
    (["recfem", "vasint", "vaslat", "vasmed",
      "rect_fem", "vas_med", "vas_int", "vas_lat"],      "#457B9D", "Quadriceps"),
    (["semimem", "semiten", "bflh", "bfsh",
      "bifemlh", "bifemsh", "grac", "sart", "sar_"],     "#2D6A4F", "Hamstrings / gracilis"),
    (["addbrev", "addlong", "addmag",
      "add_long", "add_brev", "add_mag", "pect"],         "#F4A261", "Hip adductors"),
    (["gaslat", "gasmed", "soleus",
      "med_gas", "lat_gas"],                              "#9B2226", "Gastrocnemius / soleus"),
    (["tibant", "tibpost", "edl", "ehl", "fdl", "fhl",
      "perbrev", "perlong", "tib_", "per_", "flex_", "ext_"],  "#8338EC", "Ankle / foot"),
    (["iliacus", "psoas"],                                "#FB8500", "Iliopsoas"),
    (["ercspn", "intobl", "extobl"],                      "#C77DFF", "Trunk"),
]

def _hex_to_rgba(h):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return (r, g, b, 1.0)

def muscle_group(name):
    n = name.lower()
    for substrings, colour, label in _GROUP_TABLE:
        if any(s in n for s in substrings):
            return _hex_to_rgba(colour), label
    return (0.42, 0.42, 0.42, 1.0), "Other"

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print(f"Loading {args.osim_path} …")
model = osim_api.Model(os.path.abspath(args.osim_path))
state = model.initSystem()

muscles   = model.getMuscles()
n_muscles = muscles.getSize()
muscle_names = [muscles.get(mi).getName() for mi in range(n_muscles)]

# Pre-compute legend groups (unique labels that actually appear)
legend = {}   # label → rgba
for name in muscle_names:
    rgba, label = muscle_group(name)
    legend.setdefault(label, rgba)

# ---------------------------------------------------------------------------
# Load .mot if provided
# ---------------------------------------------------------------------------
_tmp_mot = None   # track temp file for cleanup

if args.mot_path:
    with open(args.mot_path) as f:
        raw_lines = f.readlines()
    start = next(i for i, l in enumerate(raw_lines) if l.strip().lower() == "endheader") + 1
    col_names = raw_lines[start].split()
    mot_data  = np.array(
        [[float(v) for v in l.split()] for l in raw_lines[start + 1:] if l.strip()],
        dtype=np.float64,
    )
    times = mot_data[:, 0]
    if args.downsample > 1:
        times    = times[::args.downsample]
        mot_data = mot_data[::args.downsample]
    n_frames = len(times)
    fps = 1.0 / float(np.mean(np.diff(times))) if n_frames > 1 else 30.0
    coords    = model.getCoordinateSet()
    coord_map = {coords.get(i).getName(): i for i in range(coords.getSize())}
    print(f"Loaded {n_frames} frames @ {fps:.1f} fps from {args.mot_path}")
else:
    # Static default pose — use a 2-frame dummy mot so OSIMSequence is happy
    n_frames = 1
    fps      = 30.0
    col_names = []
    mot_data  = None

# ---------------------------------------------------------------------------
# Compute (or load) muscle paths
# ---------------------------------------------------------------------------
cache_hit = False
if args.cache and os.path.exists(args.cache):
    print(f"Loading muscle paths from cache: {args.cache}")
    cache       = np.load(args.cache, allow_pickle=True)
    muscle_names = list(cache["muscle_names"])
    n_muscles   = len(muscle_names)
    muscle_paths_per_frame = [
        [cache[f"m{mi}_f{fi}"] for fi in range(n_frames)]
        for mi in range(n_muscles)
    ]
    cache_hit = True
    print("Cache loaded.")

if not cache_hit:
    muscle_paths_per_frame = [[] for _ in range(n_muscles)]
    print(f"Computing muscle paths: {n_frames} frame(s) × {n_muscles} muscles …")
    state = model.initSystem()

    try:
        from tqdm import tqdm
        frame_iter = tqdm(range(n_frames), desc="Frames", unit="fr")
    except ImportError:
        frame_iter = range(n_frames)

    for fi in frame_iter:
        if mot_data is not None:
            coords    = model.getCoordinateSet()
            coord_map = {coords.get(i).getName(): i for i in range(coords.getSize())}
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

    if args.cache:
        print(f"Saving cache → {args.cache}")
        save_dict = {"muscle_names": np.array(muscle_names)}
        for mi in range(n_muscles):
            for fi in range(n_frames):
                save_dict[f"m{mi}_f{fi}"] = muscle_paths_per_frame[mi][fi]
        np.savez_compressed(args.cache, **save_dict)
        print("Cache saved.")

# ---------------------------------------------------------------------------
# Build skeleton (OSIMSequence)
# For static pose: write a 2-frame dummy mot with all-zero coords
# ---------------------------------------------------------------------------
if args.mot_path:
    mot_for_skel = os.path.abspath(args.mot_path)
else:
    import math
    coords2     = model.getCoordinateSet()
    coord_names = [coords2.get(i).getName() for i in range(coords2.getSize())]
    # Use model's default coordinate values (not zeros) for anatomically correct pose
    default_vals = []
    for i in range(coords2.getSize()):
        c = coords2.get(i)
        val = c.getDefaultValue()
        if c.getMotionType() == osim_api.Coordinate.Rotational:
            val = math.degrees(val)
        default_vals.append(val)
    defaults_str = "\t".join(f"{v:.6f}" for v in default_vals)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".mot", delete=False)
    _tmp_mot = tmp.name
    tmp.write("default_pose\nversion=1\nnRows=2\n"
              f"nColumns={len(coord_names) + 1}\ninDegrees=yes\nendheader\n")
    tmp.write("time\t" + "\t".join(coord_names) + "\n")
    tmp.write(f"0.0\t{defaults_str}\n")
    tmp.write(f"0.1\t{defaults_str}\n")
    tmp.close()
    mot_for_skel = _tmp_mot

osim_seq = None
try:
    osim_seq = OSIMSequence.from_files(
        osim_path=os.path.abspath(args.osim_path),
        mot_file=mot_for_skel,
        fps_out=int(fps) if args.mot_path else None,
        ignore_geometry=False,
        name="Skeleton",
    )
except Exception as e:
    print(f"[warning] Could not load skeleton via nimble (muscles only): {e}")

if _tmp_mot:
    os.unlink(_tmp_mot)

# Align frame counts (static pose: both are 1)
if osim_seq is not None:
    n_frames = min(n_frames, osim_seq.n_frames)
muscle_paths_per_frame = [frames[:n_frames] for frames in muscle_paths_per_frame]

# ---------------------------------------------------------------------------
# Build Lines renderables — one per muscle
# ---------------------------------------------------------------------------
muscle_renderables = []
for mi, name in enumerate(muscle_names):
    frames  = muscle_paths_per_frame[mi]
    max_pts = max(f.shape[0] for f in frames)
    if max_pts < 2:
        continue
    padded = np.zeros((n_frames, max_pts, 3), dtype=np.float32)
    for fi, f in enumerate(frames):
        n = f.shape[0]
        padded[fi, :n] = f
        if n < max_pts:
            padded[fi, n:] = f[-1]
    colour, _ = muscle_group(name)
    muscle_renderables.append(Lines(lines=padded, r_base=0.003, color=colour, name=name))

print(f"Built {len(muscle_renderables)} muscle line renderables.")

# ---------------------------------------------------------------------------
# Print legend to terminal (ANSI colours)
# ---------------------------------------------------------------------------
print("\n── Muscle legend ──")
for label, (r, g, b, _) in legend.items():
    ri, gi, bi = int(r * 255), int(g * 255), int(b * 255)
    print(f"  \033[38;2;{ri};{gi};{bi}m■\033[0m  {label}")
print()

# ---------------------------------------------------------------------------
# Viewer with imgui legend overlay
# ---------------------------------------------------------------------------
class LegendViewer(Viewer):
    """Viewer that adds a floating colour legend panel."""
    def __init__(self, *a, legend=None, **kw):
        self._legend_items = list((legend or {}).items())   # [(label, rgba), ...]
        super().__init__(*a, **kw)
        self.gui_controls["legend"] = self._gui_legend

    def _gui_legend(self):
        imgui.set_next_window_position(10, 30, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(220, 24 + 22 * len(self._legend_items), imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin("Muscle groups", True)
        if expanded:
            for label, (r, g, b, a) in self._legend_items:
                imgui.push_style_color(imgui.COLOR_TEXT, r, g, b, a)
                imgui.text(f"  {label}")
                imgui.pop_style_color()
        imgui.end()


v = LegendViewer(legend=legend)
v.run_animations = True
v.playback_fps   = fps
v.scene.camera.position = np.array([3.0, 1.5, 0.0])
v.scene.add(*([osim_seq] if osim_seq is not None else []), *muscle_renderables)

# Alias event callbacks required by moderngl_window
v.on_render               = v.render
v.on_resize               = v.resize
v.on_key_event            = v.key_event
v.on_mouse_position_event = v.mouse_position_event
v.on_mouse_press_event    = v.mouse_press_event
v.on_mouse_release_event  = v.mouse_release_event
v.on_mouse_drag_event     = v.mouse_drag_event
v.on_mouse_scroll_event   = v.mouse_scroll_event
v.on_unicode_char_entered = v.unicode_char_entered
v.on_files_dropped_event  = v.files_dropped_event
v.window.config = v
v.run()
