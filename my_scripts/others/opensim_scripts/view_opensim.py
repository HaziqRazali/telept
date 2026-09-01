"""
view_opensim.py — Interactive 3D viewer: OpenSim skeleton + colour-coded muscles.

Auto-detects the Geometry/ folder next to the .osim file, auto-converts .vtp → .ply
if needed (requires vtk), then renders bone meshes + muscle path tubes together in
an aitviewer window.

Requires:  conda activate addbiomechanics

Usage
-----
# BSM — full skeleton + 80 lower-body muscles (default pose)
#   Shows: skull, ribcage, spine, pelvis, arms, legs as solid meshes.
#   80 muscle tubes in hip/leg region (upper body is torque-actuated, no muscles).
#   Legs slightly apart in anatomical default stance. Blue dots = marker via-points.
python view_opensim.py --osim_path ~/datasets/mocap/data/skel_models_v1.1/bsm.osim

# Rajagopal — full skeleton + 80 lower-body muscles (default pose)
#   Shows: full skeleton (no skull mesh) in narrow neutral stance.
#   80 muscle tubes concentrated around hip/upper leg. No marker dots.
python view_opensim.py --osim_path ~/code/AddBiomechanics/server/data/StandardizedModels/Rajagopal2015_passiveCal_hipAbdMoved.osim

# With animated IK motion
#   Shows: skeleton + muscles animated over time. Space to play/pause.
python view_opensim.py --osim_path /path/to/model.osim --mot_path /path/to/ik.mot

GUI layout
----------
  Top-left panel   : "Muscle groups" colour legend
  Left sidebar     : Scene tree — toggle Skeleton or any individual muscle
  Bottom bar       : Playback scrubber (only active when a .mot is provided)

Controls
--------
  Left-drag   → rotate      Right-drag  → pan
  Scroll      → zoom        Space       → play / pause
  , / .       → prev/next frame         Q / Esc → quit

Muscle colour groups
--------------------
  Hip adductors          orange
  Hamstrings / gracilis  dark green
  Ankle / foot           purple
  Gastrocnemius / soleus dark red
  Gluteal / deep hip     red
  Iliopsoas              bright orange
  Quadriceps             steel blue
  Biceps / brachialis    blue          (upper-body models only)
  Triceps                red           (upper-body models only)
  Shoulder               purple        (upper-body models only)
"""

import argparse
import glob
import math
import os
import sys
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="OpenSim skeleton + muscle viewer")
parser.add_argument("--osim_path",  required=True, help="Path to .osim model file")
parser.add_argument("--mot_path",   default=None,  help="Optional .mot for animation")
parser.add_argument("--downsample", type=int, default=1, help="Use every Nth frame")
parser.add_argument("--tpose", action="store_true",
    help="Override arm DOFs to T-pose (arms horizontal, 90° abduction)")
args = parser.parse_args()

osim_path = os.path.abspath(os.path.expanduser(args.osim_path))
if not os.path.exists(osim_path):
    sys.exit(f"ERROR: {osim_path} not found")

# ---------------------------------------------------------------------------
# Auto-detect Geometry folder next to the .osim file
# Convert .vtp → .ply if needed (aitviewer only reads .ply)
# ---------------------------------------------------------------------------
osim_dir = os.path.dirname(osim_path)
geom_dir = os.path.join(osim_dir, "Geometry")

if not os.path.isdir(geom_dir):
    print(f"[warning] No Geometry/ folder at {geom_dir} — bones will not render")
    geom_dir = None
else:
    vtps = glob.glob(os.path.join(geom_dir, "*.vtp"))
    need_convert = [f for f in vtps if not os.path.exists(f + ".ply")]
    if need_convert:
        print(f"Converting {len(need_convert)} .vtp → .ply in {geom_dir} …")
        try:
            import vtk
            errors = 0
            for f in need_convert:
                try:
                    r = vtk.vtkXMLPolyDataReader()
                    r.SetFileName(f)
                    r.Update()
                    w = vtk.vtkPLYWriter()
                    w.SetFileName(f + ".ply")
                    w.SetInputConnection(r.GetOutputPort())
                    w.Write()
                except Exception as e:
                    errors += 1
                    print(f"  [error] {os.path.basename(f)}: {e}")
            print(f"Conversion done ({errors} errors).")
        except ImportError:
            print("[warning] vtk not installed — pip install vtk")
    else:
        n_ply = len(glob.glob(os.path.join(geom_dir, "*.ply")))
        print(f"Geometry: {n_ply} .ply files ready in {geom_dir}")

# ---------------------------------------------------------------------------
# aitviewer config — must be set BEFORE importing renderable classes
# Set osim_geometry to THIS model's Geometry folder so bones render correctly
# ---------------------------------------------------------------------------
from aitviewer.configuration import CONFIG as C
C.window_type = "glfw"
if geom_dir:
    C.osim_geometry = geom_dir

# ---------------------------------------------------------------------------
# OpenSim
# ---------------------------------------------------------------------------
try:
    import opensim as osim_api
except ImportError:
    sys.exit("ERROR: opensim not found — conda activate addbiomechanics")

print(f"Loading {osim_path} …")
model = osim_api.Model(osim_path)
state = model.initSystem()

muscles      = model.getMuscles()
n_muscles    = muscles.getSize()
muscle_names = [muscles.get(mi).getName() for mi in range(n_muscles)]
print(f"Found {n_muscles} muscles.")

# ---------------------------------------------------------------------------
# Muscle group colours
# ---------------------------------------------------------------------------
_GROUP_TABLE = [
    (["biclong", "bicshort", "bra"],                                 "#1F77B4", "Biceps / brachialis"),
    (["trilong", "trilat", "trimed"],                                "#D62728", "Triceps"),
    (["delt", "subscap", "infra", "supra", "teres"],                 "#9467BD", "Shoulder"),
    (["glmax", "glmed", "glmin", "glut_max", "glut_med", "glut_min",
      "piri", "tfl", "quad_fem", "gem", "peri"],                     "#E63946", "Gluteal / deep hip"),
    (["recfem", "vasint", "vaslat", "vasmed",
      "rect_fem", "vas_med", "vas_int", "vas_lat"],                  "#457B9D", "Quadriceps"),
    (["semimem", "semiten", "bflh", "bfsh",
      "bifemlh", "bifemsh", "grac", "sart", "sar_"],                 "#2D6A4F", "Hamstrings / gracilis"),
    (["addbrev", "addlong", "addmag",
      "add_long", "add_brev", "add_mag", "pect"],                    "#F4A261", "Hip adductors"),
    (["gaslat", "gasmed", "soleus", "med_gas", "lat_gas"],           "#9B2226", "Gastrocnemius / soleus"),
    (["tibant", "tibpost", "edl", "ehl", "fdl", "fhl",
      "perbrev", "perlong", "tib_", "per_", "flex_", "ext_"],        "#8338EC", "Ankle / foot"),
    (["iliacus", "psoas"],                                           "#FB8500", "Iliopsoas"),
    (["ercspn", "intobl", "extobl"],                                 "#C77DFF", "Trunk"),
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

legend = {}
for name in muscle_names:
    rgba, label = muscle_group(name)
    legend.setdefault(label, rgba)

# ---------------------------------------------------------------------------
# Load .mot or build 1-frame default-pose mot
# ---------------------------------------------------------------------------
_tmp_mot  = None
col_names = []
mot_data  = None

if args.mot_path:
    mot_path = os.path.abspath(os.path.expanduser(args.mot_path))
    with open(mot_path) as f:
        raw_lines = f.readlines()
    start     = next(i for i, l in enumerate(raw_lines) if l.strip().lower() == "endheader") + 1
    col_names = raw_lines[start].split()
    mot_data  = np.array(
        [[float(v) for v in l.split()] for l in raw_lines[start + 1:] if l.strip()],
        dtype=np.float64)
    times = mot_data[:, 0]
    if args.downsample > 1:
        times    = times[::args.downsample]
        mot_data = mot_data[::args.downsample]
    n_frames = len(times)
    fps = 1.0 / float(np.mean(np.diff(times))) if n_frames > 1 else 30.0
    mot_for_skel = mot_path
    print(f"Loaded {n_frames} frames @ {fps:.1f} fps from {mot_path}")
else:
    n_frames = 1
    fps      = 30.0
    # Build 2-frame mot using model's own default coordinate values (anatomically correct)
    coords2     = model.getCoordinateSet()
    coord_names = [coords2.get(i).getName() for i in range(coords2.getSize())]
    TPOSE_OVERRIDES = {"arm_add_r": -90.0, "arm_add_l": -90.0}
    default_vals = []
    for i in range(coords2.getSize()):
        c    = coords2.get(i)
        name = c.getName()
        val  = c.getDefaultValue()
        if c.getMotionType() == osim_api.Coordinate.Rotational:
            val = math.degrees(val)
        if args.tpose and name in TPOSE_OVERRIDES:
            val = TPOSE_OVERRIDES[name]
        default_vals.append(val)
    dstr = "\t".join(f"{v:.6f}" for v in default_vals)
    tmp  = tempfile.NamedTemporaryFile(mode="w", suffix=".mot", delete=False)
    _tmp_mot = tmp.name
    tmp.write("default_pose\nversion=1\nnRows=2\n"
              f"nColumns={len(coord_names)+1}\ninDegrees=yes\nendheader\n")
    tmp.write("time\t" + "\t".join(coord_names) + "\n")
    tmp.write(f"0.0\t{dstr}\n0.1\t{dstr}\n")
    tmp.close()
    mot_for_skel = _tmp_mot

# ---------------------------------------------------------------------------
# Compute muscle paths
# ---------------------------------------------------------------------------
print(f"Computing muscle paths: {n_frames} frame(s) × {n_muscles} muscles …")
state = model.initSystem()
muscle_paths_per_frame = [[] for _ in range(n_muscles)]

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
        pts    = []
        for p in range(pp_set.getSize()):
            loc = pp_set.get(p).getLocationInGround(state)
            pts.append([loc[0], loc[1], loc[2]])
        muscle_paths_per_frame[mi].append(np.array(pts, dtype=np.float32))

# ---------------------------------------------------------------------------
# Load skeleton via aitviewer
# ---------------------------------------------------------------------------
from aitviewer.renderables.osim import OSIMSequence
from aitviewer.renderables.lines import Lines
from aitviewer.viewer import Viewer
import imgui

osim_seq = None
try:
    osim_seq = OSIMSequence.from_files(
        osim_path=osim_path,
        mot_file=mot_for_skel,
        fps_out=int(fps) if args.mot_path else None,
        ignore_geometry=False,
        name="Skeleton",
    )
    print("Skeleton loaded.")
except Exception as e:
    print(f"[warning] Skeleton failed (muscles-only mode): {e}")

if _tmp_mot:
    os.unlink(_tmp_mot)

if osim_seq is not None:
    n_frames = min(n_frames, osim_seq.n_frames)
muscle_paths_per_frame = [frames[:n_frames] for frames in muscle_paths_per_frame]

# ---------------------------------------------------------------------------
# Build muscle line renderables
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

print(f"Built {len(muscle_renderables)} muscle renderables.")

print("\n── Muscle legend ──")
for label, (r, g, b, _) in legend.items():
    ri, gi, bi = int(r * 255), int(g * 255), int(b * 255)
    print(f"  \033[38;2;{ri};{gi};{bi}m■\033[0m  {label}")
print()

# ---------------------------------------------------------------------------
# Viewer with imgui legend overlay
# ---------------------------------------------------------------------------
class LegendViewer(Viewer):
    def __init__(self, *a, legend=None, **kw):
        self._legend_items = list((legend or {}).items())
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
