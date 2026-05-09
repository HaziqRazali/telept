"""
Render OpenSim model skeleton + muscles in the default pose, as a clean 2D
anatomical diagram (front + side views). Suitable for slides.

OpenSim world-space convention:
    Y = up (vertical)
    X = forward (anterior)
    Z = right (lateral)

Front view  = coronal plane  → horizontal = Z, vertical = Y
Side view   = sagittal plane → horizontal = X, vertical = Y

Usage (no .mot file needed):
    conda activate addbiomechanics

    # Full-body BSM (lower-body muscles + full skeleton)
    python render_muscles_static.py \
        --osim_path ~/code/SMPL2AddBiomechanics/models/bsm/bsm.osim \
        --output bsm_muscles_default.png

    # Arm26 — upper extremity (biceps, triceps, brachialis)
    python render_muscles_static.py \
        --osim_path /tmp/arm26.osim \
        --output arm26_muscles.png \
        --title "Arm26 — upper extremity (biceps, triceps, brachialis)"
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

try:
    import opensim as osim_api
except ImportError:
    sys.exit("ERROR: opensim Python package not found. conda activate addbiomechanics")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--osim_path", required=True, help="Path to .osim model file")
parser.add_argument("--output",    default="muscles_static.png", help="Output PNG path")
parser.add_argument("--title",     default="", help="Plot title")
parser.add_argument("--dpi",       type=int, default=250, help="Output resolution")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load model and compute muscle paths + body positions at default state
# ---------------------------------------------------------------------------
print(f"Loading {args.osim_path} …")
model = osim_api.Model(os.path.abspath(args.osim_path))
state = model.initSystem()   # all coordinates at their default values

# ── Muscle paths ─────────────────────────────────────────────────────────
muscles   = model.getMuscles()
n_muscles = muscles.getSize()
print(f"  {n_muscles} muscles found")

paths = []   # list of (N,3) arrays  — OpenSim world frame: X=fwd, Y=up, Z=right
names = []
for mi in range(n_muscles):
    pp_set = muscles.get(mi).getGeometryPath().getCurrentPath(state)
    pts = []
    for p in range(pp_set.getSize()):
        loc = pp_set.get(p).getLocationInGround(state)
        pts.append([loc[0], loc[1], loc[2]])
    if len(pts) >= 2:
        paths.append(np.array(pts, dtype=np.float32))
        names.append(muscles.get(mi).getName())

print(f"  {len(paths)} drawable muscle paths")

# ── Skeleton body-center positions ───────────────────────────────────────
body_set   = model.getBodySet()
body_pos   = {}   # name → [x, y, z]
for bi in range(body_set.getSize()):
    body = body_set.get(bi)
    loc  = body.findStationLocationInGround(state, osim_api.Vec3(0, 0, 0))
    body_pos[body.getName()] = np.array([loc[0], loc[1], loc[2]])

# ── Skeleton connectivity (parent → child, covers BSM and Rajagopal) ─────
# Detected automatically from the model's joint set where possible,
# but we hard-code the common BSM / Rajagopal topology for reliability.
BONE_EDGES = [
    # lower body
    ("pelvis",  "femur_r"),  ("femur_r",  "tibia_r"),
    ("tibia_r", "talus_r"),  ("talus_r",  "calcn_r"), ("calcn_r", "toes_r"),
    ("pelvis",  "femur_l"),  ("femur_l",  "tibia_l"),
    ("tibia_l", "talus_l"),  ("talus_l",  "calcn_l"), ("calcn_l", "toes_l"),
    # spine / torso — BSM names
    ("pelvis",      "lumbar_body"), ("lumbar_body", "thorax"), ("thorax", "head"),
    # spine / torso — Rajagopal names
    ("pelvis",      "torso"),
    # right arm — BSM
    ("thorax",      "scapula_r"),   ("scapula_r",  "humerus_r"),
    ("humerus_r",   "ulna_r"),      ("ulna_r",     "radius_r"), ("radius_r", "hand_r"),
    # left arm — BSM
    ("thorax",      "scapula_l"),   ("scapula_l",  "humerus_l"),
    ("humerus_l",   "ulna_l"),      ("ulna_l",     "radius_l"), ("radius_l", "hand_l"),
    # right arm — Rajagopal (torso instead of thorax)
    ("torso",       "humerus_r"),   ("torso",      "humerus_l"),
    # Arm26 model
    ("r_humerus",   "r_ulna"),      ("r_ulna",     "r_radius"), ("r_radius", "r_hand"),
    ("base",        "r_humerus"),
]

skeleton_lines = []
for a, b in BONE_EDGES:
    if a in body_pos and b in body_pos:
        skeleton_lines.append([body_pos[a], body_pos[b]])

# ---------------------------------------------------------------------------
# Colour by muscle group
# ---------------------------------------------------------------------------
def muscle_colour(name):
    n = name.lower()
    # upper extremity (Arm26 / Holzbaur)
    if any(x in n for x in ["biclong", "bicshort", "bic", "bra"]):
        return "#1F77B4"   # biceps + brachialis — blue
    if any(x in n for x in ["trilong", "trilat", "trimed", "tri"]):
        return "#D62728"   # triceps — red
    if any(x in n for x in ["delt", "subscap", "infra", "supra", "teres"]):
        return "#9467BD"   # shoulder — purple
    # lower extremity (BSM / Rajagopal)
    if any(x in n for x in ["glmax", "glmed", "glmin", "piri", "tfl"]):
        return "#E63946"
    if any(x in n for x in ["recfem", "vasint", "vaslat", "vasmed", "rect_fem", "vas_"]):
        return "#457B9D"
    if any(x in n for x in ["semimem", "semiten", "bflh", "bfsh", "bifemlh", "bifemsh",
                             "grac", "sart"]):
        return "#2D6A4F"
    if any(x in n for x in ["addbrev", "addlong", "addmag", "add_"]):
        return "#F4A261"
    if any(x in n for x in ["gaslat", "gasmed", "soleus", "med_gas", "lat_gas"]):
        return "#9B2226"
    if any(x in n for x in ["tibant", "tibpost", "edl", "ehl", "fdl", "fhl",
                             "perbrev", "perlong", "tib_", "per_", "flex_", "ext_"]):
        return "#8338EC"
    if any(x in n for x in ["iliacus", "psoas"]):
        return "#FB8500"
    return "#6C757D"

colours = [muscle_colour(n) for n in names]

# ---------------------------------------------------------------------------
# Build 2D segment lists
# Front view  (coronal):  horizontal = Z (idx 2), vertical = Y (idx 1)
# Side view (sagittal): horizontal = X (idx 0), vertical = Y (idx 1)
# ---------------------------------------------------------------------------
def make_muscle_segments_2d(paths, colours, hi, vi):
    segs, cols = [], []
    for path, col in zip(paths, colours):
        for i in range(len(path) - 1):
            segs.append([[path[i, hi], path[i, vi]], [path[i+1, hi], path[i+1, vi]]])
            cols.append(col)
    return segs, cols

def make_skel_segments_2d(skel_lines, hi, vi):
    return [[[a[hi], a[vi]], [b[hi], b[vi]]] for a, b in skel_lines]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 9), facecolor="white")

all_pts = np.vstack(paths)
y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
y_pad = (y_max - y_min) * 0.06
y_range = [y_min - y_pad, y_max + y_pad]

panel_cfg = [
    # ax, horiz_idx, vert_idx, xlabel, panel_title
    (axes[0], 2, 1, "Z  →  right (m)",    "Y  ↑  up (m)", "Front view  (coronal)"),
    (axes[1], 0, 1, "X  →  anterior (m)", "",             "Side view  (sagittal)"),
]

for ax, hi, vi, xlabel, ylabel, panel_title in panel_cfg:
    # ── skeleton bones (thick light-gray) ────────────────────────────────
    skel_segs = make_skel_segments_2d(skeleton_lines, hi, vi)
    if skel_segs:
        lc_skel = LineCollection(skel_segs, linewidths=3.5, colors="#B0B8C1",
                                 alpha=0.9, zorder=2, capstyle="round")
        ax.add_collection(lc_skel)

    # ── joint dots ───────────────────────────────────────────────────────
    if body_pos:
        bp = np.array(list(body_pos.values()))
        ax.scatter(bp[:, hi], bp[:, vi],
                   s=30, c="#4A4E69", zorder=4, alpha=0.85)

    # ── muscles ──────────────────────────────────────────────────────────
    m_segs, m_cols = make_muscle_segments_2d(paths, colours, hi, vi)
    lc_m = LineCollection(m_segs, linewidths=1.1, colors=m_cols,
                          alpha=0.80, zorder=3)
    ax.add_collection(lc_m)

    # ── axes ─────────────────────────────────────────────────────────────
    h_pts = all_pts[:, hi]
    h_min, h_max = h_pts.min(), h_pts.max()
    h_pad = (h_max - h_min) * 0.12
    ax.set_xlim(h_min - h_pad, h_max + h_pad)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.20, linewidth=0.5)
    ax.set_title(panel_title, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

# ── legend ────────────────────────────────────────────────────────────────
legend_items = [
    ("#B0B8C1", "Skeleton"),
    # upper extremity
    ("#1F77B4", "Biceps / brachialis"),
    ("#D62728", "Triceps"),
    ("#9467BD", "Shoulder muscles"),
    # lower extremity
    ("#E63946", "Gluteals / hip ext."),
    ("#457B9D", "Quadriceps"),
    ("#2D6A4F", "Hamstrings"),
    ("#F4A261", "Adductors"),
    ("#9B2226", "Plantarflexors"),
    ("#8338EC", "Dorsiflexors / peroneals"),
    ("#FB8500", "Hip flexors"),
]
handles = [mpatches.Patch(color=c, label=l) for c, l in legend_items]
fig.legend(handles=handles, loc="lower center", ncol=4,
           fontsize=7.5, frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.02))

title = args.title or os.path.basename(args.osim_path)
fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

plt.tight_layout()
out = os.path.abspath(args.output)
fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
