#!/usr/bin/env python3
"""
find_mhr_rajagopal_markers.py

Maps Rajagopal's 57 markers (Apache 2.0) onto the MHR mesh (Apache 2.0).
No SMPL, no BSM, no Max Planck IP in the chain.

Pipeline
--------
1. Load Rajagopal.osim → FK at T-pose (arms out 90°) → world XYZ for 57 markers
2. Load MHR T-pose mesh via SMPL barycentric transfer
3. Scale BOTH uniformly (isotropic) to --target_height metres
   Height measured from bottom of feet (calcn body Y) to estimated head top (C7+0.135m)
   Same scale factor applied to all 3 axes — body proportions preserved
4. KD-tree nearest-neighbour on scaled meshes → MHR vertex ID per marker
5. Print table + save mhr_rajagopal_markers.yaml

Usage
-----
    conda activate addbiomechanics
    cd ~/datasets/telept/my_scripts/opensim_scripts
    python find_mhr_rajagopal_markers.py
    python find_mhr_rajagopal_markers.py --no_save          # print only
    python find_mhr_rajagopal_markers.py --visualize        # side-by-side Open3D
    python find_mhr_rajagopal_markers.py --target_height 1.8  # scale to 1.8m
"""

import argparse
import os
import pickle

import numpy as np
from scipy.spatial import cKDTree
import yaml

OSIM_PATH = (
    "/home/haziq/code/AddBiomechanics/server/data/StandardizedModels/"
    "Rajagopal2015_passiveCal_hipAbdMoved.osim"
)
SMPL_PATH    = "/home/haziq/code/SMPL2AddBiomechanics/models/smpl/SMPL_NEUTRAL.pkl"
MAPPING_NPZ  = os.path.expanduser("~/MHR/tools/mhr_smpl_conversion/assets/smpl2mhr_mapping.npz")
J_REG_MHR_NPY = os.path.expanduser("~/MHR/tools/mhr_smpl_conversion/assets/J_regressor_mhr.npy")
FACES_NPY    = os.path.expanduser("~/MHR/faces.npy")
OUT_YAML  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "mhr_rajagopal_markers.yaml")


# T-pose: arm_add DOF indices in Rajagopal (37 DOFs total)
# [24] arm_add_r,  [31] arm_add_l  →  -90° = arms horizontal
TARGET_ARM_ADD_DEG = -90.0
HEAD_ABOVE_C7_M    = 0.135   # estimated head top above C7 marker

# ── Two-tier marker system ─────────────────────────────────────────────────────
#
# Tier 1: Surface markers  → looked up by MHR vertex ID (KD-tree)
# Tier 2: Joint center markers → J_regressor_mhr[joint_idx, :] @ mhr_verts
#          J_regressor_mhr.npy is precomputed by build_mhr_jregressor.py.
#          These are anatomically INSIDE the body (femoral head, etc.) and cannot
#          be reliably matched to skin vertices.
#
# SMPL joint indices (standard 24-joint topology):
#   0=Pelvis  1=L_Hip    2=R_Hip    3=Spine1
#   4=L_Knee  5=R_Knee   6=Spine2   7=L_Ankle
#   8=R_Ankle 9=Spine3   10=L_Foot  11=R_Foot
#  12=Neck   13=L_Collar 14=R_Collar 15=Head
#  16=L_Shoulder 17=R_Shoulder 18=L_Elbow 19=R_Elbow
#  20=L_Wrist    21=R_Wrist    22=L_Hand  23=R_Hand

# Rajagopal marker name → SMPL joint index
# (only for markers that are biomechanical joint centres, not skin landmarks)
JOINT_CENTER_MARKERS = {
    "R_HJC":  2,   # Right Hip Joint Centre   → SMPL R_Hip
    "L_HJC":  1,   # Left  Hip Joint Centre   → SMPL L_Hip
}

# Markers requiring anatomically-constrained search
# Format: name → (y_frac_lo, y_frac_hi, z_sign)
#   y_frac = fraction of body height (0=feet, 1=top)
#   z_sign = +1 for anterior (front), -1 for posterior (back), 0 = no constraint
CONSTRAINED_MARKERS = {
    "r.ASIS":   (0.52, 0.65,  +1),
    "l.ASIS":   (0.52, 0.65,  +1),
    "midASIS":  (0.52, 0.65,  +1),
    "r.PSIS":   (0.52, 0.65,  -1),
    "l.PSIS":   (0.52, 0.65,  -1),
    "midPSIS":  (0.52, 0.65,  -1),
}


# ── loaders ───────────────────────────────────────────────────────────────────

def load_rajagopal_tpose(osim_path: str):
    """Return (markers dict, skel, osim) at T-pose.

    markers: {name: world_xyz [3]}
    Also returns the skeleton so the caller can query body-node positions.
    T-pose = all DOFs zero except arm_add_r = arm_add_l = -90°.
    """
    import nimblephysics as nimble
    osim    = nimble.biomechanics.OpenSimParser.parseOsim(osim_path)
    skel    = osim.skeleton
    markers = osim.markersMap

    # T-pose DOF overrides (degrees, converted below)
    # arm_add = -90° → arms horizontal
    # pro_sup = -90° → palms facing down (matches SMPL T-pose), same sign both sides
    TPOSE_DOFS = {
        "arm_add_r": -90.0,
        "arm_add_l": -90.0,
        "pro_sup_r":  90.0,
        "pro_sup_l":  90.0,
    }
    q = np.zeros(skel.getNumDofs())
    for i in range(skel.getNumDofs()):
        dof_name = skel.getDofByIndex(i).getName()
        if dof_name in TPOSE_DOFS:
            q[i] = np.deg2rad(TPOSE_DOFS[dof_name])
    skel.setPositions(q)

    result = {}
    for name, (body, offset) in markers.items():
        world_pos = body.getWorldTransform().multiply(offset)
        result[name] = np.array(world_pos, dtype=np.float64)
    return result, skel


def rajagopal_height(raj_markers: dict, skel) -> tuple:
    """Return (floor_y, top_y, height_m) using body nodes + C7 marker.

    floor = lowest calcn body node Y (sole of foot, not just COM)
    top   = C7 marker Y + HEAD_ABOVE_C7_M (Rajagopal has no head body)
    """
    # Floor: the lowest Y among calcn body nodes
    floor_y = min(
        skel.getBodyNode(i).getWorldTransform().translation()[1]
        for i in range(skel.getNumBodyNodes())
        if "calcn" in skel.getBodyNode(i).getName()
    )
    # Ceiling: C7 marker + estimated head height
    c7_y  = raj_markers["C7"][1]
    top_y = c7_y + HEAD_ABOVE_C7_M
    return floor_y, top_y, top_y - floor_y


def load_mhr_tpose(smpl_path: str, mapping_npz: str, faces_npy: str):
    """Return (mhr_verts, mhr_faces, smpl_verts, smpl_faces, J_regressor).

    J_regressor: (24, 6890) dense array — maps SMPL verts → 24 joint centres.
    At any pose: smpl_joints = J_regressor @ smpl_verts  →  shape (24, 3)
    """
    with open(smpl_path, "rb") as f:
        model = pickle.load(f, encoding="latin1")
    smpl_verts = np.asarray(model["v_template"], dtype=np.float64)
    smpl_faces = np.asarray(model["f"],          dtype=np.int32)

    # Dense J_regressor (sparse in pkl)
    J_reg_raw = model["J_regressor"]
    if hasattr(J_reg_raw, "todense"):
        J_regressor = np.asarray(J_reg_raw.todense(), dtype=np.float64)
    else:
        J_regressor = np.asarray(J_reg_raw, dtype=np.float64)   # (24, 6890)

    mapping   = np.load(mapping_npz)
    tri_ids   = mapping["triangle_ids"]
    bary      = mapping["baryc_coords"]
    mhr_faces = np.load(faces_npy).astype(np.int32)

    tris = smpl_faces[tri_ids]
    v0   = smpl_verts[tris[:, 0]]
    v1   = smpl_verts[tris[:, 1]]
    v2   = smpl_verts[tris[:, 2]]
    mhr_verts = (bary[:, 0:1]*v0 + bary[:, 1:2]*v1 + bary[:, 2:3]*v2).astype(np.float64)
    return mhr_verts, mhr_faces, smpl_verts, smpl_faces, J_regressor


# ── core ──────────────────────────────────────────────────────────────────────

def scale_to_height(verts: np.ndarray, current_h: float, floor_y: float,
                    target_h: float) -> np.ndarray:
    """Isotropic (uniform) scale of verts to target_h metres.

    Steps:
      1. Translate so feet sit at Y=0  (subtract floor_y)
      2. Compute scale = target_h / current_h
      3. Multiply ALL x,y,z by the SAME scale factor
         → body proportions unchanged, only size changes

    Returns scaled verts with feet at Y=0.
    """
    v = verts.copy()
    v[:, 1] -= floor_y          # feet to Y=0
    v *= target_h / current_h   # uniform scale on all 3 axes
    return v


def scale_markers_to_height(markers: dict, floor_y: float,
                             current_h: float, target_h: float) -> dict:
    """Same isotropic scaling for a {name: xyz} marker dict."""
    s = target_h / current_h
    scaled = {}
    for name, pos in markers.items():
        p = pos.copy()
        p[1] -= floor_y   # feet to Y=0
        p    *= s          # uniform scale
        scaled[name] = p
    return scaled


def find_mhr_vertices(scaled_raj_markers: dict, scaled_mhr_verts: np.ndarray,
                      target_h: float) -> dict:
    """KD-tree nearest-neighbour on scaled meshes.

    Rules applied in order:
    1. Joint-center markers (JOINT_CENTER_MARKERS) are skipped — handled separately.
    2. Constrained markers (CONSTRAINED_MARKERS) restrict the search to the correct
       anatomical region (Y band + Z-sign) to avoid hemisphere errors.
    3. All other markers use global KD-tree on the full mesh.
    4. No vertex reuse: once a vertex is claimed, it is removed for subsequent
       markers at the same Y-level (prevents r_thigh1–5 all collapsing to one point).

    Returns {name: (mhr_vid, distance_m)}.
    """
    tree   = cKDTree(scaled_mhr_verts)
    result = {}
    used_vids: set = set()

    print(f"\n{'Marker':<20}  {'MHR vid':>8}  {'dist (mm)':>10}  {'note'}")
    print("-" * 56)

    for name in sorted(scaled_raj_markers):
        if name in JOINT_CENTER_MARKERS:
            print(f"{name:<20}  {'—':>8}  {'—':>10}  joint-centre (SMPL regressor)")
            continue

        pos = scaled_raj_markers[name]

        # ── Constrained search ────────────────────────────────────────────
        constraint = CONSTRAINED_MARKERS.get(name)
        if constraint is not None:
            y_lo_frac, y_hi_frac, z_sign = constraint
            y_lo = target_h * y_lo_frac
            y_hi = target_h * y_hi_frac
            mask = (scaled_mhr_verts[:, 1] >= y_lo) & (scaled_mhr_verts[:, 1] <= y_hi)
            if z_sign != 0:
                mask &= (np.sign(scaled_mhr_verts[:, 2]) == z_sign)
            cand_idx = np.where(mask)[0]
            if len(cand_idx) == 0:
                # Fallback to full mesh if region filter is empty
                cand_idx = np.arange(len(scaled_mhr_verts))
            cand_verts = scaled_mhr_verts[cand_idx]
            local_tree = cKDTree(cand_verts)
            dist, local_vid = local_tree.query(pos)
            vid = int(cand_idx[local_vid])
            note = "constrained"
        else:
            # ── Global search with no-reuse ───────────────────────────────
            # Query k=20 candidates and pick first not already used
            k = min(20, len(scaled_mhr_verts))
            dists, vids = tree.query(pos, k=k)
            vid, dist = None, None
            for d, v in zip(dists, vids):
                if int(v) not in used_vids:
                    vid, dist = int(v), d
                    break
            if vid is None:
                vid, dist = int(vids[0]), dists[0]   # last resort
            note = ""

        used_vids.add(vid)
        result[name] = (vid, float(dist))
        print(f"{name:<20}  {vid:>8}  {dist*1000:>10.2f}  {note}")

    print()
    return result


def save_yaml(mhr_markers: dict, joint_centers: dict, out_path: str):
    """Write two-tier YAML.

    surface_markers:   name: vertex_id   # dist X.XX mm
    joint_centers:     name: smpl_joint_idx   # SMPL joint name

    At runtime:
        smpl_joints = J_regressor @ smpl_verts   # (24, 3)
        pos_surface = mhr_verts[yaml['surface_markers'][name]]
        pos_jc      = smpl_joints[yaml['joint_centers'][name]]
    """
    SMPL_JOINT_NAMES = {
        0: "Pelvis", 1: "L_Hip", 2: "R_Hip", 3: "Spine1",
        4: "L_Knee", 5: "R_Knee", 6: "Spine2", 7: "L_Ankle",
        8: "R_Ankle", 9: "Spine3", 10: "L_Foot", 11: "R_Foot",
        12: "Neck", 13: "L_Collar", 14: "R_Collar", 15: "Head",
        16: "L_Shoulder", 17: "R_Shoulder", 18: "L_Elbow", 19: "R_Elbow",
        20: "L_Wrist", 21: "R_Wrist", 22: "L_Hand", 23: "R_Hand",
    }
    with open(out_path, "w") as f:
        f.write("# MHR marker mapping for Rajagopal marker set (Apache 2.0)\n")
        f.write("# Generated by find_mhr_rajagopal_markers.py\n")
        f.write("# Source: Rajagopal2015_passiveCal_hipAbdMoved.osim (Apache 2.0)\n")
        f.write("#         MHR mesh (Apache 2.0)  |  SMPL neutral (for J_regressor)\n")
        f.write("# No BSM / Max Planck commercial IP in derivation chain.\n\n")

        f.write("# ── Tier 1: Surface markers ──────────────────────────────────────\n")
        f.write("# Runtime usage:  pos = mhr_verts[vertex_id]\n")
        f.write("surface_markers:\n")
        for name in sorted(mhr_markers):
            vid, dist = mhr_markers[name]
            f.write(f"  {name}: {vid}  # dist {dist*1000:.2f} mm\n")

        f.write("\n# ── Tier 2: Joint-centre markers ─────────────────────────────────\n")
        f.write("# Runtime usage:  pos = (J_regressor_mhr @ mhr_verts)[smpl_joint_idx]\n")
        f.write("#   where J_regressor_mhr = np.load('J_regressor_mhr.npy')  # (24, 18439)\n")
        f.write("#   built offline by build_mhr_jregressor.py — no SMPL needed at runtime\n")
        f.write("joint_centers:\n")
        for name in sorted(joint_centers):
            idx = joint_centers[name]
            jname = SMPL_JOINT_NAMES.get(idx, str(idx))
            f.write(f"  {name}: {idx}  # SMPL joint: {jname}\n")

    print(f"Saved → {out_path}")
    print(f"  {len(mhr_markers)} surface markers, {len(joint_centers)} joint-centre markers")


# ── visualizer ────────────────────────────────────────────────────────────────

def visualize(osim_path: str, scaled_mhr_verts, mhr_faces,
              aligned_raj_markers, mhr_markers, target_h: float,
              raj_floor_y: float, raj_h: float, smpl_joints_mhr=None,
              z_depth_shift: float = 0.0):
    """Three-panel aitviewer scene.

    Left   (x = -X_OFFSET) : Rajagopal skeleton alone
    Middle (x = 0)         : Rajagopal skeleton + MHR mesh overlaid at origin
                             (MHR un-scaled to match Rajagopal's native height)
    Right  (x = +X_OFFSET) : MHR mesh (scaled to target_h) + yellow mapped markers
    """
    import tempfile, os, math
    import scipy.spatial.transform as st

    X_OFFSET = 2.2

    # ── Build T-pose .mot ──────────────────────────────────────────────────
    import opensim as osim_api
    model  = osim_api.Model(osim_path)
    _state = model.initSystem()
    coords = model.getCoordinateSet()
    cnames = [coords.get(i).getName() for i in range(coords.getSize())]
    TPOSE  = {
        "arm_add_r": -90.0, "arm_add_l": -90.0,
        "pro_sup_r":  90.0, "pro_sup_l":  90.0,
    }
    vals = []
    for i in range(coords.getSize()):
        c = coords.get(i)
        v = c.getDefaultValue()
        if c.getMotionType() == osim_api.Coordinate.Rotational:
            v = math.degrees(v)
        v = TPOSE.get(c.getName(), v)
        vals.append(v)
    dstr = "\t".join(f"{v:.6f}" for v in vals)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".mot", delete=False)
    tmp.write("tpose\nversion=1\nnRows=2\n"
              f"nColumns={len(cnames)+1}\ninDegrees=yes\nendheader\n")
    tmp.write("time\t" + "\t".join(cnames) + "\n")
    tmp.write(f"0.0\t{dstr}\n0.1\t{dstr}\n")
    tmp.close()
    mot_tmp = tmp.name

    # ── aitviewer config ───────────────────────────────────────────────────
    from aitviewer.configuration import CONFIG as C
    C.window_type = "glfw"
    geom_dir = os.path.join(os.path.dirname(os.path.abspath(osim_path)), "Geometry")
    if os.path.isdir(geom_dir):
        C.osim_geometry = geom_dir

    from aitviewer.renderables.osim import OSIMSequence
    from aitviewer.renderables.meshes import Meshes
    from aitviewer.renderables.spheres import Spheres
    from aitviewer.viewer import Viewer

    rot90y = st.Rotation.from_euler('y', -90, degrees=True).as_matrix().astype(np.float32)

    print("Loading Rajagopal bone meshes (left panel) ...")
    raj_left = OSIMSequence.from_files(
        osim_path=osim_path, mot_file=mot_tmp,
        fps_out=None, ignore_geometry=False, name="Rajagopal",
    )
    print("Loading Rajagopal bone meshes (middle panel) ...")
    raj_mid = OSIMSequence.from_files(
        osim_path=osim_path, mot_file=mot_tmp,
        fps_out=None, ignore_geometry=False, name="Rajagopal (overlay)",
    )
    os.unlink(mot_tmp)

    # aitviewer uses the model's default pelvis_ty to ground the skeleton (feet ~Y=0).
    # MHR scaled_mhr_verts also has feet at Y=0. Both are already grounded.

    # Left: skeleton shifted left + depth-aligned
    raj_left.position  = np.array([-X_OFFSET, 0.0, float(z_depth_shift)], dtype=np.float32)
    raj_left.rotations = np.stack([rot90y] * raj_left.n_frames, axis=0)

    # Middle: skeleton at origin + depth-aligned
    raj_mid.position  = np.array([0.0, 0.0, float(z_depth_shift)], dtype=np.float32)
    raj_mid.rotations = np.stack([rot90y] * raj_mid.n_frames, axis=0)

    # Middle: MHR at target_h (same as right panel), feet at Y=0
    mhr_mid_v = scaled_mhr_verts.copy().astype(np.float32)
    mhr_mid = Meshes(
        vertices=mhr_mid_v[np.newaxis],
        faces=mhr_faces.astype(np.int32),
        name="MHR (overlay)",
        color=(0.60, 0.78, 0.95, 0.22),
    )

    # Middle: cyan spheres for Tier-2 joint centres (HJC etc.)
    jc_spheres = None
    if smpl_joints_mhr is not None and len(JOINT_CENTER_MARKERS) > 0:
        jc_positions = np.array(
            [smpl_joints_mhr[idx].astype(np.float32) for idx in JOINT_CENTER_MARKERS.values()],
            dtype=np.float32,
        )
        jc_spheres = Spheres(
            positions=jc_positions[np.newaxis],
            radius=0.018,
            name="Joint centres (cyan)",
            color=(0.0, 0.9, 0.9, 1.0),
        )

    # Right: MHR at target_h + yellow markers
    mhr_right_v = scaled_mhr_verts.copy().astype(np.float32)
    mhr_right_v[:, 0] += X_OFFSET
    mhr_right = Meshes(
        vertices=mhr_right_v[np.newaxis],
        faces=mhr_faces.astype(np.int32),
        name="MHR mesh",
        color=(0.60, 0.78, 0.95, 0.85),
    )

    names = sorted(mhr_markers)
    mhr_sphere_positions = []
    for name in names:
        vid = mhr_markers[name][0]
        p   = scaled_mhr_verts[vid].copy().astype(np.float32)
        p[0] += X_OFFSET
        mhr_sphere_positions.append(p)

    mhr_spheres = Spheres(
        positions=np.array(mhr_sphere_positions)[np.newaxis],
        radius=0.013,
        name="MHR markers (yellow)",
        color=(1.0, 0.85, 0.1, 1.0),
    )

    # ── Launch viewer ──────────────────────────────────────────────────────
    v = Viewer(title="Rajagopal (left) | Overlay (middle) | MHR + markers (right)")
    v.run_animations = False
    v.scene.camera.position = np.array([0.0, 1.0, 6.0], dtype=np.float32)
    v.scene.add(raj_left, raj_mid, mhr_mid, mhr_right, mhr_spheres)
    if jc_spheres is not None:
        v.scene.add(jc_spheres)

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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--osim",    default=OSIM_PATH)
    parser.add_argument("--smpl",    default=SMPL_PATH)
    parser.add_argument("--target_height", type=float, default=1.70,
                        help="Scale both bodies to this height in metres (default: 1.70)")
    parser.add_argument("--no_save", action="store_true",
                        help="Print table only, do not write yaml")
    parser.add_argument("--visualize", action="store_true",
                        help="Open Open3D side-by-side viewer after computing")
    args = parser.parse_args()

    TARGET_H = args.target_height

    # ── Step 1: Rajagopal at T-pose ───────────────────────────────────────────
    print("Step 1: Loading Rajagopal markers at T-pose (arm_add = -90°) ...")
    raj_markers, skel = load_rajagopal_tpose(args.osim)
    print(f"  {len(raj_markers)} markers loaded")

    # Rajagopal height: calcn floor → C7 + estimated head
    raj_floor_y, raj_top_y, raj_h = rajagopal_height(raj_markers, skel)
    print(f"  Rajagopal: floor Y={raj_floor_y*100:.1f} cm, "
          f"top Y={raj_top_y*100:.1f} cm, height={raj_h*100:.1f} cm")

    # ── Step 2: MHR T-pose mesh ───────────────────────────────────────────────
    print("Step 2: Loading MHR T-pose via SMPL barycentric transfer ...")
    mhr_verts, mhr_faces, smpl_verts, smpl_faces, J_regressor = load_mhr_tpose(
        args.smpl, MAPPING_NPZ, FACES_NPY
    )
    print(f"  MHR verts: {mhr_verts.shape}")

    mhr_floor_y = mhr_verts[:, 1].min()
    mhr_top_y   = mhr_verts[:, 1].max()
    mhr_h       = mhr_top_y - mhr_floor_y
    print(f"  MHR:       floor Y={mhr_floor_y*100:.1f} cm, "
          f"top Y={mhr_top_y*100:.1f} cm, height={mhr_h*100:.1f} cm")

    # ── Step 3: Isotropic scale BOTH to target_height ────────────────────────
    print(f"\nStep 3: Uniformly scaling both to {TARGET_H:.2f} m "
          f"(same scale factor on X, Y, Z — proportions preserved) ...")

    scaled_mhr_verts  = scale_to_height(mhr_verts,    mhr_h,   mhr_floor_y,  TARGET_H)
    scaled_raj_markers = scale_markers_to_height(raj_markers, raj_floor_y, raj_h, TARGET_H)

    raj_scale = TARGET_H / raj_h
    mhr_scale = TARGET_H / mhr_h
    print(f"  Rajagopal scale factor: {raj_scale:.4f}  "
          f"(original {raj_h*100:.1f} cm → {TARGET_H*100:.0f} cm)")
    print(f"  MHR       scale factor: {mhr_scale:.4f}  "
          f"(original {mhr_h*100:.1f} cm → {TARGET_H*100:.0f} cm)")

    # ── Step 3b: Rotate Rajagopal markers into MHR coordinate frame ─────────
    # Nimblephysics world space: body faces +X.
    # SMPL/MHR space:            body faces +Z.
    # A -90° rotation about Y maps +X → +Z, aligning the two frames.
    # Without this the KD-tree compares front/back and left/right incorrectly.
    from scipy.spatial.transform import Rotation as _Rot
    _raj_to_mhr = _Rot.from_euler('y', -90, degrees=True).as_matrix()
    aligned_raj_markers = {
        name: _raj_to_mhr @ pos
        for name, pos in scaled_raj_markers.items()
    }

    # ── Step 3c: Depth (Z) correction ────────────────────────────────────────
    # Rajagopal defines markers at bony landmarks relative to skeletal reference
    # frames.  The entire body is systematically POSTERIOR to the MHR mesh in
    # the depth (Z) direction.  Diagnosis (from _debug_depth.py):
    #   - R_Sternum (anterior!) ends up at Z=-0.06m, MHR chest front ≈ +0.10m
    #   - C7 (posterior)  ends up at Z=-0.20m, MHR back surface   ≈ -0.12m
    #   - Mean marker Z = -0.042m vs MHR mean Z = +0.014m  →  shift ≈ +5.6cm
    #
    # The draft-KD-tree residual approach doesn't work (finds wrong matches
    # before the shift is applied).  Instead, align population means directly.
    print("\nStep 3c: Computing depth (Z) correction (population mean alignment) ...")
    _marker_zs = np.array([
        pos[2] for name, pos in aligned_raj_markers.items()
        if name not in JOINT_CENTER_MARKERS
    ])
    z_depth_shift = float(np.mean(scaled_mhr_verts[:, 2]) - np.mean(_marker_zs))
    print(f"  Marker mean Z = {np.mean(_marker_zs)*100:+.1f} cm  |  "
          f"MHR mean Z = {np.mean(scaled_mhr_verts[:,2])*100:+.1f} cm  |  "
          f"Depth shift = {z_depth_shift*100:+.1f} cm")
    aligned_raj_markers = {
        name: pos + np.array([0.0, 0.0, z_depth_shift])
        for name, pos in aligned_raj_markers.items()
    }

    # ── Step 4: Nearest MHR vertex per marker ────────────────────────────────
    print("\nStep 4: Finding nearest MHR vertex for each Rajagopal marker ...")
    mhr_markers = find_mhr_vertices(aligned_raj_markers, scaled_mhr_verts, TARGET_H)

    # ── Step 5: Tier-2 joint centres via J_regressor_mhr ─────────────────────
    # J_regressor_mhr was built offline (build_mhr_jregressor.py) and maps
    # MHR vertices directly to 24 SMPL joint centres — no SMPL needed at runtime.
    print("Step 5: Computing joint-centre markers from J_regressor_mhr ...")
    J_reg_mhr = np.load(J_REG_MHR_NPY).astype(np.float64)  # (24, 18439)
    smpl_joints_world = J_reg_mhr @ scaled_mhr_verts        # (24, 3) — already scaled
    # Rotate to MHR frame (MHR T-pose faces +Z, no rotation needed; but for
    # diagnostic display we keep same convention as aligned_raj_markers)
    smpl_joints_mhr = smpl_joints_world   # J_reg_mhr operates in MHR frame already

    joint_center_results = {}   # name → smpl_joint_idx  (no vertex ID needed)
    for name, smpl_idx in sorted(JOINT_CENTER_MARKERS.items()):
        jc_pos  = smpl_joints_mhr[smpl_idx]
        # Diagnostic: nearest MHR surface vertex to the joint centre
        from scipy.spatial import cKDTree as _KDTree2
        dist_to_surf = float(_KDTree2(scaled_mhr_verts).query(jc_pos)[0])
        print(f"  {name:<20}  smpl_joint={smpl_idx:>2}  "
              f"depth_into_body={dist_to_surf*1000:.1f} mm")
        joint_center_results[name] = smpl_idx

    if not args.no_save:
        save_yaml(mhr_markers, joint_center_results, OUT_YAML)

    if args.visualize:
        visualize(args.osim, scaled_mhr_verts, mhr_faces,
                  aligned_raj_markers, mhr_markers, TARGET_H,
                  raj_floor_y, raj_h, smpl_joints_mhr,
                  z_depth_shift=z_depth_shift)


if __name__ == "__main__":
    main()
