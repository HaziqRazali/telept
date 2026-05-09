#!/usr/bin/env python3
"""
visualize_smpl_bsm_markers.py — Visualize BSM marker positions on SMPL and MHR meshes.

Loads the neutral SMPL template (T-pose, zero shape), places coloured spheres at every
BSM marker vertex (105 markers from bsm_markers.yaml).
Also computes the MHR mesh via barycentric transfer and finds the nearest MHR vertex
for each SMPL marker position (KD-tree), then draws the same markers on the MHR mesh.

Usage
-----
    conda activate addbiomechanics
    cd ~/datasets/telept/my_scripts/opensim_scripts

    python visualize_smpl_bsm_markers.py                        # both meshes + markers
    python visualize_smpl_bsm_markers.py --no_mhr               # SMPL-only view
    python visualize_smpl_bsm_markers.py --no_labels            # skip text labels
    python visualize_smpl_bsm_markers.py --save_mhr_markers     # write mhr_markers.yaml
    python visualize_smpl_bsm_markers.py \\
        --smpl_model   ~/code/SMPL2AddBiomechanics/models/smpl/SMPL_NEUTRAL.pkl \\
        --markers_yaml ~/code/SMPL2AddBiomechanics/smpl2ab/data/bsm_markers.yaml

Outputs
-------
    Console   — table of  marker_name  →  MHR vertex id  (distance in mm)
    --save_mhr_markers writes mhr_markers.yaml next to this script

Controls
--------
    Left-drag  → rotate     Right-drag → pan     Scroll → zoom
    Q / Esc    → quit
"""

import argparse
import colorsys
import os
import pickle

import numpy as np
from scipy.spatial import cKDTree
import yaml
import open3d as o3d


# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_SMPL_MODEL   = "/home/haziq/code/SMPL2AddBiomechanics/models/smpl/SMPL_NEUTRAL.pkl"
DEFAULT_MARKERS_YAML = "/home/haziq/code/SMPL2AddBiomechanics/smpl2ab/data/bsm_markers.yaml"

SPHERE_RADIUS  = 0.012   # metres (~1.2 cm)
MHR_X_OFFSET   = 1.5    # metres — how far right to shift the MHR mesh


# ── loaders ───────────────────────────────────────────────────────────────────

def load_smpl_tpose(pkl_path: str):
    """Return (vertices [6890,3] float64, faces [F,3] int32) for the T-pose."""
    with open(pkl_path, "rb") as f:
        model = pickle.load(f, encoding="latin1")
    verts = np.asarray(model["v_template"], dtype=np.float64)   # [6890, 3]
    faces = np.asarray(model["f"],          dtype=np.int32)     # [13776, 3]
    return verts, faces


def load_markers(yaml_path: str) -> dict:
    """Return {marker_name: vertex_id} from bsm_markers.yaml."""
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return {k: int(v) for k, v in raw.items()}


def load_mhr_from_smpl(smpl_verts: np.ndarray, smpl_faces: np.ndarray):
    """Return (mhr_verts [18439,3], mhr_faces [36874,3]) by barycentric transfer
    of SMPL T-pose vertices onto the MHR mesh topology.

    Uses ~/MHR/tools/mhr_smpl_conversion/assets/smpl2mhr_mapping.npz which stores:
        triangle_ids  (18439,)    index into smpl_faces for each MHR vertex
        baryc_coords  (18439, 3)  barycentric weights (w0, w1, w2) within that triangle

    MHR vertex i = w0*smpl_verts[tri[0]] + w1*smpl_verts[tri[1]] + w2*smpl_verts[tri[2]]

    No torch / pymomentum / mhr package needed — pure numpy.
    Produces an MHR mesh in the exact same pose as the input SMPL mesh.
    """
    mapping_npz = os.path.expanduser(
        "~/MHR/tools/mhr_smpl_conversion/assets/smpl2mhr_mapping.npz"
    )
    faces_npy = os.path.expanduser("~/MHR/faces.npy")

    if not os.path.exists(mapping_npz):
        raise FileNotFoundError(f"SMPL→MHR mapping not found: {mapping_npz}")
    if not os.path.exists(faces_npy):
        raise FileNotFoundError(f"MHR faces.npy not found: {faces_npy}")

    print("  Loading SMPL→MHR barycentric mapping ...")
    mapping   = np.load(mapping_npz)
    tri_ids   = mapping["triangle_ids"]   # (18439,)
    bary      = mapping["baryc_coords"]   # (18439, 3)
    mhr_faces = np.load(faces_npy).astype(np.int32)   # (36874, 3)

    # smpl_faces[tri_ids] → (18439, 3) vertex indices into smpl_verts
    tris      = smpl_faces[tri_ids]              # (18439, 3)
    v0 = smpl_verts[tris[:, 0]]                 # (18439, 3)
    v1 = smpl_verts[tris[:, 1]]
    v2 = smpl_verts[tris[:, 2]]

    mhr_verts = (bary[:, 0:1] * v0 +
                 bary[:, 1:2] * v1 +
                 bary[:, 2:3] * v2).astype(np.float64)   # (18439, 3)

    print(f"  MHR verts computed via barycentric transfer: {mhr_verts.shape}")
    return mhr_verts, mhr_faces


def find_mhr_markers(smpl_verts: np.ndarray, markers: dict, mhr_verts: np.ndarray) -> dict:
    """For each BSM marker (SMPL vertex id) find the nearest MHR vertex id.

    smpl_verts : (6890, 3)  SMPL T-pose vertices
    markers    : {name: smpl_vertex_id}  from bsm_markers.yaml
    mhr_verts  : (18439, 3) MHR vertices in the same coordinate space (barycentric transfer)

    Returns {name: (mhr_vertex_id, distance_metres)}, sorted by name.
    Also prints a table to console.
    """
    tree = cKDTree(mhr_verts)   # build once, query 105 times

    result = {}
    print(f"\n{'Marker':<10}  {'SMPL vid':>8}  {'MHR vid':>8}  {'dist (mm)':>10}")
    print("-" * 44)
    for name in sorted(markers):
        smpl_vid    = markers[name]
        smpl_pos    = smpl_verts[smpl_vid]          # (3,)
        dist, mhr_vid = tree.query(smpl_pos)        # nearest MHR vertex
        result[name] = (int(mhr_vid), float(dist))
        print(f"{name:<10}  {smpl_vid:>8}  {mhr_vid:>8}  {dist*1000:>10.2f}")
    print()
    return result


# ── geometry builders ─────────────────────────────────────────────────────────

def _hsv_palette(n: int):
    """Spread n colours evenly around the hue wheel."""
    return [list(colorsys.hsv_to_rgb(i / n, 0.85, 0.95)) for i in range(n)]


def build_body_mesh(verts, faces, color=(0.78, 0.78, 0.78)) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(list(color))
    return mesh


def build_marker_spheres(verts: np.ndarray, markers: dict, x_offset: float = 0.0):
    """Return list of (name, pos_xyz [3], sphere_mesh) sorted alphabetically.

    markers  : {name: vertex_id}  OR  {name: (vertex_id, dist)}  (MHR variant)
    x_offset : applied to all sphere X positions (for side-by-side layout)
    """
    palette = _hsv_palette(len(markers))
    items   = []
    for i, name in enumerate(sorted(markers)):
        val = markers[name]
        vid = val[0] if isinstance(val, tuple) else val
        pos = verts[vid].copy()
        pos[0] += x_offset
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=SPHERE_RADIUS, resolution=8)
        sphere.translate(pos)
        sphere.paint_uniform_color(palette[i])
        sphere.compute_vertex_normals()
        items.append((name, pos, sphere))
    return items


# ── viewers ───────────────────────────────────────────────────────────────────

def run_gui_viewer(smpl_mesh, mhr_mesh, smpl_marker_items, mhr_marker_items, show_labels: bool):
    """Open3D GUI viewer with optional 3-D text labels (requires Open3D ≥ 0.14)."""
    import open3d.visualization.gui       as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()

    title = "SMPL + BSM Markers" if mhr_mesh is None else "SMPL + BSM Markers  |  MHR + markers (right)"
    win    = app.create_window(title, 1600, 900)
    widget = gui.SceneWidget()
    widget.scene = rendering.Open3DScene(win.renderer)
    win.add_child(widget)

    widget.scene.set_background([0.13, 0.13, 0.13, 1.0])

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"

    widget.scene.add_geometry("smpl", smpl_mesh, mat)

    if mhr_mesh is not None:
        widget.scene.add_geometry("mhr", mhr_mesh, mat)

    for name, pos, sphere in smpl_marker_items:
        widget.scene.add_geometry(f"sm_{name}", sphere, mat)
        if show_labels:
            lbl       = widget.add_3d_label(pos.tolist(), name)
            lbl.color = gui.Color(1.0, 1.0, 0.25)   # yellow

    for name, pos, sphere in mhr_marker_items:
        widget.scene.add_geometry(f"mm_{name}", sphere, mat)
        if show_labels:
            lbl       = widget.add_3d_label(pos.tolist(), name)
            lbl.color = gui.Color(0.4, 1.0, 0.4)    # green for MHR side

    bounds = widget.scene.bounding_box
    widget.setup_camera(60.0, bounds, bounds.get_center())
    widget.scene.scene.enable_sun_light(True)

    app.run()


def run_simple_viewer(smpl_mesh, mhr_mesh, smpl_marker_items, mhr_marker_items):
    """Fallback viewer using draw_geometries (no text labels)."""
    geoms = [smpl_mesh]
    if mhr_mesh is not None:
        geoms.append(mhr_mesh)
    geoms += [sphere for _, _, sphere in smpl_marker_items]
    geoms += [sphere for _, _, sphere in mhr_marker_items]
    title = "SMPL + BSM Markers" if mhr_mesh is None else "SMPL + BSM Markers  |  MHR + markers (right)"
    o3d.visualization.draw_geometries(
        geoms,
        window_name=title,
        width=1600,
        height=900,
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--smpl_model", default=DEFAULT_SMPL_MODEL,
        help="Path to SMPL_NEUTRAL.pkl (or MALE/FEMALE)",
    )
    parser.add_argument(
        "--markers_yaml", default=DEFAULT_MARKERS_YAML,
        help="Path to bsm_markers.yaml",
    )
    parser.add_argument(
        "--no_labels", action="store_true",
        help="Skip 3-D text labels (faster / less clutter)",
    )
    parser.add_argument(
        "--no_mhr", action="store_true",
        help="Skip MHR mesh (SMPL-only view)",
    )
    parser.add_argument(
        "--save_mhr_markers", action="store_true",
        help="Save MHR marker→vertex mapping to mhr_markers.yaml next to this script",
    )
    args = parser.parse_args()

    # ── SMPL ──────────────────────────────────────────────────────────────────
    print(f"Loading SMPL model  …  {args.smpl_model}")
    smpl_verts, smpl_faces = load_smpl_tpose(args.smpl_model)
    print(f"  vertices {smpl_verts.shape}  faces {smpl_faces.shape}")

    print(f"Loading markers     …  {args.markers_yaml}")
    markers = load_markers(args.markers_yaml)
    print(f"  {len(markers)} markers loaded")

    smpl_mesh        = build_body_mesh(smpl_verts, smpl_faces, color=(0.78, 0.78, 0.78))
    smpl_marker_items = build_marker_spheres(smpl_verts, markers)

    # ── MHR (optional) ────────────────────────────────────────────────────────
    mhr_mesh         = None
    mhr_marker_items = []
    if not args.no_mhr:
        try:
            print("Loading MHR mesh (barycentric transfer from SMPL T-pose) ...")
            mhr_verts, mhr_faces = load_mhr_from_smpl(smpl_verts, smpl_faces)
            print(f"  vertices {mhr_verts.shape}  faces {mhr_faces.shape}")

            # Find nearest MHR vertex for each BSM marker (same coord space)
            print("Finding nearest MHR vertex for each BSM marker ...")
            mhr_markers = find_mhr_markers(smpl_verts, markers, mhr_verts)

            # Optionally save the mapping
            if args.save_mhr_markers:
                out_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "mhr_markers.yaml")
                with open(out_yaml, "w") as f:
                    f.write("# BSM marker name → nearest MHR vertex id\n")
                    f.write("# Generated by visualize_smpl_bsm_markers.py\n")
                    for name in sorted(mhr_markers):
                        vid, dist = mhr_markers[name]
                        f.write(f"{name}: {vid}  # dist {dist*1000:.2f} mm\n")
                print(f"  Saved MHR marker mapping → {out_yaml}")

            # Shift MHR rightward and align vertical centres
            smpl_cy           = (smpl_verts[:, 1].max() + smpl_verts[:, 1].min()) / 2
            mhr_cy            = (mhr_verts[:, 1].max()  + mhr_verts[:, 1].min())  / 2
            y_shift           = smpl_cy - mhr_cy
            mhr_verts_shifted = mhr_verts.copy()
            mhr_verts_shifted[:, 0] += MHR_X_OFFSET
            mhr_verts_shifted[:, 1] += y_shift

            mhr_mesh = build_body_mesh(mhr_verts_shifted, mhr_faces, color=(0.60, 0.78, 0.95))
            print(f"  MHR mesh placed at x+{MHR_X_OFFSET:.1f} m (right of SMPL)")

            # Marker spheres on MHR: apply same x/y shift so they sit on the shifted mesh
            mhr_verts_for_spheres = mhr_verts.copy()
            mhr_verts_for_spheres[:, 1] += y_shift
            mhr_marker_items = build_marker_spheres(
                mhr_verts_for_spheres, mhr_markers, x_offset=MHR_X_OFFSET
            )

        except Exception as e:
            print(f"  [WARN] MHR not available ({e}) — showing SMPL only.")

    # ── visualize ─────────────────────────────────────────────────────────────
    try:
        import open3d.visualization.gui as _gui   # noqa: F401
        run_gui_viewer(smpl_mesh, mhr_mesh, smpl_marker_items, mhr_marker_items,
                       show_labels=not args.no_labels)
    except (ImportError, AttributeError):
        print("open3d.visualization.gui not available — falling back to draw_geometries (no labels)")
        run_simple_viewer(smpl_mesh, mhr_mesh, smpl_marker_items, mhr_marker_items)


if __name__ == "__main__":
    main()
