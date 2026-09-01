#!/usr/bin/env python3
"""
Resistance band force synthesis + OpenSim Inverse Dynamics.

Works for any resistance band exercise (bicep curl, leg extension, etc.).

Steps:
  1. SMPL FK  → attachment-body world positions over time
  2. Anchor   = (wrist_x[0], floor_y, wrist_z[0])   (Y-up coords)
  3. Force    = constant force_mag N, direction toward anchor each frame
  4. Write    external_forces.sto  (force in ground frame, applied to hand body)
              external_loads.xml
  5. Run      opensim.InverseDynamicsTool  →  inverse_dynamics.sto
  6. Parse    joint moments
  7. Save     .npz for show_ab_results.py  (pass as --id_path, pipeline Step 4)

Usage (from ~/code/SMPL2AddBiomechanics, pipeline Step 3):
  conda activate addbiomechanics
  python ~/datasets/telept/my_scripts/opensim_scripts/resistance_band_id.py \\
      --osim_path output/dumbbell_biceps_curls/dumbbell_biceps_curls/osim_results/Models/match_markers_but_ignore_physics.osim \\
      --mot_path  output/dumbbell_biceps_curls/dumbbell_biceps_curls/osim_results/IK/dumbbell_biceps_curls_segment_0_ik.mot \\
      --npz       /tmp/smpl2ab_dumbbell_biceps_curls/dumbbell_biceps_curls/dumbbell_biceps_curls.npz \\
      --output    output/dumbbell_biceps_curls/band_id.npz \\
      --force_n   49.05
"""

import argparse
import os
import sys
import numpy as np
import torch

# SMPL joint indices (24 joints)
_SMPL_IDX = {
    'l_shoulder': 16, 'r_shoulder': 17,
    'l_elbow':    18, 'r_elbow':    19,
    'l_wrist':    20, 'r_wrist':    21,
    'l_foot':     10, 'r_foot':     11,
}

# BSM body name for band attachment (hand body origin ≈ wrist joint center)
_BAND_BODY = {'left': 'hand_l', 'right': 'hand_r'}

# BSM coordinate names → ID output will be "<coord>_moment"
_ELBOW_COORD    = {'left': 'elbow_flexion_l',  'right': 'elbow_flexion_r'}
_SHOULDER_COORDS = {
    'left':  ['shoulder_l_x', 'shoulder_l_y', 'shoulder_l_z'],
    'right': ['shoulder_r_x', 'shoulder_r_y', 'shoulder_r_z'],
}

# Arm-side columns to pre-select in the GUI torque panel
_ARM_COLS = {
    'left': [
        'elbow_flexion_l_moment',
        'pro_sup_l_moment',
        'wrist_flexion_l_moment',
        'wrist_deviation_l_moment',
        'shoulder_l_x_moment',
        'shoulder_l_y_moment',
        'shoulder_l_z_moment',
        'scapula_abduction_l_moment',
        'scapula_elevation_l_moment',
        'scapula_upward_rot_l_moment',
    ],
    'right': [
        'elbow_flexion_r_moment',
        'pro_sup_r_moment',
        'wrist_flexion_r_moment',
        'wrist_deviation_r_moment',
        'shoulder_r_x_moment',
        'shoulder_r_y_moment',
        'shoulder_r_z_moment',
        'scapula_abduction_r_moment',
        'scapula_elevation_r_moment',
        'scapula_upward_rot_r_moment',
    ],
}


# ---------------------------------------------------------------------------
# Step 1: SMPL FK
# ---------------------------------------------------------------------------

def _load_smpl_npz(npz_path):
    """Load a SMPL/SMPLX .npz without importing smpl2ab (avoids 'config' dep)."""
    raw = np.load(npz_path, allow_pickle=True)
    d   = {k: raw[k] for k in raw.keys()}
    if not isinstance(d['gender'], str):
        d['gender'] = str(d['gender'])
    # SMPLX has 156-dim poses; keep only SMPL 72 body params
    if d['poses'].shape[1] == 156:
        p = np.zeros((d['poses'].shape[0], 72))
        p[:, :66] = d['poses'][:, :66]
        d['poses'] = p
    fps_key = next((k for k in d if k.endswith('rate')), None)
    d['fps'] = float(d[fps_key]) if fps_key else None
    return d


def _smpl_joint_positions(npz_path, fps_out=30):
    """Run SMPL FK → (joints (F,24,3) Y-up, time (F,))."""
    from aitviewer.models.smpl import SMPLLayer
    from aitviewer.utils.so3 import resample_rotations
    from aitviewer.utils.utils import resample_positions

    data   = _load_smpl_npz(npz_path)
    gender = data.get('gender', 'neutral')
    fps_in = float(data.get('fps') or fps_out)
    poses  = data['poses']   # (F, 72)
    trans  = data['trans']   # (F, 3)
    betas  = data['betas']   # (N,)

    if fps_in != fps_out:
        ps    = poses.reshape([-1, poses.shape[1] // 3, 3])
        poses = resample_rotations(ps, fps_in, fps_out).reshape([-1, poses.shape[1]])
        trans = resample_positions(trans, fps_in, fps_out)

    n       = poses.shape[0]
    smpl    = SMPLLayer(model_type='smpl', gender=gender)
    betas_t = torch.tensor(betas[:10], dtype=torch.float32).unsqueeze(0).expand(n, -1)
    with torch.no_grad():
        _, joints_t = smpl(
            poses_body=torch.tensor(poses[:, 3:72], dtype=torch.float32),
            poses_root=torch.tensor(poses[:, :3],   dtype=torch.float32),
            betas=betas_t,
            trans=torch.tensor(trans, dtype=torch.float32),
        )
    joints = joints_t.numpy()[:, :24, :]  # (F, 24, 3)  Y-up (first 24 = SMPL joints)
    time   = np.arange(n) / float(fps_out)
    return joints, time


# ---------------------------------------------------------------------------
# Step 2: read IK .mot times
# ---------------------------------------------------------------------------

def _load_mot_times(mot_path):
    """Return time array (T,) from an OpenSim .mot file."""
    with open(mot_path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines)
                 if l.strip().lower() == 'endheader') + 2
    return np.array([float(l.split()[0]) for l in lines[start:] if l.strip()],
                    dtype=np.float64)


# ---------------------------------------------------------------------------
# Steps 4a/4b: write .sto and ExternalLoads XML
# ---------------------------------------------------------------------------

def _write_sto(path, times, force_vecs, force_name='band_force'):
    """
    Write OpenSim external-forces .sto.
    force_vecs : (T, 3)  N, expressed in ground frame
    Application point is the origin of the applying body (all zeros, body frame).
    """
    hdr = (
        f"{force_name}\n"
        "version=1\n"
        f"nRows={len(times)}\n"
        "nColumns=7\n"
        "inDegrees=no\n"
        "endheader\n"
        f"time\t{force_name}_vx\t{force_name}_vy\t{force_name}_vz"
        f"\t{force_name}_px\t{force_name}_py\t{force_name}_pz\n"
    )
    rows = [
        f"{t:.6f}\t{fx:.6f}\t{fy:.6f}\t{fz:.6f}\t0.000000\t0.000000\t0.000000"
        for t, (fx, fy, fz) in zip(times, force_vecs)
    ]
    with open(path, 'w') as f:
        f.write(hdr + '\n'.join(rows) + '\n')


def _write_ext_loads_xml(path, sto_path_abs, body_name, force_name='band_force'):
    """
    Write OpenSim ExternalLoads XML.
    Force is expressed in ground frame; point is at body origin (body frame zeros).
    """
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <ExternalLoads name="external_loads">
        <objects>
            <ExternalForce name="{force_name}">
                <applied_to_body>{body_name}</applied_to_body>
                <force_expressed_in_body>ground</force_expressed_in_body>
                <point_expressed_in_body>{body_name}</point_expressed_in_body>
                <force_identifier>{force_name}_v</force_identifier>
                <point_identifier>{force_name}_p</point_identifier>
                <data_source_name>{sto_path_abs}</data_source_name>
            </ExternalForce>
        </objects>
        <datafile>{sto_path_abs}</datafile>
    </ExternalLoads>
</OpenSimDocument>
"""
    with open(path, 'w') as f:
        f.write(xml)


# ---------------------------------------------------------------------------
# Step 5: run InverseDynamicsTool via setup XML
# ---------------------------------------------------------------------------

def _write_id_setup_xml(path, osim_path, mot_path, ext_loads_xml,
                         results_dir, t_start, t_end,
                         out_filename='inverse_dynamics.sto'):
    """Write InverseDynamicsTool setup XML (all absolute paths)."""
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <InverseDynamicsTool name="InverseDynamics">
        <model_file>{osim_path}</model_file>
        <time_range>{t_start:.6f} {t_end:.6f}</time_range>
        <forces_to_exclude>Muscles</forces_to_exclude>
        <output_gen_force_file>{out_filename}</output_gen_force_file>
        <coordinates_file>{mot_path}</coordinates_file>
        <external_loads_file>{ext_loads_xml}</external_loads_file>
        <results_directory>{results_dir}</results_directory>
        <lowpass_cutoff_frequency_for_coordinates>6</lowpass_cutoff_frequency_for_coordinates>
    </InverseDynamicsTool>
</OpenSimDocument>
"""
    with open(path, 'w') as f:
        f.write(xml)


def _run_id(osim_path, mot_path, ext_loads_xml, results_dir):
    """Run InverseDynamicsTool, return path to output .sto."""
    try:
        import opensim as osim
    except ImportError:
        sys.exit("ERROR: opensim not found — conda activate addbiomechanics")

    os.makedirs(results_dir, exist_ok=True)
    out_filename = 'inverse_dynamics.sto'

    ik_times = _load_mot_times(mot_path)
    t_start, t_end = float(ik_times[0]), float(ik_times[-1])

    setup_xml = os.path.join(results_dir, 'id_setup.xml')
    _write_id_setup_xml(
        path=setup_xml,
        osim_path=os.path.abspath(osim_path),
        mot_path=os.path.abspath(mot_path),
        ext_loads_xml=os.path.abspath(ext_loads_xml),
        results_dir=os.path.abspath(results_dir),
        t_start=t_start, t_end=t_end,
        out_filename=out_filename,
    )

    print(f"  setup XML → {setup_xml}")
    tool = osim.InverseDynamicsTool(setup_xml)
    success = tool.run()
    if not success:
        sys.exit("ERROR: InverseDynamicsTool.run() returned False — check OpenSim console output above")

    return os.path.join(results_dir, out_filename)


# ---------------------------------------------------------------------------
# Step 6: parse .sto
# ---------------------------------------------------------------------------

def _parse_sto(sto_path):
    """Return (col_names list, data ndarray (T, N_cols))."""
    with open(sto_path) as f:
        lines = f.readlines()
    start = next(i for i, l in enumerate(lines)
                 if l.strip().lower() == 'endheader') + 1
    col_names = lines[start].split()
    data = np.array(
        [[float(v) for v in l.split()] for l in lines[start + 1:] if l.strip()],
        dtype=np.float64,
    )
    return col_names, data


# ---------------------------------------------------------------------------
# Band endpoints helpers
# ---------------------------------------------------------------------------

def _load_endpoints_npz(path, ik_times):
    """
    Load band endpoints from a user-supplied .npz.

    Expected keys
    -------------
    anchor_pos  (3,)    fixed end of band (wall / floor attachment), Y-up world frame (m)
    limb_pos    (T, 3)  moving end attached to the user limb, Y-up world frame (m)
    time        (T,)    seconds — must overlap the IK .mot time range (within 0.5 s)

    Returns (anchor_pos (3,), limb_pos (T_ik, 3)) resampled onto the IK time axis.
    """
    raw = np.load(path)

    for key in ('anchor_pos', 'limb_pos', 'time'):
        assert key in raw, (
            f"endpoints .npz is missing key '{key}'. "
            f"Required keys: anchor_pos (3,), limb_pos (T,3), time (T,)")

    anchor_pos = raw['anchor_pos'].astype(np.float64)
    limb_pos   = raw['limb_pos'].astype(np.float64)
    ep_time    = raw['time'].astype(np.float64)

    assert anchor_pos.shape == (3,), \
        f"anchor_pos must be shape (3,), got {anchor_pos.shape}"
    assert limb_pos.ndim == 2 and limb_pos.shape[1] == 3, \
        f"limb_pos must be shape (T, 3), got {limb_pos.shape}"
    assert ep_time.ndim == 1, \
        f"time must be 1-D, got shape {ep_time.shape}"
    assert ep_time.shape[0] == limb_pos.shape[0], (
        f"time length ({ep_time.shape[0]}) must equal "
        f"limb_pos first dimension ({limb_pos.shape[0]})")

    tol = 0.5  # seconds
    assert abs(ep_time[0] - ik_times[0]) < tol, (
        f"endpoints time start ({ep_time[0]:.3f} s) differs from IK start "
        f"({ik_times[0]:.3f} s) by more than {tol} s")
    assert abs(ep_time[-1] - ik_times[-1]) < tol, (
        f"endpoints time end ({ep_time[-1]:.3f} s) differs from IK end "
        f"({ik_times[-1]:.3f} s) by more than {tol} s")

    # Resample limb_pos onto IK time axis
    limb_pos_ik = np.stack(
        [np.interp(ik_times, ep_time, limb_pos[:, ax]) for ax in range(3)],
        axis=1)  # (T_ik, 3)

    return anchor_pos, limb_pos_ik


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def simulate(osim_path, mot_path, endpoints_path, output_path,
             force_n=9.81, arm='left', work_dir=None,
             save_endpoints_path=None):
    """
    Full pipeline: band endpoints → OpenSim ID → save .npz.

    Parameters
    ----------
    osim_path           : scaled .osim from SMPL2AddBiomechanics pipeline
    mot_path            : IK .mot from pipeline
    endpoints_path      : path to band_endpoints.npz (anchor_pos, limb_pos, time)
    output_path         : where to save the output .npz
    force_n             : band force magnitude in Newtons
    arm                 : 'left' or 'right'
    work_dir            : directory for intermediate files (default: next to output)
    save_endpoints_path : optional path to re-save the loaded endpoints .npz
    """
    body_name = _BAND_BODY[arm]

    # ── 1. Load IK time axis ────────────────────────────────────────────────
    print("[1/6] Reading IK time axis...")
    ik_times = _load_mot_times(mot_path)   # (T,)
    n_frames  = len(ik_times)

    # ── 2. Load band endpoints ──────────────────────────────────────────────
    print("[2/6] Loading band endpoints...")
    anchor, limb = _load_endpoints_npz(endpoints_path, ik_times)
    print(f"         anchor_pos = {anchor}")

    # ── 4. Band force vectors ────────────────────────────────────────────────
    print("[3/6] Computing band force vectors...")
    force_mag  = force_n   # N
    band_vec   = anchor[np.newaxis] - limb               # (T, 3)  toward anchor
    band_len   = np.linalg.norm(band_vec, axis=1, keepdims=True).clip(min=1e-8)
    force_vecs = band_vec / band_len * force_mag          # (T, 3)  in ground frame

    # ── Save endpoints if requested ──────────────────────────────────────────
    if save_endpoints_path:
        ep_dir = os.path.dirname(os.path.abspath(save_endpoints_path))
        if ep_dir:
            os.makedirs(ep_dir, exist_ok=True)
        np.savez(
            save_endpoints_path,
            anchor_pos=anchor,    # (3,)
            limb_pos=limb,        # (T, 3)
            time=ik_times,        # (T,)
        )
        print(f"  Band endpoints → {save_endpoints_path}")

    # ── 5. Write intermediate files ──────────────────────────────────────────
    if work_dir is None:
        work_dir = os.path.join(os.path.dirname(os.path.abspath(output_path)), 'id_work')
    os.makedirs(work_dir, exist_ok=True)

    sto_path = os.path.join(work_dir, 'external_forces.sto')
    xml_path = os.path.join(work_dir, 'external_loads.xml')
    _write_sto(sto_path, ik_times, force_vecs)
    _write_ext_loads_xml(xml_path, os.path.abspath(sto_path), body_name)
    print(f"[4/6] External forces → {sto_path}")

    # ── 6. Run OpenSim ID ────────────────────────────────────────────────────
    results_dir = os.path.join(work_dir, 'id_results')
    print("[5/6] Running OpenSim Inverse Dynamics...")
    id_sto = _run_id(osim_path, mot_path, xml_path, results_dir)

    # ── 7. Parse ID output ───────────────────────────────────────────────────
    print("[6/6] Parsing ID output...")
    id_cols, id_data = _parse_sto(id_sto)
    col_idx = {c: i for i, c in enumerate(id_cols)}

    def _get_col(name):
        if name not in col_idx:
            print(f"  [warning] column '{name}' not found in ID output — using zeros")
            return np.zeros(len(id_data))
        return id_data[:, col_idx[name]]

    elbow_col        = _ELBOW_COORD[arm] + '_moment'
    shldr_cols       = [c + '_moment' for c in _SHOULDER_COORDS[arm]]
    tau_elbow        = _get_col(elbow_col)                                    # (T,)
    tau_shoulder     = np.stack([_get_col(c) for c in shldr_cols], axis=1)   # (T, 3)
    tau_shoulder_mag = np.linalg.norm(tau_shoulder, axis=1)                   # (T,)

    # Store all non-time columns so the GUI can show any joint
    id_col_names = [c for c in id_cols if c != 'time']
    id_col_data  = id_data[:, [col_idx[c] for c in id_col_names]]

    # Which columns to show pre-selected in the GUI panel (arm-side joints)
    arm_col_names = [c for c in _ARM_COLS[arm] if c in set(id_col_names)]

    # ── 8. Save .npz ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(
        output_path,
        anchor_pos=anchor,                  # (3,)      fixed end of band, Y-up world frame
        limb_pos=limb,                      # (T, 3)    moving end (limb attachment), Y-up
        force_vec=force_vecs,               # (T, 3)  N, Y-up ground frame
        band_len=band_len[:, 0],            # (T,)    metres
        time=ik_times,                      # (T,)    seconds
        tau_elbow=tau_elbow,                # (T,)    N·m  (elbow_flexion)
        tau_shoulder_mag=tau_shoulder_mag,  # (T,)    N·m  (GH resultant)
        tau_shoulder=tau_shoulder,          # (T, 3)  N·m  (x/y/z)
        id_col_names=np.array(id_col_names),
        id_data=id_col_data,                # (T, N)  N·m  all moments
        arm_col_names=np.array(arm_col_names),
        force_n=np.array(force_n),
        arm=np.array(arm),
    )

    print(f"\n✓ Saved → {output_path}")
    print(f"  Frames      : {n_frames}   Duration: {ik_times[-1]:.1f} s")
    print(f"  Anchor      : {anchor}")
    print(f"  Band length : {band_len.min():.2f} – {band_len.max():.2f} m")
    print(f"  τ_elbow     : peak {np.abs(tau_elbow).max():.1f} N·m")
    print(f"  τ_shoulder  : peak {tau_shoulder_mag.max():.1f} N·m")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resistance band force synthesis + OpenSim Inverse Dynamics')
    parser.add_argument('--osim_path', required=True,
                        help='Scaled .osim from pipeline (match_markers_but_ignore_physics.osim)')
    parser.add_argument('--mot_path',  required=True,
                        help='IK .mot from pipeline (*_segment_0_ik.mot)')
    parser.add_argument('--endpoints', required=True,
                        help='Path to band_endpoints.npz with keys: anchor_pos (3,), limb_pos (T,3), time (T,)')
    parser.add_argument('--output',    required=True,
                        help='Output .npz path (band_id.npz)')
    parser.add_argument('--force_n',   type=float, default=9.81,
                        help='Band force magnitude in Newtons (default: 9.81 N = ~1 kg load)')
    parser.add_argument('--arm',       default='left', choices=['left', 'right'],
                        help='Which arm/leg the band attaches to (default: left)')
    parser.add_argument('--save_endpoints', default=None, metavar='PATH',
                        help='Re-save the loaded endpoints to a new .npz (e.g. for archiving)')
    parser.add_argument('--work_dir',  default=None,
                        help='Directory for intermediate files (default: next to output)')
    args = parser.parse_args()

    simulate(
        osim_path=args.osim_path,
        mot_path=args.mot_path,
        endpoints_path=args.endpoints,
        output_path=args.output,
        force_n=args.force_n,
        arm=args.arm,
        work_dir=args.work_dir,
        save_endpoints_path=args.save_endpoints,
    )
