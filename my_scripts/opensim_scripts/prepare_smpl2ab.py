"""
prepare_smpl2ab.py — Convert various SMPL-X source formats to the .npz format
expected by run_smpl2bsm.sh (i.e. smpl2addbio.py).

Supported input formats
-----------------------
  fit3d JSON   e.g. .../fit3d/train/s03/smplx/band_pull_apart.json
                    keys: global_orient (T,1,3,3), body_pose (T,21,3,3),
                          transl (T,3), betas (T,10)   [rotation matrices]

Output .npz keys (required by smpl2addbio.py / load_smpl_seq)
--------------------------------------------------------------
  poses          (T, 72)  axis-angle: global_orient(3) + body(63) + zeros(6)
  trans          (T, 3)
  betas          (10,)    first frame (constant across sequence)
  gender         str      'neutral' unless known
  mocap_framerate float

Usage
-----
  python prepare_smpl2ab.py \
      --input  /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json \
      --output /home/haziq/datasets/mocap/data/fit3d/train/s03/addbiomechanics \
      --fps    50 \
      --gender neutral

  Output layout (one subdir per trial, safe to reuse --output for all JSONs):
      addbiomechanics/
          band_pull_apart/
              band_pull_apart.npz
          squat/
              squat.npz
          ...

  Then run the pipeline on one trial:
      bash run_smpl2bsm.sh .../addbiomechanics/band_pull_apart/ <osim_output_dir>
"""

import argparse
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

def resample(poses, trans, fps_in, fps_out):
    """
    Resample poses (T, 72) and trans (T, 3) from fps_in to fps_out.
    Axis-angle vectors are resampled via SLERP per joint.
    """
    T = poses.shape[0]
    t_in  = np.linspace(0, 1, T)
    T_out = max(1, round(T * fps_out / fps_in))
    t_out = np.linspace(0, 1, T_out)

    # Resample each axis-angle channel via SLERP
    n_joints = poses.shape[1] // 3
    poses_out = np.zeros((T_out, poses.shape[1]), dtype=np.float32)
    for j in range(n_joints):
        aa = poses[:, j*3:(j+1)*3]            # (T, 3)
        rots = Rotation.from_rotvec(aa)
        slerp = Slerp(t_in, rots)
        poses_out[:, j*3:(j+1)*3] = slerp(t_out).as_rotvec().astype(np.float32)

    # Resample translations linearly
    trans_out = interp1d(t_in, trans, axis=0)(t_out).astype(np.float32)

    return poses_out, trans_out


# ---------------------------------------------------------------------------
# Rotation matrix → axis-angle
# ---------------------------------------------------------------------------

def rotmat_to_aa(rotmats):
    """
    Convert rotation matrices to axis-angle vectors.

    Parameters
    ----------
    rotmats : np.ndarray  shape (..., 3, 3)

    Returns
    -------
    np.ndarray  shape (..., 3)
    """
    orig_shape = rotmats.shape[:-2]
    flat = rotmats.reshape(-1, 3, 3)
    aa = Rotation.from_matrix(flat).as_rotvec()   # (N, 3)
    return aa.reshape(*orig_shape, 3)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_fit3d_json(path):
    """Load a fit3d SMPL-X JSON and return a dict with poses/trans/betas/gender."""
    with open(path) as f:
        d = json.load(f)

    global_orient = np.array(d['global_orient'], dtype=np.float32)   # (T, 1, 3, 3)
    body_pose     = np.array(d['body_pose'],     dtype=np.float32)   # (T, 21, 3, 3)
    transl        = np.array(d['transl'],        dtype=np.float32)   # (T, 3)
    betas         = np.array(d['betas'],         dtype=np.float32)   # (T, 10)

    T = global_orient.shape[0]

    # Rotation matrices → axis-angle
    go_aa = rotmat_to_aa(global_orient[:, 0])          # (T, 3)
    bp_aa = rotmat_to_aa(body_pose)                    # (T, 21, 3)
    bp_aa = bp_aa.reshape(T, 63)                       # (T, 63)

    # fit3d stores SMPL-X in a Z-up world frame; aitviewer / AddBiomechanics
    # expect Y-up.  Apply a -90° rotation around X to both the global orient
    # and the translation so the subject stands upright everywhere downstream
    # (marker placement, IK, and visualisation).
    R_fix = Rotation.from_euler('x', -90, degrees=True)
    go_aa = (R_fix * Rotation.from_rotvec(go_aa)).as_rotvec().astype(np.float32)
    transl = (R_fix.as_matrix() @ transl[:, :, np.newaxis]).squeeze(-1).astype(np.float32)

    # Build SMPL poses (T, 72): global(3) + body(63) + hand joints 22&23 zeros(6)
    poses = np.zeros((T, 72), dtype=np.float32)
    poses[:, :3]   = go_aa
    poses[:, 3:66] = bp_aa
    # joints 22 and 23 left at zero (smpl2addbio convention)

    return {
        'poses':  poses,
        'trans':  transl,
        'betas':  betas[0],   # (10,) — constant shape
        'gender': 'neutral',
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input',  required=True,
                        help='Path to input file (fit3d .json)')
    parser.add_argument('--output', required=True,
                        help='Root output directory. A per-trial subdirectory '
                             '<stem>/ is created automatically, so it is safe '
                             'to reuse the same --output for multiple inputs.')
    parser.add_argument('--fps',        type=float, default=50.0,
                        help='Input frame rate of the sequence (default: 50)')
    parser.add_argument('--target_fps', type=float, default=30.0,
                        help='Output frame rate written to the NPZ (default: 30). '
                             'show_ab_results.py requires fps_in %% fps_out == 0, '
                             'and it hardcodes fps_out=30, so keep default at 30.')
    parser.add_argument('--gender', default='neutral',
                        choices=['neutral', 'male', 'female'],
                        help='Subject gender (default: neutral)')
    args = parser.parse_args()

    ext = os.path.splitext(args.input)[1].lower()

    if ext == '.json':
        data = load_fit3d_json(args.input)
    else:
        raise ValueError(f'Unsupported input format: {ext}')

    data['gender'] = args.gender

    T = data['poses'].shape[0]

    # Resample to target_fps if different from source fps
    if abs(args.fps - args.target_fps) > 0.01:
        data['poses'], data['trans'] = resample(
            data['poses'], data['trans'], args.fps, args.target_fps)
        print(f'  resampled {T} frames @ {args.fps}fps → {data["poses"].shape[0]} frames @ {args.target_fps}fps')
        T = data['poses'].shape[0]

    data['mocap_framerate'] = np.float32(args.target_fps)

    print(f'Loaded {T} frames from {args.input}')
    print(f'  gender={data["gender"]}  fps={args.target_fps}')
    print(f'  poses: {data["poses"].shape}  trans: {data["trans"].shape}  betas: {data["betas"].shape}')

    stem = os.path.splitext(os.path.basename(args.input))[0]
    trial_dir = os.path.join(args.output, stem)
    os.makedirs(trial_dir, exist_ok=True)
    out_path = os.path.join(trial_dir, f'{stem}.npz')

    np.savez(out_path, **data)
    print(f'Saved → {out_path}')
    print(f'\nNext step:')
    print(f'  bash run_smpl2bsm.sh {trial_dir} <osim_output_dir>')


if __name__ == '__main__':
    main()
