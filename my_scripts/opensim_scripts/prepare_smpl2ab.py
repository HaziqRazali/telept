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
  python prepare_smpl2ab.py \\
      --input  /home/haziq/datasets/mocap/data/fit3d/train/s03/smplx/band_pull_apart.json \\
      --output /tmp/smpl2ab_input/band_pull_apart/ \\
      --fps    50 \\
      --gender neutral
"""

import argparse
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation


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
                        help='Output directory. The .npz will be saved inside '
                             'with the same stem as the input file.')
    parser.add_argument('--fps',    type=float, default=50.0,
                        help='Frame rate of the sequence (default: 50)')
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
    data['mocap_framerate'] = np.float32(args.fps)

    T = data['poses'].shape[0]
    print(f'Loaded {T} frames from {args.input}')
    print(f'  gender={data["gender"]}  fps={args.fps}')
    print(f'  poses: {data["poses"].shape}  trans: {data["trans"].shape}  betas: {data["betas"].shape}')

    os.makedirs(args.output, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.input))[0]
    out_path = os.path.join(args.output, f'{stem}.npz')

    np.savez(out_path, **data)
    print(f'Saved → {out_path}')
    print(f'\nNext step:')
    print(f'  bash run_smpl2bsm.sh {args.output} <output_dir>')


if __name__ == '__main__':
    main()
