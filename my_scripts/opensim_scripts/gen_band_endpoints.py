#!/usr/bin/env python3
"""
Generate band_endpoints.npz from a SMPL .npz via forward kinematics.

This is a one-off helper for when you don't yet have measured endpoints.
In production, supply band_endpoints.npz from actual measurements or tracking.

Output keys
-----------
anchor_pos  (3,)    fixed end of band, Y-up world frame (m)
limb_pos    (T, 3)  moving end attached to the limb, Y-up world frame (m)
time        (T,)    seconds

Usage (from ~/code/SMPL2AddBiomechanics):
  conda activate addbiomechanics
  python ~/datasets/telept/my_scripts/opensim_scripts/gen_band_endpoints.py \
      --npz    /tmp/smpl2ab_dumbbell_biceps_curls/dumbbell_biceps_curls/dumbbell_biceps_curls.npz \
      --output output/dumbbell_biceps_curls/band_endpoints.npz \
      --arm    left
"""

import argparse
import os
import sys
import numpy as np

# reuse FK helpers from resistance_band_id.py (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resistance_band_id import _smpl_joint_positions, _SMPL_IDX


def gen_endpoints(npz_path, output_path, arm='left', fps=30, anchor_xyz=None):
    arm_prefix = arm[0]  # 'l' or 'r'

    print(f"Running SMPL FK on {npz_path} ...")
    joints, time = _smpl_joint_positions(npz_path, fps_out=fps)
    # joints: (F, 24, 3)  Y-up

    limb_pos = joints[:, _SMPL_IDX[f'{arm_prefix}_wrist'], :]  # (F, 3)

    if anchor_xyz is not None:
        anchor_pos = np.array(anchor_xyz, dtype=np.float64)
    else:
        # Auto-derive: at floor level, under the wrist at t=0
        foot_y = float(joints[:, [_SMPL_IDX['l_foot'], _SMPL_IDX['r_foot']], 1].min())
        anchor_pos = np.array([limb_pos[0, 0], foot_y, limb_pos[0, 2]])

    print(f"  anchor_pos = {anchor_pos}")
    print(f"  limb_pos   : {limb_pos.shape}  y range [{limb_pos[:,1].min():.3f}, {limb_pos[:,1].max():.3f}] m")
    print(f"  time       : {time[0]:.3f} - {time[-1]:.3f} s  ({len(time)} frames @ {fps} fps)")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or '.', exist_ok=True)
    np.savez(output_path, anchor_pos=anchor_pos, limb_pos=limb_pos, time=time)
    print(f"\n✓ Saved -> {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate band_endpoints.npz from SMPL FK (one-off helper)')
    parser.add_argument('--npz',    required=True, help='SMPL .npz (source motion)')
    parser.add_argument('--output', required=True, help='Output band_endpoints.npz path')
    parser.add_argument('--arm',    default='left', choices=['left', 'right'],
                        help='Which arm/leg attaches to the band (default: left)')
    parser.add_argument('--fps',    type=int, default=30, help='Resample fps (default: 30)')
    parser.add_argument('--anchor', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                        default=None,
                        help='Override anchor in Y-up metres. If omitted, auto-derived from floor.')
    args = parser.parse_args()

    gen_endpoints(args.npz, args.output, arm=args.arm, fps=args.fps, anchor_xyz=args.anchor)
