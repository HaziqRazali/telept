#!/usr/bin/env python3
"""
Visualize two BSM runs (full vs bony markers) simultaneously in the aitviewer GUI.

Usage (run from ~/code/SMPL2AddBiomechanics):
  python /path/to/compare_viz.py <full_dir> <bony_dir> --npz PATH

  full_dir / bony_dir  subject dirs produced by run_smpl2bsm.sh
                       e.g. output/band_pull_apart_full/band_pull_apart
  --npz                path to the original SMPL .npz  (default: auto-detect from /tmp)
  --trial              trial stem (default: auto-detect from IK .mot files)
  --offset             X-axis separation between the two skeletons in metres (default: 0.8)
  --body_model         smpl or smplx (default: smplx)

Keyboard shortcuts in the GUI:
  Space   play/pause
  ← / →   step one frame
  Scene tree panel (left): click the eye icon to toggle any layer

Example:
  cd ~/code/SMPL2AddBiomechanics
  python ~/datasets/telept/my_scripts/opensim_scripts/visualization_scripts/visualize_bony_vs_full.py \
    /home/haziq/code/SMPL2AddBiomechanics/output/barbell_row_full/barbell_row \
    /home/haziq/code/SMPL2AddBiomechanics/output/barbell_row_bony/barbell_row \
    --npz /tmp/smpl2ab_barbell_row/barbell_row/barbell_row.npz \
    --offset 0
"""

import argparse
import glob
import os
import sys

import numpy as np
import yaml

# must be run from (or have) SMPL2AddBiomechanics on the path
AB_ROOT = os.path.expanduser('~/code/SMPL2AddBiomechanics')
for p in (AB_ROOT, os.path.join(AB_ROOT, 'smpl2ab')):
    if p not in sys.path:
        sys.path.insert(0, p)

from aitviewer.renderables.osim import OSIMSequence
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.markers import Markers
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.utils.so3 import resample_rotations
from aitviewer.utils.utils import resample_positions

from smpl2ab.markers.smpl_markers import SmplMarker
from smpl2ab.utils.smpl_utils import load_smpl_seq

FPS = 30

# colours: full = blue skeleton + green markers, bony = orange skeleton + yellow markers
FULL_SKEL_COLOR    = (0.3, 0.55, 1.0, 1.0)
BONY_SKEL_COLOR    = (1.0, 0.45, 0.15, 1.0)
FULL_MARKERS_COLOR = (0.1, 0.9, 0.1, 1.0)
BONY_MARKERS_COLOR = (1.0, 0.85, 0.0, 1.0)


# ── helpers ──────────────────────────────────────────────────────────────────

def find_mot(subject_dir, trial=None):
    pattern = os.path.join(subject_dir, 'osim_results/IK/*.mot')
    mots = sorted(glob.glob(pattern))
    if not mots:
        raise FileNotFoundError(f'No .mot file in {subject_dir}/osim_results/IK/')
    if trial:
        mots = [m for m in mots if os.path.basename(m).startswith(trial)]
        if not mots:
            raise FileNotFoundError(f'No .mot matching trial "{trial}"')
    return mots[0]


def auto_npz(subject_dir):
    """Try to find the SMPL .npz from the default /tmp location."""
    trial_stem = os.path.basename(subject_dir)
    candidate = f'/tmp/smpl2ab_{trial_stem}/{trial_stem}/{trial_stem}.npz'
    if os.path.exists(candidate):
        return candidate
    return None


def load_osim_seq(subject_dir, trial, name, color, x_offset=0.0):
    osim_path = os.path.join(subject_dir,
                             'osim_results/Models/match_markers_but_ignore_physics.osim')
    mot_path  = find_mot(subject_dir, trial)
    seq = OSIMSequence.from_files(
        osim_path=osim_path,
        mot_file=mot_path,
        name=name,
        fps_out=FPS,
        color_skeleton_per_part=False,
        show_joint_angles=False,
        is_rigged=False,
        ignore_geometry=True,
        z_up=False,
    )
    if x_offset != 0.0:
        p = seq.position.copy()
        p[0] += x_offset
        seq.position = p
    # hide the built-in blue IK-fitted markers (we show SMPL virtual markers instead)
    seq.markers_seq.is_visible = False
    return seq


def load_smpl_mesh(npz_path, body_model, x_offset=0.0, name='SMPL mesh'):
    smpl_data = load_smpl_seq(npz_path)
    gender    = smpl_data['gender']
    layer     = SMPLLayer(model_type=body_model, gender=gender)
    poses     = smpl_data['poses']
    fps_in    = float(smpl_data['fps'])

    if fps_in != FPS:
        ps   = resample_rotations(np.reshape(poses, [poses.shape[0], -1, 3]), fps_in, FPS)
        ps   = np.reshape(ps, [-1, poses.shape[1]])
        trans = resample_positions(smpl_data['trans'], fps_in, FPS)
    else:
        ps    = poses
        trans = smpl_data['trans']

    if x_offset != 0.0:
        trans = trans.copy()
        trans[:, 0] += x_offset

    seq = SMPLSequence(
        poses_root=ps[:, :3],
        poses_body=ps[:, 3:],
        smpl_layer=layer,
        betas=smpl_data['betas'][np.newaxis],
        trans=trans,
        name=name,
        show_joint_angles=False,
        z_up=False,
    )
    return seq


def load_markers(smpl_seq, marker_dict_path, name, color):
    markers_dict = yaml.load(open(marker_dict_path), Loader=yaml.FullLoader)
    sm = SmplMarker(smpl_seq.vertices, markers_dict, fps=FPS, name=name)
    return Markers(
        sm.marker_trajectory,
        markers_labels=sm.marker_names,
        name=name,
        color=color,
        z_up=False,
    )


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Side-by-side GUI comparison of full vs bony BSM runs')
    parser.add_argument('full_dir',
                        help='Subject dir from the full-marker run  '
                             '(e.g. output/band_pull_apart_full/band_pull_apart)')
    parser.add_argument('bony_dir',
                        help='Subject dir from the bony-marker run  '
                             '(e.g. output/band_pull_apart_bony/band_pull_apart)')
    parser.add_argument('--npz', default=None,
                        help='Path to the original SMPL .npz (auto-detected from /tmp if omitted)')
    parser.add_argument('--trial', default=None,
                        help='Trial stem, e.g. band_pull_apart (auto-detected if omitted)')
    parser.add_argument('--offset', type=float, default=0.8,
                        help='X-axis gap between the two skeletons in metres (default: 0.8)')
    parser.add_argument('--body_model', default='smpl', choices=['smpl', 'smplx'])
    parser.add_argument('--full_marker_dict',
                        default=os.path.expanduser(
                            '~/code/SMPL2AddBiomechanics/smpl2ab/data/bsm_markers.yaml'))
    parser.add_argument('--bony_marker_dict',
                        default=os.path.expanduser(
                            '~/code/SMPL2AddBiomechanics/smpl2ab/data/bsm_markers_bony.yaml'))
    args = parser.parse_args()

    # ── resolve paths ──
    full_dir = os.path.expanduser(args.full_dir)
    bony_dir = os.path.expanduser(args.bony_dir)

    npz = args.npz
    if npz is None:
        npz = auto_npz(os.path.basename(full_dir))
    if npz is None or not os.path.exists(npz):
        sys.exit(
            'ERROR: could not auto-detect SMPL .npz — pass --npz explicitly.\n'
            f'  Tried: {npz}')

    print(f'Full dir : {full_dir}')
    print(f'Bony dir : {bony_dir}')
    print(f'SMPL npz : {npz}')
    print(f'Offset   : {args.offset} m  (bony skeleton shifted right)')

    to_display = []

    # ── SMPL meshes (one per skeleton so the bony one is shifted too) ──
    smpl_full = load_smpl_mesh(npz, args.body_model, x_offset=0.0,          name='SMPL mesh (full)')
    smpl_bony = load_smpl_mesh(npz, args.body_model, x_offset=args.offset,  name='SMPL mesh (bony)')
    to_display += [smpl_full, smpl_bony]

    # ── OpenSim skeletons ──
    full_seq = load_osim_seq(full_dir, args.trial,
                              f'Full skeleton (105 markers)', FULL_SKEL_COLOR,
                              x_offset=0.0)
    bony_seq = load_osim_seq(bony_dir, args.trial,
                              f'Bony skeleton (57 markers)', BONY_SKEL_COLOR,
                              x_offset=args.offset)
    to_display += [full_seq, bony_seq]

    # ── SMPL virtual markers ──
    full_mkr = load_markers(smpl_full, args.full_marker_dict,
                             'Full markers — green (105)', FULL_MARKERS_COLOR)
    bony_mkr = load_markers(smpl_bony, args.bony_marker_dict,
                             'Bony markers — yellow (57)', BONY_MARKERS_COLOR)
    to_display += [full_mkr, bony_mkr]

    # ── launch viewer ──
    from aitviewer.configuration import CONFIG as C
    C.window_type = 'glfw'

    v = Viewer()
    v.run_animations = True
    # camera sits between the two skeletons
    v.scene.camera.position = np.array([10.0 + args.offset / 2, 2.5, 0.0])
    v.scene.add(*to_display)
    v.lock_to_node(smpl_full, (2, 0.7, 2), smooth_sigma=5.0)
    v.playback_fps = FPS

    # wire aitviewer glfw event handlers
    v.on_render                 = v.render
    v.on_resize                 = v.resize
    v.on_key_event              = v.key_event
    v.on_mouse_position_event   = v.mouse_position_event
    v.on_mouse_press_event      = v.mouse_press_event
    v.on_mouse_release_event    = v.mouse_release_event
    v.on_mouse_drag_event       = v.mouse_drag_event
    v.on_mouse_scroll_event     = v.mouse_scroll_event
    v.on_unicode_char_entered   = v.unicode_char_entered
    v.on_files_dropped_event    = v.files_dropped_event
    v.window.config = v
    v.run()


if __name__ == '__main__':
    main()
