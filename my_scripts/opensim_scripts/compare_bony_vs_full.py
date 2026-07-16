"""
Compare AddBiomechanics IK results between full and bony marker sets.

Usage:
    python compare_bony_vs_full.py <full_subject_dir> <bony_subject_dir> [trial_name]

    full_subject_dir   e.g. ~/code/SMPL2AddBiomechanics/output/band_pull_apart_full/band_pull_apart
    bony_subject_dir   e.g. ~/code/SMPL2AddBiomechanics/output/band_pull_apart_bony/band_pull_apart
    trial_name         e.g. band_pull_apart  (auto-detected if omitted)

Outputs:
    1. Marker RMSE — per-marker and overall, for both runs.
    2. Joint angle RMSE — per-DOF difference between the two .mot files.
    3. Body scale factors — per-segment % difference in the scaled osim models.
"""

import os
import sys
import glob
import re
import numpy as np
import pandas as pd


def find_trial(subject_dir):
    csvs = glob.glob(os.path.join(subject_dir, 'osim_results/IK/*_marker_errors.csv'))
    if not csvs:
        raise FileNotFoundError(f'No marker_errors.csv under {subject_dir}')
    return re.sub(r'_segment_\d+_marker_errors\.csv$', '',
                  os.path.basename(sorted(csvs)[0]))


def load_marker_errors(subject_dir, trial):
    path = os.path.join(subject_dir, f'osim_results/IK/{trial}_segment_0_marker_errors.csv')
    df = pd.read_csv(path)
    marker_cols = [c for c in df.columns if c != 'Timestep']
    summary = df[df['Timestep'] == 'All Timesteps RMSE'][marker_cols].astype(float).iloc[0]
    return summary, marker_cols


def load_mot(subject_dir, trial):
    path = os.path.join(subject_dir, f'osim_results/IK/{trial}_segment_0_ik.mot')
    with open(path) as f:
        lines = f.readlines()
    header_end = next(i for i, l in enumerate(lines) if l.strip() == 'endheader')
    return pd.read_csv(path, sep='\t', skiprows=header_end + 1)


def load_scale_factors(subject_dir):
    path = os.path.join(subject_dir,
                        'osim_results/Models/match_markers_but_ignore_physics.osim')
    if not os.path.exists(path):
        return None
    content = open(path).read()
    # Scale factors live inside <Mesh> blocks (not <FrameGeometry> which is always 0.2)
    # Pattern: find each Body, then within it find the first <Mesh> scale_factors
    body_pattern = r'<Body name="([^"]+)">(.*?)</Body>'
    mesh_pattern = r'<Mesh\b[^>]*>(.*?)</Mesh>'
    sf_pattern   = r'<scale_factors>([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)</scale_factors>'
    rows = []
    for body_name, body_content in re.findall(body_pattern, content, re.DOTALL):
        for mesh_content in re.findall(mesh_pattern, body_content, re.DOTALL):
            sf = re.search(sf_pattern, mesh_content)
            if sf:
                x, y, z = sf.groups()
                rows.append({'body': body_name,
                             'sx': float(x), 'sy': float(y), 'sz': float(z)})
                break  # first Mesh per Body is sufficient
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    full_dir = os.path.expanduser(sys.argv[1])
    bony_dir = os.path.expanduser(sys.argv[2])
    trial    = sys.argv[3] if len(sys.argv) > 3 else find_trial(full_dir)

    print(f'\nTrial: {trial}')
    print(f'Full:  {full_dir}')
    print(f'Bony:  {bony_dir}')

    # ── 1. Marker RMSE ───────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('1. MARKER RMSE  (metres)')
    print('='*60)

    full_rmse, full_markers = load_marker_errors(full_dir, trial)
    bony_rmse, bony_markers = load_marker_errors(bony_dir, trial)
    common = [m for m in bony_markers if m in full_markers]

    print(f'\n{"Marker":<12} {"Full":>10} {"Bony":>10} {"Δ (mm)":>10}')
    print('-' * 44)
    diffs = []
    for m in sorted(common):
        d = (bony_rmse[m] - full_rmse[m]) * 1000
        diffs.append(d)
        print(f'{m:<12} {full_rmse[m]:>10.4f} {bony_rmse[m]:>10.4f} {d:>+10.2f}')
    print('-' * 44)
    print(f'{"Mean":<12} {full_rmse[common].mean():>10.4f} '
          f'{bony_rmse[common].mean():>10.4f} {np.mean(diffs):>+10.2f}')
    print(f'\nFull run — all {len(full_markers)} markers:   '
          f'{full_rmse.mean()*1000:.2f} mm')
    print(f'Bony run — {len(bony_markers)} bony markers: '
          f'{bony_rmse.mean()*1000:.2f} mm')

    # ── 2. Joint angle differences ───────────────────────────────────────────
    print('\n' + '='*60)
    print('2. JOINT ANGLE DIFFERENCE  (radians, full − bony)')
    print('='*60)

    full_mot = load_mot(full_dir, trial)
    bony_mot = load_mot(bony_dir, trial)
    dof_cols = [c for c in full_mot.columns if c != 'time']
    n = min(len(full_mot), len(bony_mot))
    full_mot = full_mot.iloc[:n]
    bony_mot = bony_mot.iloc[:n]

    print(f'\n{"DOF":<30} {"RMSE (rad)":>12} {"Max |Δ| (rad)":>14}')
    print('-' * 58)
    rows, all_sq = [], []
    for dof in dof_cols:
        diff = full_mot[dof].values - bony_mot[dof].values
        rmse = np.sqrt(np.mean(diff**2))
        maxd = np.max(np.abs(diff))
        rows.append((dof, rmse, maxd))
        all_sq.append(diff**2)
    for dof, rmse, maxd in sorted(rows, key=lambda r: r[1], reverse=True):
        flag = '  ← possible sign/gimbal flip' if rmse > 1.5 else ''
        print(f'{dof:<30} {rmse:>12.5f} {maxd:>14.5f}{flag}')
    print('-' * 58)
    print(f'{"Overall":<30} {np.sqrt(np.mean(np.concatenate(all_sq))):>12.5f}')

    # ── 3. Body scale factors ────────────────────────────────────────────────
    print('\n' + '='*60)
    print('3. BODY SCALE FACTORS  (mean x/y/z, % diff bony vs full)')
    print('='*60)

    fs = load_scale_factors(full_dir)
    bs = load_scale_factors(bony_dir)
    if fs is None or bs is None:
        print('  scaled osim not found — skipping')
    else:
        m = fs.merge(bs, on='body', suffixes=('_f', '_b'))
        m['mean_f']   = (m['sx_f'] + m['sy_f'] + m['sz_f']) / 3
        m['mean_b']   = (m['sx_b'] + m['sy_b'] + m['sz_b']) / 3
        m['pct_diff'] = (m['mean_b'] - m['mean_f']) / m['mean_f'] * 100

        print(f'\n{"Segment":<30} {"Full":>10} {"Bony":>10} {"% diff":>10}')
        print('-' * 62)
        for _, row in m.sort_values('pct_diff', key=abs, ascending=False).iterrows():
            print(f'{row["body"]:<30} {row["mean_f"]:>10.4f} '
                  f'{row["mean_b"]:>10.4f} {row["pct_diff"]:>+10.2f}')
        print('-' * 62)
        print(f'{"Mean abs % diff":<52} {m["pct_diff"].abs().mean():>+10.2f}')

    print()


if __name__ == '__main__':
    main()
