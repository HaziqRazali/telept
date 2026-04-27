#!/usr/bin/env python3
"""
_gen_fixture_lod3_worker.py
===========================
Worker script for gen_mhr_fixture.py.

Run inside the mhr_new conda env to execute the pymomentum LOD3 FK forward pass
using the same MHR.from_files(lod=3) pipeline as export_mhr_assets.py.

Input:  --params  float32[204] binary file  (model_params from SAM3DBody)
Output: --out     float32[4899*3] binary file (skinned verts in cm, Y-UP)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

MHR_REPO   = Path("/home/haziq/MHR")
MHR_ASSETS = MHR_REPO / "assets"
LOD        = 3

sys.path.insert(0, str(MHR_REPO))
from mhr.mhr import MHR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True, help="float32[204] binary file")
    parser.add_argument("--out",    required=True, help="output float32[N*3] binary file")
    args = parser.parse_args()

    # Load LOD3 model (CPU — no CUDA needed for a fixture)
    device = torch.device("cpu")
    print(f"[lod3_worker] Loading MHR LOD{LOD}...")
    m  = MHR.from_files(MHR_ASSETS, device, lod=LOD, wants_pose_correctives=False)
    c  = m.character
    ct = m.character_torch

    # Read model_params[204] produced by SAM3DBody
    model_params_204 = np.fromfile(args.params, dtype=np.float32)  # (204,)
    assert model_params_204.shape == (204,), f"Expected 204 params, got {model_params_204.shape}"

    # Pad to 321 dims (pymomentum param_transform expects full 321-dim vector)
    mp_321 = np.zeros(321, dtype=np.float32)
    mp_321[:204] = model_params_204
    mp_t = torch.from_numpy(mp_321).unsqueeze(0)  # (1, 321)

    # LOD3 FK: model_params → joint_params → skeleton_state → skinned verts
    rest_verts_np = np.array(c.mesh.vertices, dtype=np.float32)   # (4899, 3)
    rest_verts_t  = torch.from_numpy(rest_verts_np).unsqueeze(0)  # (1, 4899, 3)

    with torch.no_grad():
        jp    = ct.model_parameters_to_joint_parameters(mp_t)    # (1, 889)
        ss    = ct.joint_parameters_to_skeleton_state(jp)        # (1, 127, 8)
        verts = ct.linear_blend_skinning(ss, rest_verts_t)       # (1, 4899, 3)

    verts_cm = verts[0].numpy().astype(np.float32)  # (4899, 3), cm, Y-UP
    n_verts  = verts_cm.shape[0]

    verts_cm.flatten().tofile(args.out)
    print(f"[lod3_worker] n_verts={n_verts}  Y range [{verts_cm[:,1].min():.3f}, {verts_cm[:,1].max():.3f}] cm")
    print(f"[lod3_worker] Wrote {Path(args.out).stat().st_size} bytes → {args.out}")


if __name__ == "__main__":
    main()
