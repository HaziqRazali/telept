#!/usr/bin/env python3
"""
build_mhr_jregressor.py

Converts the SMPL J_regressor (24, 6890) into a MHR J_regressor (24, 18439).
Output: ~/MHR/tools/mhr_smpl_conversion/assets/J_regressor_mhr.npy

At runtime (no SMPL needed):
    J_reg_mhr = np.load('J_regressor_mhr.npy')   # (24, 18439) float32
    joints = J_reg_mhr @ mhr_verts                # (24, 3)
    # joint 1 = L_Hip, joint 2 = R_Hip, etc.

Math:
    mhr_verts = A @ smpl_verts   (A is (18439, 6890) sparse barycentric matrix)
    joints = J_reg @ smpl_verts
    We want J_mhr such that J_mhr @ A = J_reg
    → for each row j: solve  A^T x = J_reg[j]  (underdetermined, min-norm via lsqr)
    → J_mhr[j] = x_min_norm
"""
import os, pickle
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr

SMPL_PATH   = "/home/haziq/code/SMPL2AddBiomechanics/models/smpl/SMPL_NEUTRAL.pkl"
MAPPING_NPZ = os.path.expanduser("~/MHR/tools/mhr_smpl_conversion/assets/smpl2mhr_mapping.npz")
OUT_NPY     = os.path.expanduser("~/MHR/tools/mhr_smpl_conversion/assets/J_regressor_mhr.npy")

print("Loading SMPL ...")
with open(SMPL_PATH, "rb") as f:
    m = pickle.load(f, encoding="latin1")
smpl_verts = np.asarray(m["v_template"], dtype=np.float64)       # (6890, 3)
smpl_faces = np.asarray(m["f"],          dtype=np.int32)
J_reg_raw  = m["J_regressor"]
J_reg = np.asarray(J_reg_raw.todense() if hasattr(J_reg_raw, "todense") else J_reg_raw,
                   dtype=np.float64)                              # (24, 6890)

print("Loading MHR barycentric mapping ...")
mapping = np.load(MAPPING_NPZ)
tri_ids = mapping["triangle_ids"]   # (18439,)
bary    = mapping["baryc_coords"]   # (18439, 3)
n_mhr, n_smpl = 18439, 6890

# Build sparse A (18439, 6890)
tris = smpl_faces[tri_ids]           # (18439, 3) SMPL vertex indices per MHR vert
rows = np.repeat(np.arange(n_mhr), 3)
cols = tris.ravel()
vals = bary.ravel()
A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_mhr, n_smpl))

# Sanity check
v0,v1,v2 = smpl_verts[tris[:,0]], smpl_verts[tris[:,1]], smpl_verts[tris[:,2]]
mhr_ref = bary[:,0:1]*v0 + bary[:,1:2]*v1 + bary[:,2:3]*v2
err = float(np.abs((A @ smpl_verts) - mhr_ref).max())
print(f"Barycentric matrix check: max_err={err:.2e}  (should be ~0)")

# Solve for J_reg_mhr: A^T x = J_reg[j]  (min-norm, 24 joints)
print("Solving 24 least-squares systems (this takes ~30s) ...")
J_reg_mhr = np.zeros((24, n_mhr), dtype=np.float64)
AT = A.T.tocsr()
for j in range(24):
    res = lsqr(AT, J_reg[j], atol=1e-10, btol=1e-10, iter_lim=100000)
    J_reg_mhr[j] = res[0]
    if (j+1) % 6 == 0:
        print(f"  {j+1}/24 done")

# Verify accuracy
smpl_joints = J_reg @ smpl_verts      # (24, 3) ground truth
mhr_joints  = J_reg_mhr @ mhr_ref    # (24, 3) from MHR verts
diff_mm = np.abs(smpl_joints - mhr_joints) * 1000
print(f"Joint centre errors vs SMPL J_reg:  max={diff_mm.max():.2f} mm  mean={diff_mm.mean():.2f} mm")

np.save(OUT_NPY, J_reg_mhr.astype(np.float32))
print(f"Saved → {OUT_NPY}   shape={J_reg_mhr.shape}   dtype=float32")
