"""
Export MHR static assets for LOD3 to the Flutter app's assets folder.

Run once with:
    conda run -n mhr_new python3 app/server/export_mhr_assets.py

Outputs to app/mobile/assets/mhr/:
  param_transform.bin   float32 (889, 321)   - model_params → joint_params linear map
  skeleton.bin          custom packed         - joint parents, offsets, pre_rotations
  inv_bind_pose.bin     float32 (127, 4, 4)  - world-to-joint in T-pose
  skin_idx.bin          int16  (N, 8)         - sparse skin weight joint indices
  skin_wgt.bin          float32 (N, 8)        - sparse skin weight values
  rest_verts.bin        float32 (N, 3)        - T-pose vertex positions (cm)
  faces.bin             int32  (F, 3)         - triangle face indices
"""

import sys
import struct
import numpy as np
import torch
from pathlib import Path

LOD = 3
MHR_ASSETS = Path("/home/haziq/MHR/assets")
OUT_DIR = Path(__file__).resolve().parent.parent / "mobile" / "assets" / "mhr"

# ---------------------------------------------------------------------------
# Bootstrap: MHR repo path
# ---------------------------------------------------------------------------
MHR_REPO = Path("/home/haziq/MHR")
sys.path.insert(0, str(MHR_REPO))
from mhr.mhr import MHR

print(f"Loading MHR LOD{LOD} …")
m = MHR.from_files(folder=MHR_ASSETS, device=torch.device("cpu"), lod=LOD,
                   wants_pose_correctives=False)
c  = m.character
ct = m.character_torch

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Parameter transform  (889, 321)  float32
# ---------------------------------------------------------------------------
T = c.parameter_transform.transform.astype(np.float32)   # (889, 321)
assert T.shape == (889, 321), f"unexpected shape {T.shape}"
T.tofile(OUT_DIR / "param_transform.bin")
print(f"  param_transform.bin  {T.shape}  {T.nbytes/1e6:.2f} MB")

# ---------------------------------------------------------------------------
# 2. Skeleton: parents (127,) int16 | offsets (127,3) float32 | pre_rots (127,4) float32
#    Binary layout:
#      [0]    int32  n_joints
#      [4]    int16[n_joints]  parents   (-1 = root)
#      padded to 4-byte boundary
#      float32[n_joints, 3]  offsets
#      float32[n_joints, 4]  pre_rotations  (x,y,z,w)
# ---------------------------------------------------------------------------
sk = c.skeleton
n_joints = sk.size
parents   = np.array(sk.joint_parents, dtype=np.int16)     # (127,)
offsets   = np.array(sk.offsets,       dtype=np.float32)   # (127, 3)
pre_rots  = np.array(sk.pre_rotations, dtype=np.float32)   # (127, 4)  xyzw

skel_path = OUT_DIR / "skeleton.bin"
with open(skel_path, "wb") as f:
    f.write(struct.pack("<i", n_joints))           # 4 bytes
    f.write(parents.tobytes())                     # 127 * 2 = 254 bytes
    # pad to 4-byte boundary
    pad = (4 - (254 % 4)) % 4
    f.write(b"\x00" * pad)
    f.write(offsets.tobytes())                     # 127 * 3 * 4 = 1524 bytes
    f.write(pre_rots.tobytes())                    # 127 * 4 * 4 = 2032 bytes
print(f"  skeleton.bin         n_joints={n_joints}  {skel_path.stat().st_size/1e3:.1f} KB")

# ---------------------------------------------------------------------------
# 3. Inverse bind pose  (127, 4, 4)  float32
# ---------------------------------------------------------------------------
ibp = c.inverse_bind_pose.astype(np.float32)   # (127, 4, 4)
ibp.tofile(OUT_DIR / "inv_bind_pose.bin")
print(f"  inv_bind_pose.bin    {ibp.shape}  {ibp.nbytes/1e3:.1f} KB")

# ---------------------------------------------------------------------------
# 4 & 5. Skin weights  indices: int16  weights: float32  (N, 8)
# ---------------------------------------------------------------------------
sw  = c.skin_weights
idx = sw.index.astype(np.int16)    # (N, 8)
wgt = sw.weight.astype(np.float32) # (N, 8)
n_verts = sw.num_vertices
assert idx.shape == (n_verts, 8)

idx.tofile(OUT_DIR / "skin_idx.bin")
wgt.tofile(OUT_DIR / "skin_wgt.bin")
print(f"  skin_idx.bin         {idx.shape}  {idx.nbytes/1e3:.1f} KB")
print(f"  skin_wgt.bin         {wgt.shape}  {wgt.nbytes/1e3:.1f} KB")

# ---------------------------------------------------------------------------
# 6. Rest vertices  (N, 3)  float32  — T-pose, centred, in centimetres
# ---------------------------------------------------------------------------
verts = np.array(c.mesh.vertices, dtype=np.float32)   # (N, 3)
verts.tofile(OUT_DIR / "rest_verts.bin")
print(f"  rest_verts.bin       {verts.shape}  {verts.nbytes/1e3:.1f} KB")

# ---------------------------------------------------------------------------
# 7. Faces  (F, 3)  int32
# ---------------------------------------------------------------------------
faces = np.array(c.mesh.faces, dtype=np.int32)   # (F, 3)
faces.tofile(OUT_DIR / "faces.bin")
print(f"  faces.bin            {faces.shape}  {faces.nbytes/1e3:.1f} KB")

# ---------------------------------------------------------------------------
# Manifest — tiny text file so Dart can sanity-check on load
# ---------------------------------------------------------------------------
manifest = (
    f"lod={LOD}\n"
    f"n_joints={n_joints}\n"
    f"n_verts={n_verts}\n"
    f"n_faces={len(faces)}\n"
    f"param_transform_rows=889\n"
    f"param_transform_cols=321\n"
    f"skin_influences=8\n"
)
(OUT_DIR / "manifest.txt").write_text(manifest)
print(f"  manifest.txt         written")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
total = sum(p.stat().st_size for p in OUT_DIR.iterdir())
print(f"\nDone.  Total: {total/1e6:.2f} MB  →  {OUT_DIR}")
print(f"\nJoint names (first 10):")
for i, name in enumerate(sk.joint_names[:10]):
    print(f"  {i:3d}  {name}")
