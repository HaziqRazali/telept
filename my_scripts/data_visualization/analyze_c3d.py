"""
Print every C3D frame's non-NaN markers.
Usage:
    python analyze_c3d.py [c3d_file]
"""

import sys
import numpy as np
import ezc3d

C3D_PATH = "/home/haziq/datasets/telept/data/mocap_recordings/LED1.c3d"

path = sys.argv[1] if len(sys.argv) > 1 else C3D_PATH

c   = ezc3d.c3d(path)
fps = float(c["parameters"]["POINT"]["RATE"]["value"][0])
xyz = c["data"]["points"][:3]   # [3, M, F]

label_keys = sorted(k for k in c["parameters"]["POINT"] if k.startswith("LABELS"))
labels = []
for k in label_keys:
    labels.extend(c["parameters"]["POINT"][k]["value"])
labels = labels[:xyz.shape[1]]

M, F = xyz.shape[1], xyz.shape[2]
print(f"File   : {path}")
print(f"FPS    : {fps}  |  Markers: {M}  |  Frames: {F}")
print("-" * 72)

for f in range(F):
    t_ms = f / fps * 1000
    # markers with all 3 coords finite at this frame
    visible = [labels[m] for m in range(M) if np.all(np.isfinite(xyz[:, m, f]))]
    if visible:
        print(f"Frame {f:5d}  t={t_ms:9.2f} ms  |  {', '.join(visible)}")
    # uncomment the line below to also print frames where nothing is visible:
    # else:
    #     print(f"Frame {f:5d}  t={t_ms:9.2f} ms  |  (none)")
