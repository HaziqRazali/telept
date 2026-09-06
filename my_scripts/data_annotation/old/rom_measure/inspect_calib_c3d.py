"""Quick inspect of mocap_rgb_calib_sync01.c3d: markers, labels, frames."""
import numpy as np
import ezc3d

path = "/data/haziq/telept/mocap_rgb_calib_sync01.c3d"
c = ezc3d.c3d(path)

pts = c["data"]["points"]          # [4, M, F] X,Y,Z,residual
xyz = pts[:3]                      # [3, M, F]
res = pts[3]                       # [M, F]
frame_rate = float(c["parameters"]["POINT"]["RATE"]["value"][0])
units = c["parameters"]["POINT"]["UNITS"]["value"][0]

label_keys = sorted(k for k in c["parameters"]["POINT"] if k.startswith("LABELS"))
labels = []
for k in label_keys:
    labels.extend(c["parameters"]["POINT"][k]["value"])
labels = labels[: xyz.shape[1]]

M, F = xyz.shape[1], xyz.shape[2]
print(f"File          : {path}")
print(f"Markers (M)   : {M}")
print(f"Frames (F)    : {F}")
print(f"Rate          : {frame_rate} fps  ({F/frame_rate:.2f} s)")
print(f"Units         : {units}")
print(f"Point sub-fields present: {list(c['parameters']['POINT'].keys())}")
try:
    print(f"Analog rate   : {c['parameters']['ANALOG']['RATE']['value'][0]} Hz")
    print(f"Analog frames : {c['data']['analogs'].shape}")
except Exception as e:
    print("Analog: none")

print("\n-- Marker labels --")
for i, l in enumerate(labels):
    print(f"   #{i:2d}: {l}")

# Tracking quality per marker
valid = (res >= 0) & np.isfinite(xyz).all(axis=0)
print("\n-- Per-marker tracked fraction --")
for i, l in enumerate(labels):
    frac = valid[i].mean() if i < M else float("nan")
    print(f"   #{i:2d}: {l:<16} {frac*100:5.1f}%")

# Bounding box of all valid points (overall volume)
v = valid
if v.any():
    xs, ys, zs = xyz[0][v], xyz[1][v], xyz[2][v]
    print("\n-- Overall 3D bbox of valid points ({} units) --".format(units))
    print(f"   X: [{xs.min():.2f}, {xs.max():.2f}]")
    print(f"   Y: [{ys.min():.2f}, {ys.max():.2f}]")
    print(f"   Z: [{zs.min():.2f}, {zs.max():.2f}]")
