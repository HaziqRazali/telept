"""
Inspect a .c3d file and print all keys, shapes, and dtypes.
Usage:
    python inspect_c3d.py
"""

import numpy as np
import ezc3d

C3D_PATH = "/data/telept/my_scripts/orbbec_scripts/Take 2014-10-02 05.49.50 AM.c3d"

c = ezc3d.c3d(C3D_PATH)

# ── 1. Point data (3D marker positions) ─────────────────────────────────────
print("=" * 60)
print("POINT DATA  (markers)")
print("=" * 60)
points = c["data"]["points"]  # shape: (4, n_markers, n_frames)  [X,Y,Z,W]
print(f"  data['points']  shape={points.shape}  dtype={points.dtype}")
print(f"  axes: (XYZW=4, n_markers={points.shape[1]}, n_frames={points.shape[2]})")

marker_labels = c["parameters"]["POINT"]["LABELS"]["value"]
print(f"\n  Marker labels ({len(marker_labels)}):")
for i, lbl in enumerate(marker_labels):
    print(f"    [{i:3d}] {lbl}")

# ── 2. Analog data (force plates, EMG, etc.) ─────────────────────────────────
print("\n" + "=" * 60)
print("ANALOG DATA")
print("=" * 60)
analog = c["data"]["analogs"]  # shape: (1, n_channels, n_analog_frames)
print(f"  data['analogs']  shape={analog.shape}  dtype={analog.dtype}")

try:
    analog_labels = c["parameters"]["ANALOG"]["LABELS"]["value"]
    print(f"\n  Analog labels ({len(analog_labels)}):")
    for i, lbl in enumerate(analog_labels):
        print(f"    [{i:3d}] {lbl}")
except KeyError:
    print("  (no analog labels found)")

# ── 3. All parameter groups ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PARAMETERS")
print("=" * 60)
for group_name, group in c["parameters"].items():
    print(f"\n  [{group_name}]")
    for param_name, param in group.items():
        val = param.get("value", None)
        if isinstance(val, np.ndarray):
            print(f"    {param_name:30s}  shape={val.shape}  dtype={val.dtype}")
        elif isinstance(val, list):
            print(f"    {param_name:30s}  list  len={len(val)}  sample={val[:3] if len(val) > 0 else '[]'}")
        else:
            print(f"    {param_name:30s}  value={val}")

# ── 4. Header summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("HEADER")
print("=" * 60)
header = c["header"]
for k, v in header.items():
    if isinstance(v, dict):
        for kk, vv in v.items():
            print(f"  {k}.{kk:30s}  {vv}")
    else:
        print(f"  {k:35s}  {v}")
