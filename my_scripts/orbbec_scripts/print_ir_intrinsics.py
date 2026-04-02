"""
Print IR (depth sensor) camera intrinsics and distortion coefficients
for the Orbbec Femto Bolt.

Usage:
    conda activate orbbec
    python print_ir_intrinsics.py

The camera must be plugged in.  The script starts the pipeline briefly to
read calibration parameters then exits.
"""

import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat


def main():
    pipeline = Pipeline()
    config   = Config()

    # Enable depth/IR stream so calibration params are populated.
    # IR and depth share the same sensor on the Femto Bolt.
    try:
        depth_list    = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
    except Exception as e:
        print(f"Warning: could not enable depth stream: {e}")

    pipeline.start(config)

    try:
        param = pipeline.get_camera_param()
    finally:
        pipeline.stop()

    ir = param.depth_intrinsic   # OBCameraIntrinsic for the depth/IR sensor

    print("=" * 60)
    print("Orbbec Femto Bolt — IR / Depth sensor intrinsics")
    print("=" * 60)
    print(f"  Resolution : {ir.width} x {ir.height}")
    print(f"  fx = {ir.fx}")
    print(f"  fy = {ir.fy}")
    print(f"  cx = {ir.cx}")
    print(f"  cy = {ir.cy}")

    # Distortion (depth sensor)
    dc = param.depth_distortion   # OBCameraDistortion
    print("\n  Distortion coefficients (k1 k2 p1 p2 k3):")
    print(f"  k1={dc.k1}  k2={dc.k2}  k3={dc.k3}")
    print(f"  p1={dc.p1}  p2={dc.p2}")

    # Print as copy-pasteable numpy arrays
    print("\n" + "=" * 60)
    print("Copy-paste into calibrate_pnp.py:")
    print("=" * 60)
    print(f"""
K = np.array([
    [{ir.fx}, 0.0,    {ir.cx}],
    [0.0,    {ir.fy}, {ir.cy}],
    [0.0,    0.0,    1.0     ],
], dtype=np.float64)

DIST = np.array([
    {dc.k1},
    {dc.k2},
    {dc.p1},
    {dc.p2},
    {dc.k3},
], dtype=np.float64)  # k1, k2, p1, p2, k3 (OpenCV order)
""")


if __name__ == "__main__":
    main()
