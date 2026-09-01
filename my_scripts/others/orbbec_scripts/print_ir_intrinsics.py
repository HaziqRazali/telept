"""
Print IR and RGB camera intrinsics, distortion coefficients, and
IR→RGB extrinsics for the Orbbec Femto Bolt.

Usage:
    conda activate orbbec
    python print_ir_intrinsics.py

The camera must be plugged in.  The script starts the pipeline briefly to
read calibration parameters then exits.

(orbbec) haziq@IHP-SPL-236F7SB:/data/telept/my_scripts/orbbec_scripts$ python print_ir_intrinsics.py 
load extensions from /home/haziq/anaconda3/envs/orbbec/lib/python3.10/site-packages/extensions
============================================================
Orbbec Femto Bolt — IR / Depth sensor intrinsics
============================================================
  Resolution : 640 x 576
  fx = 504.14373779296875
  fy = 504.02496337890625
  cx = 329.8353271484375
  cy = 334.18988037109375

  Distortion (k1 k2 p1 p2 k3):
  k1=17.021440505981445  k2=9.416301727294922  k3=0.3900681436061859
  p1=7.297786214621738e-05  p2=3.846113759209402e-05

Copy-paste (IR):
K_IR = np.array([
    [504.14373779296875, 0.0, 329.8353271484375],
    [0.0, 504.02496337890625, 334.18988037109375],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
DIST_IR = np.array([17.021440505981445, 9.416301727294922, 7.297786214621738e-05, 3.846113759209402e-05, 0.3900681436061859], dtype=np.float64)  # k1,k2,p1,p2,k3

============================================================
Orbbec Femto Bolt — RGB sensor intrinsics
============================================================
  Resolution : 1920 x 1080
  fx = 1123.86669921875
  fy = 1123.028076171875
  cx = 948.0269165039062
  cy = 539.6485595703125

  Distortion (k1 k2 p1 p2 k3):
  k1=0.07333821058273315  k2=-0.10178927332162857  k3=0.041689008474349976
  p1=-0.0004722462617792189  p2=-0.00022512981377076358

Copy-paste (RGB):
K_RGB = np.array([
    [1123.86669921875, 0.0, 948.0269165039062],
    [0.0, 1123.028076171875, 539.6485595703125],
    [0.0, 0.0, 1.0],
], dtype=np.float64)
DIST_RGB = np.array([0.07333821058273315, -0.10178927332162857, -0.0004722462617792189, -0.00022512981377076358, 0.041689008474349976], dtype=np.float64)  # k1,k2,p1,p2,k3

============================================================
IR -> RGB extrinsics  (rotation row-major 3x3, translation mm)
============================================================
  R = [
    0.99382645,  -0.00147046,  0.00159027,
    0.00128496,  0.99382663,  0.11093678,
    -0.00174358,  -0.11093448,  0.99382615,
  ]
  t = [-32.834217,  -1.315084,  1.306755]  (mm)

Copy-paste (extrinsics):
R_ir_to_rgb = np.array([
    [0.9938264489173889, -0.0014704565983265638, 0.0015902734594419599],
    [0.0012849620543420315, 0.9938266277313232, 0.11093678325414658],
    [-0.0017435838235542178, -0.11093448102474213, 0.993826150894165],
], dtype=np.float64)
t_ir_to_rgb = np.array([-32.8342170715332, -1.315084457397461, 1.3067550659179688], dtype=np.float64)  # mm
(orbbec) haziq@IHP-SPL-236F7SB:/data/telept/my_scripts/orbbec_scripts$ 
"""

import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat


def print_intrinsic(label, intr, dist):
    print(f"  Resolution : {intr.width} x {intr.height}")
    print(f"  fx = {intr.fx}")
    print(f"  fy = {intr.fy}")
    print(f"  cx = {intr.cx}")
    print(f"  cy = {intr.cy}")
    print(f"\n  Distortion (k1 k2 p1 p2 k3):")
    print(f"  k1={dist.k1}  k2={dist.k2}  k3={dist.k3}")
    print(f"  p1={dist.p1}  p2={dist.p2}")
    print(f"\nCopy-paste ({label}):")
    print(f"K_{label} = np.array([")
    print(f"    [{intr.fx}, 0.0, {intr.cx}],")
    print(f"    [0.0, {intr.fy}, {intr.cy}],")
    print(f"    [0.0, 0.0, 1.0],")
    print(f"], dtype=np.float64)")
    print(f"DIST_{label} = np.array([{dist.k1}, {dist.k2}, {dist.p1}, {dist.p2}, {dist.k3}], dtype=np.float64)  # k1,k2,p1,p2,k3")


def main():
    pipeline = Pipeline()
    config   = Config()

    # Enable both depth/IR and color streams so all params are populated.
    try:
        depth_list    = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)
    except Exception as e:
        print(f"Warning: could not enable depth stream: {e}")

    try:
        color_list    = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = color_list.get_default_video_stream_profile()
        config.enable_stream(color_profile)
    except Exception as e:
        print(f"Warning: could not enable color stream: {e}")

    pipeline.start(config)

    try:
        param = pipeline.get_camera_param()
    finally:
        pipeline.stop()

    ir = param.depth_intrinsic
    dc = param.depth_distortion
    rgb = param.rgb_intrinsic
    rc = param.rgb_distortion
    ex = param.transform          # OBExtrinsic: rotation (3x3 row-major) + translation (mm)

    print("=" * 60)
    print("Orbbec Femto Bolt — IR / Depth sensor intrinsics")
    print("=" * 60)
    print_intrinsic("IR", ir, dc)

    print()
    print("=" * 60)
    print("Orbbec Femto Bolt — RGB sensor intrinsics")
    print("=" * 60)
    print_intrinsic("RGB", rgb, rc)

    print()
    print("=" * 60)
    print("IR -> RGB extrinsics  (rotation row-major 3x3, translation mm)")
    print("=" * 60)
    R = ex.rot        # 3x3 numpy array
    t = ex.transform  # 3-element array (mm)
    print(f"  R = [")
    for row in range(3):
        print(f"    {R[row, 0]:.8f},  {R[row, 1]:.8f},  {R[row, 2]:.8f},")
    print(f"  ]")
    print(f"  t = [{t[0]:.6f},  {t[1]:.6f},  {t[2]:.6f}]  (mm)")
    print()
    print("Copy-paste (extrinsics):")
    print(f"R_ir_to_rgb = np.array([")
    for row in range(3):
        print(f"    [{R[row, 0]}, {R[row, 1]}, {R[row, 2]}],")
    print(f"], dtype=np.float64)")
    print(f"t_ir_to_rgb = np.array([{t[0]}, {t[1]}, {t[2]}], dtype=np.float64)  # mm")


if __name__ == "__main__":
    main()
