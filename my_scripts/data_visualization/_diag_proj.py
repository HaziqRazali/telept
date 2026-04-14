"""Quick diagnostic: check C3D→IR and C3D→RGB projection chain."""
import cv2, numpy as np, ezc3d, glob, os

TAKE = "/data/telept/data/mocap/20141017/take1"
pnp  = np.load(os.path.join(TAKE, "pnp_results.npz"))
rvec = pnp["rvec_median"].reshape(3,1)
tvec = pnp["tvec_median"].reshape(3,1)
K_ir = pnp["K"]
print(f"rvec: {rvec.flatten()}")
print(f"tvec: {tvec.flatten()}")
print(f"K_IR:\n{K_ir}")

R_ir_rgb = np.array([
    [ 0.9938264489173889, -0.0014704565983265638,  0.0015902734594419599],
    [ 0.0012849620543420315,  0.9938266277313232,  0.11093678325414658 ],
    [-0.0017435838235542178, -0.11093448102474213,  0.993826150894165  ],
])
t_ir_rgb = np.array([-32.8342170715332, -1.315084457397461, 1.3067550659179688]) / 1000.0
K_rgb = np.array([[1123.86669921875,0,948.0269165039062],
                  [0,1123.028076171875,539.6485595703125],
                  [0,0,1]])
DIST_RGB = np.array([0.07333821058273315,-0.10178927332162857,
                     -0.0004722462617792189,-0.00022512981377076358,
                     0.041689008474349976])

R_c2i, _ = cv2.Rodrigues(rvec)
R_c2r = R_ir_rgb @ R_c2i
t_c2r = (R_ir_rgb @ tvec).flatten() + t_ir_rgb
rv_rgb, _ = cv2.Rodrigues(R_c2r)
tv_rgb = t_c2r.reshape(3,1)
print(f"\nC3D->RGB rvec: {rv_rgb.flatten()}")
print(f"C3D->RGB tvec: {tv_rgb.flatten()}")

# Sanity: C3D world origin – what pixel does it land on?
p0 = np.zeros((1,1,3), dtype=np.float64)
ir_o,  _ = cv2.projectPoints(p0, rvec,   tvec,   K_ir, np.zeros(5))
rgb_o, _ = cv2.projectPoints(p0, rv_rgb, tv_rgb, K_rgb, DIST_RGB)
print(f"\nC3D origin → IR  pixel: {ir_o.reshape(2)}")
print(f"C3D origin → RGB pixel: {rgb_o.reshape(2)}")

# Load C3D, project the 5 expected markers at a reference frame
c    = ezc3d.c3d(glob.glob(os.path.join(TAKE, "*.c3d"))[0])
xyz  = c["data"]["points"][:3]   # (3, M, F)
# Use c3d_frame_offset from offset.txt as the sync reference frame
offset = {}
with open(os.path.join(TAKE, "offset.txt")) as f:
    for line in f:
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            offset[k.strip()] = v.strip()
c3d_ref = int(offset["optitrack_frame"])

pts = xyz[:, :, c3d_ref].T   # (M, 3)
vis = np.where(np.all(np.isfinite(pts), axis=1))[0]
print(f"\nReference C3D frame {c3d_ref}: {len(vis)} visible markers")

# Use the valid_ir_frames from npz to identify expected marker indices
valid_ir = pnp["valid_ir_frames"]
expected_n = int(pnp["expected_n"])
print(f"expected_n={expected_n}  valid IR frames in npz: {len(valid_ir)}")

pts3d = pts[vis].astype(np.float64)
ir_pts,  _ = cv2.projectPoints(pts3d, rvec,   tvec,   K_ir, np.zeros(5))
rgb_pts, _ = cv2.projectPoints(pts3d, rv_rgb, tv_rgb, K_rgb, DIST_RGB)
print(f"\nAll visible markers at reference frame (IR 640x576, RGB 1920x1080):")
in_ir  = lambda p: 0<=p[0]<640 and 0<=p[1]<576
in_rgb = lambda p: 0<=p[0]<1920 and 0<=p[1]<1080
for i in range(len(pts3d)):
    ip = ir_pts[i,0]; rp = rgb_pts[i,0]
    if in_ir(ip) or in_rgb(rp):
        print(f"  marker {vis[i]:3d}  3D=({pts3d[i,0]:.3f},{pts3d[i,1]:.3f},{pts3d[i,2]:.3f})m"
              f"  IR=({ip[0]:.0f},{ip[1]:.0f})  RGB=({rp[0]:.0f},{rp[1]:.0f})")
