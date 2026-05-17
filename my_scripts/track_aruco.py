"""
Track ArUco markers from webcam and draw XYZ axes on each detected marker.
Uses 4x4_50 dictionary by default (matching the printed marker).

Usage:
    python track_aruco.py
    python track_aruco.py --marker-size 0.05          # marker physical size in metres
    python track_aruco.py --dict 4x4_100              # use a different dictionary
    python track_aruco.py --output recording.mp4      # save annotated video
"""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")  # fix black window in conda envs

import argparse
import numpy as np
import cv2
import cv2.aruco as aruco

ARUCO_DICTS = {
    "4x4_50":   aruco.DICT_4X4_50,
    "4x4_100":  aruco.DICT_4X4_100,
    "4x4_250":  aruco.DICT_4X4_250,
    "4x4_1000": aruco.DICT_4X4_1000,
    "5x5_50":   aruco.DICT_5X5_50,
    "6x6_50":   aruco.DICT_6X6_50,
}


def build_camera_matrix(frame_width: int, frame_height: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a rough camera intrinsic matrix from frame size.
    Focal length is estimated as ~70% of the larger dimension.
    Replace with real calibration data for accurate pose.
    """
    fx = fy = max(frame_width, frame_height) * 0.7
    cx, cy = frame_width / 2.0, frame_height / 2.0
    camera_matrix = np.array(
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs


def main():
    parser = argparse.ArgumentParser(description="ArUco marker tracker with XYZ axes overlay")
    parser.add_argument("--camera",      type=int,   default=0,        help="Camera device index")
    parser.add_argument("--dict",        type=str,   default="4x4_50", choices=ARUCO_DICTS.keys())
    parser.add_argument("--marker-size", type=float, default=0.05,     help="Physical marker side length in metres")
    parser.add_argument("--output",      type=str,   default=None,     help="Save annotated video to this file (e.g. out.mp4)")
    args = parser.parse_args()

    # --- ArUco setup ---
    aruco_dict   = aruco.getPredefinedDictionary(ARUCO_DICTS[args.dict])
    detector_params = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, detector_params)

    # --- Webcam ---
    # Open with explicit V4L2 backend so format props take effect before streaming
    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")

    # Must set FOURCC before the first read on V4L2
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Drain initial black frames (camera warm-up)
    for _ in range(30):
        cap.read()

    print(f"Tracking ArUco ({args.dict}), marker size {args.marker_size*100:.1f} cm. Press Q to quit.")

    cv2.namedWindow("ArUco Tracker — Q to quit", cv2.WINDOW_NORMAL)
    camera_matrix = dist_coeffs = None
    writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        h, w = frame.shape[:2]
        if camera_matrix is None:
            camera_matrix, dist_coeffs = build_camera_matrix(w, h)

        if args.output and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
            print(f"Saving video to: {args.output}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            # Draw detected marker borders
            aruco.drawDetectedMarkers(frame, corners, ids)

            # Estimate pose for each marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, args.marker_size, camera_matrix, dist_coeffs
            )

            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                # Draw XYZ axes:  X=red, Y=green, Z=blue (into the camera)
                cv2.drawFrameAxes(
                    frame, camera_matrix, dist_coeffs,
                    rvec, tvec,
                    length=args.marker_size * 0.8,   # axis length as fraction of marker
                    thickness=3,
                )

                # Label marker ID above the top-left corner
                corner_pts = corners[i][0]          # shape (4, 2)
                top_left   = corner_pts[0].astype(int)
                mid_x = int(corner_pts[:, 0].mean())
                mid_y = int(corner_pts[:, 1].mean())

                cv2.putText(
                    frame,
                    f"ID {ids[i][0]}",
                    (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                )

                # Distance from camera (Z translation)
                dist_m = float(np.linalg.norm(tvec))
                cv2.putText(
                    frame,
                    f"{dist_m*100:.1f} cm",
                    (mid_x - 30, mid_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                )
        else:
            cv2.putText(
                frame, "No markers detected",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2,
            )

        if writer is not None:
            writer.write(frame)

        cv2.imshow("ArUco Tracker — Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Video saved: {args.output}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
