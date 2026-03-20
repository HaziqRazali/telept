"""
Circle annotator on the first frame of a video.
  - Each left-click places a circle at that point
  - Press 'z' to undo the last circle
  - Press 'r' to reset all circles
  - Press Enter to save all XY coordinates to a .txt file
  - Press 'q' to quit without saving

Usage:
    conda activate orbbec
    python annotate_circle.py
    python /home/haziq/datasets/telept/my_scripts/orbbec_scripts/annotate_circle.py --video_path /home/haziq/datasets/telept/data/ipad/170326_18-04/170326_18-04.mp4
"""

import cv2
import os
import argparse

VIDEO_PATH = "/data/telept/my_scripts/orbbec_scripts/recording_20260318_161152.mp4"

CIRCLE_RADIUS = 10   # display radius (pixels)
CIRCLE_COLOR  = (0, 255, 0)
MAX_DISPLAY   = (1600, 900)  # max display width, height

circles = []   # list of (x, y)
frame_orig = None
display_scale = 1.0  # ratio: display / original


def get_display_scale(w, h):
    """Return scale so the frame fits within MAX_DISPLAY."""
    sw = MAX_DISPLAY[0] / w
    sh = MAX_DISPLAY[1] / h
    return min(sw, sh, 1.0)  # never upscale


def draw_overlay(img):
    vis = img.copy()
    for i, (x, y) in enumerate(circles):
        cv2.circle(vis, (x, y), CIRCLE_RADIUS, CIRCLE_COLOR, 2)
        cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)
        cv2.putText(vis, str(i + 1), (x + 12, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    n = len(circles)
    cv2.putText(vis, f"{n} point(s) | Click=add | z=undo | r=reset | Enter=save | q=quit",
                (10, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1)
    return vis


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # With WINDOW_NORMAL, coords are already in image (original) space
        circles.append((x, y))
        cv2.imshow("Annotate Circles", draw_overlay(frame_orig))


def main():
    global frame_orig, display_scale

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default=VIDEO_PATH,
                        help="Path to the input video file")
    parser.add_argument("--max_display", type=int, nargs=2, default=list(MAX_DISPLAY),
                        metavar=("W", "H"),
                        help="Max display window size (default: 1600 900)")
    args = parser.parse_args()

    video_path = args.video_path
    out_txt = os.path.splitext(video_path)[0] + "_circle.txt"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: could not read first frame.")
        return

    frame_orig = frame.copy()
    h, w = frame_orig.shape[:2]
    MAX_DISPLAY_OVERRIDE = tuple(args.max_display)
    display_scale = min(MAX_DISPLAY_OVERRIDE[0] / w,
                        MAX_DISPLAY_OVERRIDE[1] / h, 1.0)
    if display_scale < 1.0:
        print(f"  Frame {w}x{h} → displayed at {int(w*display_scale)}x{int(h*display_scale)} "
              f"(scale={display_scale:.2f}). Saved coords are in original resolution.")

    cv2.namedWindow("Annotate Circles", cv2.WINDOW_GUI_NORMAL)
    dw = int(w * display_scale)
    dh = int(h * display_scale)
    cv2.resizeWindow("Annotate Circles", dw, dh)
    cv2.setMouseCallback("Annotate Circles", mouse_cb)
    cv2.imshow("Annotate Circles", draw_overlay(frame_orig))

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):  # Enter
            if not circles:
                print("No circles placed yet.")
                continue
            with open(out_txt, "w") as f:
                for i, (x, y) in enumerate(circles):
                    f.write(f"{i + 1}: x={x}, y={y}\n")
            print(f"Saved {len(circles)} point(s) to: {out_txt}")
            for i, (x, y) in enumerate(circles):
                print(f"  [{i+1}] ({x}, {y})")
            break

        elif key == ord('z'):  # undo
            if circles:
                removed = circles.pop()
                print(f"Removed: {removed}")
                cv2.imshow("Annotate Circles", draw_overlay(frame_orig))

        elif key == ord('r'):  # reset
            circles.clear()
            cv2.imshow("Annotate Circles", draw_overlay(frame_orig))

        elif key == ord('q'):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
