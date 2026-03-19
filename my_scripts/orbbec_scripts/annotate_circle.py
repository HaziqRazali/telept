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
"""

import cv2
import os

VIDEO_PATH = "/data/telept/my_scripts/orbbec_scripts/recording_20260318_161152.mp4"
OUT_TXT    = os.path.splitext(VIDEO_PATH)[0] + "_circle.txt"

CIRCLE_RADIUS = 10   # display radius (pixels)
CIRCLE_COLOR  = (0, 255, 0)

circles = []   # list of (x, y)
frame_orig = None


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
        circles.append((x, y))
        cv2.imshow("Annotate Circles", draw_overlay(frame_orig))


def main():
    global frame_orig

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: cannot open {VIDEO_PATH}")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: could not read first frame.")
        return

    frame_orig = frame.copy()

    cv2.namedWindow("Annotate Circles")
    cv2.setMouseCallback("Annotate Circles", mouse_cb)
    cv2.imshow("Annotate Circles", draw_overlay(frame_orig))

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):  # Enter
            if not circles:
                print("No circles placed yet.")
                continue
            with open(OUT_TXT, "w") as f:
                for i, (x, y) in enumerate(circles):
                    f.write(f"{i + 1}: x={x}, y={y}\n")
            print(f"Saved {len(circles)} point(s) to: {OUT_TXT}")
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
