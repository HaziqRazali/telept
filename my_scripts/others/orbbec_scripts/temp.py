#!/usr/bin/env python3
"""Create a black-and-white image with white filled circles at coordinates
listed in a .txt file. Defaults point to the files you specified.

Usage:
    python temp.py [video_path] [txt_path] [output_path] [radius]

If the video can't be opened, the script defaults to 640x480.
If coordinates are normalized in [0,1], they'll be scaled to frame size.
"""
import os
import sys
import cv2
import numpy as np


def parse_coords(txt_path):
    import re
    num_re = re.compile(r"[-+]?\d*\.?\d+")
    coords = []
    with open(txt_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            # Preferred: explicit x=..., y=... pairs
            mx = re.search(r"x\s*=\s*([-+]?\d*\.?\d+)", s, re.I)
            my = re.search(r"y\s*=\s*([-+]?\d*\.?\d+)", s, re.I)
            if mx and my:
                try:
                    x = float(mx.group(1))
                    y = float(my.group(1))
                    coords.append((x, y))
                    continue
                except Exception:
                    pass

            # Fallback: extract numbers, but skip a leading index like "1:"
            nums = num_re.findall(s)
            if not nums:
                continue

            # If line starts with an index like '1:' then nums[0] is that index
            if re.match(r"^\s*\d+\s*:\s*", s) and len(nums) >= 3:
                try:
                    x = float(nums[1])
                    y = float(nums[2])
                    coords.append((x, y))
                    continue
                except Exception:
                    continue

            if len(nums) >= 2:
                try:
                    x = float(nums[0])
                    y = float(nums[1])
                    coords.append((x, y))
                except Exception:
                    continue
    return coords


def get_frame_size(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    h, w = frame.shape[:2]
    return w, h


def create_image(w, h, coords, radius=5):
    img = np.zeros((h, w), dtype=np.uint8)
    if not coords:
        return img
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    # detect normalized coords in [0,1]
    norm = False
    try:
        if max(abs(x) for x in xs) <= 1.01 and max(abs(y) for y in ys) <= 1.01:
            norm = True
    except ValueError:
        norm = False

    for x, y in coords:
        if norm:
            xi = int(round(x * w))
            yi = int(round(y * h))
        else:
            xi = int(round(x))
            yi = int(round(y))
        # clamp
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            continue

        print(xi, yi)
        cv2.circle(img, (xi, yi), int(radius), 255, thickness=-1)
    return img


def main():
    default_video = "/home/haziq/datasets/telept/data/mocap/2026_03_20.mp4"
    default_txt = "/home/haziq/datasets/telept/data/mocap/2026_03_20_circle.txt"
    default_out = "/home/haziq/datasets/telept/my_scripts/orbbec_scripts/temp_output.png"

    args = sys.argv[1:]
    video = args[0] if len(args) >= 1 else default_video
    txt = args[1] if len(args) >= 2 else default_txt
    out = args[2] if len(args) >= 3 else default_out
    radius = int(args[3]) if len(args) >= 4 else 5

    if not os.path.exists(txt):
        print(f"TXT file not found: {txt}")
        sys.exit(1)

    size = get_frame_size(video)
    if size is None:
        print(f"Warning: can't open video '{video}'; defaulting to 640x480")
        w, h = 640, 480
    else:
        w, h = size

    coords = parse_coords(txt)
    if not coords:
        print(f"No coordinates parsed from {txt}")

    img = create_image(w, h, coords, radius=radius)
    saved = cv2.imwrite(out, img)
    if saved:
        print(f"Wrote image to {out}")
    else:
        print(f"Failed to write image to {out}")


if __name__ == "__main__":
    main()
