"""
Orbbec Femto Bolt - RGB-only stream viewer with FPS display and video recording.
Records to the same folder as this script.
Run with:
    conda activate orbbec
    python stream_viewer.py
Press 'q' to quit and save the recording.
"""

import time
import os
import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    pipeline = Pipeline()
    config = Config()

    # Color-only stream for maximum FPS: 1920x1080 @ 30fps MJPG
    profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        color_profile = profile_list.get_video_stream_profile(1920, 1080, OBFormat.MJPG, 30)
    except Exception:
        color_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    pipeline.start(config)

    # Video writer setup
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(SAVE_DIR, f"recording_{timestamp}.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    print(f"Streaming started. Recording to: {video_path}")
    print("Press 'q' to quit and save.")

    # FPS tracking
    fps = 0.0
    frame_count = 0
    t_start = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue

            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_img = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
            if color_img is None:
                continue

            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 0.5:  # update FPS every 0.5s
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            writer.write(color_img)

            # Overlay FPS
            cv2.putText(color_img, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("Femto Bolt - Color (press q to quit)", color_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        writer.release()
        cv2.destroyAllWindows()
        print(f"Saved: {video_path}")


if __name__ == "__main__":
    main()
