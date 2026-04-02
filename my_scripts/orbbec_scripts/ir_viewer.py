"""
Orbbec Femto Bolt - Synchronized Active IR + RGB viewer and recorder.

The Femto Bolt alternates every hardware frame between:
  - Flood IR  (room lit up, max ~1000-2000): skipped for IR, RGB still recorded
  - Active IR (ToF laser pulse, retroreflective markers saturate to 65535): recorded

Frame rates:
  RGB : 30fps (every frameset, full rate)
  IR  : ~15fps (active IR only — hardware limitation, every other frame is flood)

Saves:
  ir_YYYYMMDD_HHMMSS/
      frame_000000.npy  — raw 16-bit IR (1024x1024 uint16), lossless, ~15fps
      frame_000001.npy
      ...
      sync_index.npy    — int array: sync_index[i] = RGB frame number for IR frame i
  rgb_YYYYMMDD_HHMMSS.avi  — color video (1920x1080 @ 30fps, MJPG)

sync_index.npy lets you find the exact RGB frame captured at the same moment as
IR frame i:  rgb_frame_number = sync_index[i]

Run with:
    conda activate orbbec
    python ir_viewer.py
Press 'q' to quit and save.
"""

import time
import os
import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat

SAVE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE_IR_THRESHOLD = 5000  # frames with max below this are flood IR, skip them


def main():
    pipeline = Pipeline()
    config = Config()

    # IR stream: 1024x1024 @ 30fps Y16
    ir_profiles = pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
    try:
        ir_profile = ir_profiles.get_video_stream_profile(1024, 1024, OBFormat.Y16, 30)
    except Exception:
        ir_profile = ir_profiles.get_default_video_stream_profile()
    config.enable_stream(ir_profile)

    # RGB stream: 1920x1080 @ 30fps MJPG
    color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    try:
        color_profile = color_profiles.get_video_stream_profile(1920, 1080, OBFormat.MJPG, 30)
    except Exception:
        color_profile = color_profiles.get_default_video_stream_profile()
    config.enable_stream(color_profile)

    pipeline.start(config)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # IR: save raw 16-bit frames as .npy files — lossless, no codec issues
    ir_dir = os.path.join(SAVE_DIR, f"ir_{timestamp}")
    os.makedirs(ir_dir, exist_ok=True)

    # RGB: MJPG+AVI at full 30fps
    rgb_path = os.path.join(SAVE_DIR, f"rgb_{timestamp}.avi")
    rgb_writer = cv2.VideoWriter(rgb_path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1920, 1080))
    if not rgb_writer.isOpened():
        print("ERROR: Could not open RGB video writer.")
        pipeline.stop()
        return

    print("Streaming started.")
    print(f"  IR  -> {ir_dir}/frame_NNNNNN.npy  (~15fps, raw 16-bit)")
    print(f"  RGB -> {rgb_path}  (30fps)")
    print("Press 'q' to quit and save.")

    fps = 0.0
    frame_count = 0
    saved_ir_count = 0
    rgb_frame_count = 0     # total RGB frames written (30fps)
    sync_index = []         # sync_index[i] = RGB frame number paired with IR frame i
    t_start = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            ir_frame = frames.get_ir_frame()
            color_frame = frames.get_color_frame()
            if ir_frame is None or color_frame is None:
                continue

            # Decode RGB and write every frame at 30fps
            color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
            color_img = cv2.imdecode(color_data, cv2.IMREAD_COLOR)
            if color_img is None:
                continue
            rgb_writer.write(color_img)
            current_rgb_idx = rgb_frame_count
            rgb_frame_count += 1

            # Raw 16-bit IR
            ir_data = np.frombuffer(ir_frame.get_data(), dtype=np.uint16).reshape(
                (ir_frame.get_height(), ir_frame.get_width()))

            # Skip flood IR frames — only save active IR (~15fps)
            if ir_data.max() < ACTIVE_IR_THRESHOLD:
                continue

            # Save IR as raw 16-bit numpy array
            np.save(os.path.join(ir_dir, f"frame_{saved_ir_count:06d}.npy"), ir_data)
            sync_index.append(current_rgb_idx)
            saved_ir_count += 1

            # FPS (IR rate)
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            # Display IR (scale 16-bit -> 8-bit for preview)
            ir_8 = (ir_data >> 8).astype(np.uint8)
            ir_disp = cv2.cvtColor(ir_8, cv2.COLOR_GRAY2BGR)
            cv2.putText(ir_disp, f"IR {fps:.1f}fps  #{saved_ir_count}  |  RGB #{rgb_frame_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Active IR (q to quit)", ir_disp)

            # Display RGB (scaled down for screen)
            rgb_disp = cv2.resize(color_img, (960, 540))
            cv2.putText(rgb_disp, f"RGB 30fps  #{rgb_frame_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("RGB (q to quit)", rgb_disp)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        rgb_writer.release()
        cv2.destroyAllWindows()
        np.save(os.path.join(ir_dir, "sync_index.npy"), np.array(sync_index, dtype=np.int32))
        print(f"Saved {saved_ir_count} IR frames (~15fps), {rgb_frame_count} RGB frames (30fps).")
        print(f"  IR   -> {ir_dir}/")
        print(f"  RGB  -> {rgb_path}")
        print(f"  Sync -> {ir_dir}/sync_index.npy")


if __name__ == "__main__":
    main()
