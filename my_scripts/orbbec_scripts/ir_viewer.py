"""
Orbbec Femto Bolt - IR stream viewer with FPS display.
Retroreflective mocap markers appear as bright blobs in IR.

Run with:
    conda activate orbbec
    python ir_viewer.py
Press 'q' to quit.
"""

import time
import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat


def main():
    pipeline = Pipeline()
    config = Config()

    # IR stream: 1024x1024 @ 30fps
    profile_list = pipeline.get_stream_profile_list(OBSensorType.IR_SENSOR)
    try:
        ir_profile = profile_list.get_video_stream_profile(1024, 1024, OBFormat.Y16, 30)
    except Exception:
        ir_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(ir_profile)

    pipeline.start(config)
    print("IR streaming started. Press 'q' to quit.")

    fps = 0.0
    frame_count = 0
    t_start = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            ir_frame = frames.get_ir_frame()
            if ir_frame is None:
                continue

            # Raw 16-bit IR data
            ir_data = np.frombuffer(ir_frame.get_data(), dtype=np.uint16)
            ir_data = ir_data.reshape((ir_frame.get_height(), ir_frame.get_width()))

            # Normalize to 8-bit for display (auto-stretch contrast)
            ir_min, ir_max = ir_data.min(), ir_data.max()
            if ir_max > ir_min:
                ir_8 = ((ir_data - ir_min) / (ir_max - ir_min) * 255).astype(np.uint8)
            else:
                ir_8 = np.zeros_like(ir_data, dtype=np.uint8)

            # Apply CLAHE for local contrast enhancement (helps see markers)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            ir_8 = clahe.apply(ir_8)

            # Colorize with a "hot" colormap so bright markers stand out
            ir_color = cv2.applyColorMap(ir_8, cv2.COLORMAP_HOT)

            # FPS
            frame_count += 1
            elapsed = time.time() - t_start
            if elapsed >= 0.5:
                fps = frame_count / elapsed
                frame_count = 0
                t_start = time.time()

            cv2.putText(ir_color, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(ir_color, f"min={ir_min}  max={ir_max}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
            cv2.imshow("Femto Bolt - IR (press q to quit)", ir_color)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
