import pyrealsense2 as rs
import numpy as np
import cv2

# Setup pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            continue

        # Convert images
        depth_image = np.asanyarray(depth.get_data())
        color_image = np.asanyarray(color.get_data())

        # Depth colormap
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack and display
        combined = np.hstack((color_image, depth_colormap))
        cv2.imshow('RealSense', combined)

        if cv2.waitKey(1) == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
