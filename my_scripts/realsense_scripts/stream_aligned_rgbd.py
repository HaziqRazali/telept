import pyrealsense2 as rs
import numpy as np
import cv2

# Setup pipeline and config
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline_profile = pipeline.start(config)

# Get depth scale
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # in meters per unit

# Create align object (align depth to color)
align = rs.align(rs.stream.color)

# Mouse callback
mouse_x, mouse_y = -1, -1

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x, mouse_y = x, y

cv2.namedWindow("RGB | Depth | Overlay")
cv2.setMouseCallback("RGB | Depth | Overlay", mouse_callback)

try:
    while True:
        # Wait for frames and align them
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image_raw = np.asanyarray(depth_frame.get_data())  # uint16

        # --- Jet-colored depth image ---
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image_raw, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # --- Depth overlay on color ---
        height, width = depth_image_raw.shape
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
        depth_data = depth_data.astype(np.float32) * depth_scale  # convert to meters
        normalized_depth = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_overlay = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(color_image, 0.5, depth_overlay, 0.5, 0)

        # --- Combine all views ---
        combined = np.hstack((color_image, depth_colormap, blended))
        cv2.imshow("RGB | Depth | Overlay", combined)

        # --- Print distance under mouse ---
        view_width = width  # 640
        if 0 <= mouse_x < view_width * 3 and 0 <= mouse_y < height:
            # Translate x back to original coordinate
            x_in_image = mouse_x % view_width
            distance = depth_frame.get_distance(x_in_image, mouse_y)
            print(f"Mouse at ({mouse_x}, {mouse_y}) → Pixel ({x_in_image}, {mouse_y}) → Distance: {distance:.3f} meters", end='\r')

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):  # ESC or q
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
