"""Distance read-out with an Intel RealSense depth camera.

Depth (z16) and colour (bgr8) streams are started at 640x480 @ 30 fps.
For every frame the distance at the image centre is read with
depth_frame.get_distance(), which returns metres, and drawn on the colour
image. The colour image and a JET colour map of the depth image are shown
side by side.

Original file: 距離判斷_深度相機.py

Credit: the stream setup and the depth colour-mapping code (convertScaleAbs
alpha=0.03 + COLORMAP_JET, np.hstack of colour and depth) are adapted from
the Intel RealSense SDK 2.0 (librealsense) Python example
wrappers/python/examples/opencv_viewer_example.py, licensed under the
Apache License 2.0 (https://github.com/IntelRealSense/librealsense).

Usage (requires a connected RealSense camera and pyrealsense2):
    python realsense_center_distance.py
Press 'q' to quit.
"""
import argparse

import cv2
import numpy as np
import pyrealsense2 as rs


def main():
    # Configure the RealSense streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable the depth and colour streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for a frame set
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert the images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Target point: the image centre
            height, width = depth_image.shape
            center_x, center_y = width // 2, height // 2

            # Distance (depth) at the centre point, in metres
            distance = depth_frame.get_distance(center_x, center_y)

            # Draw the distance on the colour image
            cv2.circle(color_image, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(color_image, f"Distance: {distance:.2f} meters", (center_x - 100, center_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the colour image and the colour-mapped depth image
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            cv2.imshow('RealSense Depth and Color Stream', images)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()


def parse_args():
    # No options: the stream settings are fixed (640x480 @ 30 fps). The parser
    # only provides --help, like the other scripts.
    parser = argparse.ArgumentParser(
        description="Show the depth (in metres) at the image centre from a connected "
                    "Intel RealSense camera, next to a JET colour map of the depth stream.")
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    main()
