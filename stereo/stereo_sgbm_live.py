"""Live stereo disparity from two webcams (StereoSGBM).

Frames are read from two cameras, converted to grayscale and matched with
OpenCV's semi-global block matcher (StereoSGBM). Depth is computed as

    Z = (B * f) / d

with the baseline B in millimetres, the focal length f in pixels and the
disparity d in pixels. The script displays the two camera views and the
disparity map; the depth array is computed but not visualised in this
prototype.

Original file: 距離判斷_雙眼.py

Usage:
    python stereo_sgbm_live.py --left-camera 0 --right-camera 1 --baseline 100 --focal-length 800
Press 'q' to quit.
"""
import argparse

import cv2
import numpy as np

# SGBM parameters
WINDOW_SIZE = 5
MIN_DISP = 0
NUM_DISP = 16 * 5  # must be a multiple of 16

# Baseline (distance between the two cameras, millimetres)
BASELINE_MM = 100  # set to the actual camera spacing
# Focal length (pixels)
FOCAL_LENGTH = 800  # set from camera calibration


def create_matcher(window_size=WINDOW_SIZE, min_disp=MIN_DISP, num_disp=NUM_DISP):
    return cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=window_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )


def run(left_index, right_index, baseline, focal_length):
    # Open the left and right cameras
    cap_left = cv2.VideoCapture(left_index)  # left camera
    cap_right = cv2.VideoCapture(right_index)  # right camera

    stereo = create_matcher()

    while True:
        # Read the left and right images
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or not ret_right:
            print("Could not read frames")
            break

        # Convert to grayscale
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # Disparity map
        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

        # Depth map in millimetres (small epsilon avoids division by zero);
        # computed for reference, not displayed in this prototype
        depth = (baseline * focal_length) / (disparity + 1e-5)

        # Show the images
        cv2.imshow('Left', frame_left)
        cv2.imshow('Right', frame_right)
        cv2.imshow('Disparity', (disparity - MIN_DISP) / NUM_DISP)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live stereo disparity (StereoSGBM) from two webcams.")
    parser.add_argument("--left-camera", type=int, default=0,
                        help="index of the left camera (default: 0)")
    parser.add_argument("--right-camera", type=int, default=1,
                        help="index of the right camera (default: 1)")
    parser.add_argument("--baseline", type=float, default=BASELINE_MM,
                        help=f"distance between the cameras in mm (default: {BASELINE_MM})")
    parser.add_argument("--focal-length", type=float, default=FOCAL_LENGTH,
                        help=f"focal length in pixels (default: {FOCAL_LENGTH})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.left_camera, args.right_camera, args.baseline, args.focal_length)
