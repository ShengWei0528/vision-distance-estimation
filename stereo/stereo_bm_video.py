"""Stereo distance estimation from a pair of left/right videos (StereoBM).

For each synchronised frame pair a disparity map is computed with OpenCV's
block matcher (StereoBM) and converted to depth with

    Z = (f * B) / d

where f is the focal length (pixels), B the baseline between the two cameras
(metres) and d the disparity (pixels). Distances between 0 and 10 m are
written onto the left frame on a regular grid; disparity and depth maps are
shown as normalised grayscale images.

Original file: 距離判斷_影片_雙眼2.py (v2).
v1 (距離判斷_影片_雙眼.py) showed the raw left/right frames and the maps
and used a 0.08 m baseline; v2 adds the on-image distance grid and uses 0.1 m.

Usage:
    python stereo_bm_video.py left_video.mp4 right_video.mp4 --baseline 0.1 --focal-length 800
Press 'q' to quit.
"""
import argparse

import cv2
import numpy as np

# Camera parameters (placeholder values; replace with real calibration results)
FOCAL_LENGTH = 800  # focal length (pixels)
BASELINE = 0.1  # camera baseline (metres)


def calculate_depth(disparity, focal_length, baseline):
    """
    Convert a disparity map to a depth map.

    :param disparity: disparity map (pixels)
    :param focal_length: focal length (pixels)
    :param baseline: camera baseline (metres)
    :return: depth map (metres); 0 where the disparity is invalid
    """
    depth = np.zeros(disparity.shape, dtype=np.float32)
    # Avoid division by zero: only use pixels with a positive disparity
    depth[disparity > 0] = (focal_length * baseline) / disparity[disparity > 0]
    return depth


def annotate_depth(frame, depth_map, step=50):
    """
    Write distances onto the image.

    :param frame: original image (left view)
    :param depth_map: depth map
    :param step: grid spacing of the labels (pixels)
    """
    annotated_frame = frame.copy()
    for y in range(0, depth_map.shape[0], step):
        for x in range(0, depth_map.shape[1], step):
            distance = depth_map[y, x]
            if 0 < distance < 10:  # skip invalid or too-distant values
                cv2.putText(annotated_frame, f"{distance:.2f}m", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return annotated_frame


def run(left_video_path, right_video_path, focal_length, baseline,
        num_disparities=16, block_size=15, step=50):
    left_capture = cv2.VideoCapture(left_video_path)
    right_capture = cv2.VideoCapture(right_video_path)

    if not left_capture.isOpened() or not right_capture.isOpened():
        print("Could not open the video files")
        return

    # Block-matching disparity estimator
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

    while True:
        # Read one frame from each video
        ret_left, left_frame = left_capture.read()
        ret_right, right_frame = right_capture.read()

        if not ret_left or not ret_right:
            print("End of video")
            break

        # Convert to grayscale
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        # Disparity map (StereoBM returns fixed-point values scaled by 16)
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Depth map
        depth_map = calculate_depth(disparity, focal_length, baseline)

        # Write distances on the left view
        annotated_frame = annotate_depth(left_frame, depth_map, step)

        # Normalise for display
        disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

        # Show the results
        cv2.imshow("Annotated Depth", annotated_frame)
        cv2.imshow("Disparity Map", disparity_normalized.astype(np.uint8))
        cv2.imshow("Depth Map", depth_normalized.astype(np.uint8))

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    left_capture.release()
    right_capture.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stereo (StereoBM) depth estimation from left/right video files.")
    parser.add_argument("left_video", help="path to the left-camera video")
    parser.add_argument("right_video", help="path to the right-camera video")
    parser.add_argument("--focal-length", type=float, default=FOCAL_LENGTH,
                        help=f"focal length in pixels (default: {FOCAL_LENGTH})")
    parser.add_argument("--baseline", type=float, default=BASELINE,
                        help=f"distance between the cameras in metres (default: {BASELINE})")
    parser.add_argument("--num-disparities", type=int, default=16,
                        help="StereoBM numDisparities, a multiple of 16 (default: 16)")
    parser.add_argument("--block-size", type=int, default=15,
                        help="StereoBM blockSize, an odd number (default: 15)")
    parser.add_argument("--step", type=int, default=50,
                        help="grid spacing of the distance labels in pixels (default: 50)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.left_video, args.right_video, args.focal_length, args.baseline,
        args.num_disparities, args.block_size, args.step)
