"""Monocular depth estimation with MiDaS and a centre-region depth read-out.

Every video frame is passed through a MiDaS model loaded from torch.hub
(intel-isl/MiDaS). The predicted depth map is min-max normalised to 0-255,
colour-mapped (MAGMA) and shown next to the input frame. The mean value of a
square region at the centre of the normalised map is multiplied by an
empirical scale factor (--depth-scale) and printed on the frame.

Note: MiDaS predicts *relative inverse* depth and the map is re-normalised for
every frame, so the read-out is an uncalibrated indicator, not a metric
distance. --depth-scale has to be calibrated for a given camera and scene.

Original file: 距離判斷_單目深度估計.py (v1).
See midas_depth_video.py for the refactored v2 that can also save a video.

Usage:
    python midas_depth_readout.py path/to/video.mp4 --model DPT_Large
Press 'q' to quit.
"""
import argparse

import cv2
import numpy as np
import torch

# Assumed depth-to-distance scale (calibrate for the actual camera)
DEPTH_SCALE = 0.1
# Half size of the centre region (pixels)
REGION_SIZE = 20


def load_midas(model_type):
    """Load a MiDaS model and its matching input transform from torch.hub."""
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()

    # Select GPU/CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)

    # Load the model transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform if model_type == "DPT_Large" else midas_transforms.small_transform
    return midas, transform, device


def run(video_path, model_type, depth_scale, region_size):
    midas, transform, device = load_midas(model_type)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
    else:
        print("Video opened successfully.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame or end of video reached.")
            break

        # Prepare the input tensor
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(input_image).to(device)

        # Force input_tensor into shape [1, 3, H, W]
        if input_tensor.dim() == 5:
            input_tensor = input_tensor.squeeze(1)  # drop the extra 2nd dimension
        elif input_tensor.dim() == 4 and input_tensor.shape[1] == 1:
            input_tensor = input_tensor.squeeze(1)  # drop the singleton channel

        # Make sure the final shape is [1, 3, H, W]
        input_tensor = input_tensor.unsqueeze(0) if input_tensor.dim() == 3 else input_tensor

        # Depth estimation
        with torch.no_grad():
            depth_map = midas(input_tensor).squeeze().cpu().numpy()

        # Post-process the depth map
        depth_map = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

        # Match the colour map size to the frame size
        depth_colormap = cv2.resize(depth_colormap, (frame.shape[1], frame.shape[0]))

        # Estimate the value of the centre region
        center_x, center_y = depth_map.shape[1] // 2, depth_map.shape[0] // 2
        center_region = depth_map[center_y - region_size:center_y + region_size,
                                  center_x - region_size:center_x + region_size]
        avg_depth = np.mean(center_region)  # mean depth value of the centre region
        distance_cm = avg_depth * depth_scale  # convert with the empirical scale

        # Draw the read-out on the frame
        cv2.putText(frame, f"Distance: {distance_cm:.2f} cm", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame and the depth map side by side
        combined = np.hstack((frame, depth_colormap))
        cv2.imshow("Frame and Depth Map", combined)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close windows
    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="MiDaS monocular depth estimation with a centre-region read-out.")
    parser.add_argument("video", help="path to the input video")
    parser.add_argument("--model", default="DPT_Large", choices=["DPT_Large", "MiDaS_small"],
                        help="MiDaS model type loaded via torch.hub (default: DPT_Large)")
    parser.add_argument("--depth-scale", type=float, default=DEPTH_SCALE,
                        help=f"empirical depth-to-distance scale (default: {DEPTH_SCALE})")
    parser.add_argument("--region-size", type=int, default=REGION_SIZE,
                        help=f"half size of the centre region in pixels (default: {REGION_SIZE})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.video, args.model, args.depth_scale, args.region_size)
