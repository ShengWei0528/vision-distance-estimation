"""Monocular depth estimation on a video with MiDaS (refactored, can save output).

A DepthEstimator class wraps a MiDaS model loaded from torch.hub
(intel-isl/MiDaS). Each frame is converted to a min-max normalised,
MAGMA colour-mapped depth map that is shown side by side with the original
frame and, optionally, written to an output video (width x 2).

MiDaS predicts relative inverse depth (brighter = closer in this colour map);
the output is a visualisation, not a metric distance.

Original file: 距離判斷_單目深度估計2.py (v2).
v2 restructures v1 (see midas_depth_readout.py) into a class and adds video
export; the centre-region read-out of v1 is not part of v2.

Usage:
    python midas_depth_video.py path/to/video.mp4 --output output_depth.mp4
Press 'q' to quit.
"""
import argparse

import cv2
import numpy as np
import torch


class DepthEstimator:
    def __init__(self, model_type: str = "DPT_Large", debug: bool = False):
        """
        Initialise the depth estimator.

        Args:
            model_type (str): MiDaS model type ("DPT_Large" or "MiDaS_small")
            debug (bool): print the input tensor shape for every frame
        """
        self.model_type = model_type
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()

    def _initialize_model(self):
        """Load the MiDaS model and its input transform."""
        try:
            # Load the MiDaS model
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.model.eval()
            self.model.to(self.device)

            # Load the matching transform
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = (midas_transforms.dpt_transform
                              if self.model_type == "DPT_Large"
                              else midas_transforms.small_transform)

            print(f"Model initialised on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Model initialisation failed: {str(e)}")

    def process_frame(self, frame: np.ndarray):
        """
        Produce a depth map for a single frame.

        Args:
            frame (np.ndarray): input frame in BGR format
        """
        # BGR -> RGB
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prepare the input tensor
        input_tensor = self.transform(input_image).to(self.device)

        # Make sure the tensor shape is [1, 3, H, W]
        if input_tensor.dim() == 5:  # shape [1, 1, 3, H, W]
            input_tensor = input_tensor.squeeze(1)  # drop the extra dimension

        if self.debug:
            print(f"Input tensor shape: {input_tensor.shape}")

        # Predict the depth map
        with torch.no_grad():
            depth_map = self.model(input_tensor).squeeze().cpu().numpy()

        # Post-process the depth map
        depth_map = cv2.normalize(depth_map, None, 0, 255,
                                  norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        depth_colormap = cv2.resize(depth_colormap,
                                    (frame.shape[1], frame.shape[0]))

        return frame, depth_colormap


def process_video(video_path: str, output_path: str = None,
                  model_type: str = "DPT_Large", debug: bool = False):
    """
    Process a video file and visualise its depth.

    Args:
        video_path (str): input video path
        output_path (str, optional): path of the output video to save
        model_type (str): MiDaS model type
        debug (bool): print per-frame tensor shapes
    """
    # Initialise the depth estimator
    depth_estimator = DepthEstimator(model_type, debug)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Error: could not open the video file")

    # Initialise the writer if an output path is given
    writer = None
    if output_path:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        writer = cv2.VideoWriter(output_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps,
                                 (frame_width * 2, frame_height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process the frame
            original, depth_map = depth_estimator.process_frame(frame)

            # Concatenate horizontally
            combined = np.hstack((original, depth_map))

            # Show the result
            cv2.imshow("Original and Depth Map", combined)

            # Save the frame if a writer exists
            if writer:
                writer.write(combined)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error while processing the video: {str(e)}")

    finally:
        # Clean up
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="MiDaS monocular depth visualisation for a video file.")
    parser.add_argument("video", help="path to the input video")
    parser.add_argument("--output", default=None,
                        help="optional path for the side-by-side output video (e.g. output_depth.mp4)")
    parser.add_argument("--model", default="DPT_Large", choices=["DPT_Large", "MiDaS_small"],
                        help="MiDaS model type loaded via torch.hub (default: DPT_Large)")
    parser.add_argument("--debug", action="store_true",
                        help="print the input tensor shape for every frame")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(args.video, args.output, args.model, args.debug)
