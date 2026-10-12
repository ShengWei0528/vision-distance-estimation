"""Monocular distance estimation on a video file using similar triangles.

Each frame goes through grayscale -> Canny edges -> contours. The three
contours with the largest area are boxed and labelled with

    D = (W * F) / P

where W is the known real-world width of the object (cm), F is the focal
length (pixels) and P is the perceived width of the box (pixels).

Original file: 距離判斷_影片2.py (v2).
v2 adds writing the annotated frames to a new video file; v1
(距離判斷_影片.py) only displayed them.

Usage:
    python similar_triangles_video.py path/to/video.mp4 --output output_video.mp4
Press 'q' to quit.
"""
import argparse

import cv2

# Known real width of the object (cm); adjust to the actual object size
KNOWN_WIDTH = 4.0

# Focal length in pixels (from camera calibration, or an empirical value)
FOCAL_LENGTH = 700


def calculate_distance(known_width, focal_length, perceived_width):
    """
    Estimate the distance to an object.

    :param known_width: real width of the object
    :param focal_length: focal length of the camera (pixels)
    :param perceived_width: width of the object measured in the image (pixels)
    :return: estimated distance (same unit as known_width)
    """
    if perceived_width == 0:
        return float('inf')  # object not detected -> infinite distance
    return (known_width * focal_length) / perceived_width


def process_video(video_path, known_width, focal_length, output_path=None):
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Optional VideoWriter for the annotated result
    out = None
    if output_path:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # keep the original FPS
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # codec
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Resizable display window
    window_name = "Object Detection and Distance Estimation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)  # initial window size

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Canny edge detection to find the object edges
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort by contour area and keep the three largest
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

        for contour in contours:
            # Bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Estimated distance of the object
            distance = calculate_distance(known_width, focal_length, w)

            # Draw the box and the distance label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write the annotated frame to the output video
        if out is not None:
            out.write(frame)

        # Fit the frame to the current window size
        current_width = cv2.getWindowImageRect(window_name)[2]
        current_height = cv2.getWindowImageRect(window_name)[3]
        resized_frame = cv2.resize(frame, (current_width, current_height))

        cv2.imshow(window_name, resized_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close windows
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monocular distance estimation (similar triangles) on a video file.")
    parser.add_argument("video", help="path to the input video")
    parser.add_argument("--output", default=None,
                        help="optional path for the annotated output video (e.g. output_video.mp4)")
    parser.add_argument("--known-width", type=float, default=KNOWN_WIDTH,
                        help=f"real width of the target object in cm (default: {KNOWN_WIDTH})")
    parser.add_argument("--focal-length", type=float, default=FOCAL_LENGTH,
                        help=f"camera focal length in pixels (default: {FOCAL_LENGTH})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(args.video, args.known_width, args.focal_length, args.output)
