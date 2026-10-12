"""Monocular distance estimation from a live webcam using similar triangles.

Every contour found by Canny edge detection is enclosed in a bounding box and
its distance is estimated with the pinhole-camera relation

    D = (W * F) / P

where W is the known real-world width of the object (cm), F is the focal
length (pixels) and P is the perceived width of the bounding box (pixels).

This is the first prototype of the project.
Original file: 距離判斷_單眼.py

Usage:
    python similar_triangles_webcam.py --camera 0 --known-width 20 --focal-length 600
Press 'q' to quit.
"""
import argparse

import cv2

# Known width of the object in centimeters
KNOWN_WIDTH = 20.0  # Adjust based on the actual object's width

# Focal length (calibrated or estimated)
FOCAL_LENGTH = 600  # Adjust based on the camera calibration results


def calculate_distance(known_width, focal_length, perceived_width):
    """
    Calculate the distance to the object based on its perceived width in the image.

    :param known_width: The actual width of the object
    :param focal_length: The camera's focal length
    :param perceived_width: The width of the object in the image
    :return: The calculated distance
    """
    if perceived_width == 0:
        return float('inf')  # Return infinity if the object is not detected
    return (known_width * focal_length) / perceived_width


def run(camera_index, known_width, focal_length):
    # Open the camera
    cap = cv2.VideoCapture(camera_index)

    while True:
        # Read the frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use Canny edge detection to find object edges
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate over the contours (each one is treated as a candidate obstacle)
        for contour in contours:
            # Calculate the bounding box for the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate the distance to the object
            distance = calculate_distance(known_width, focal_length, w)

            # Draw the bounding box and distance text on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame with detected objects and distance
        cv2.imshow("Obstacle Detection", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all windows
    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Live monocular distance estimation (similar triangles) from a webcam.")
    parser.add_argument("--camera", type=int, default=0,
                        help="webcam index passed to cv2.VideoCapture (default: 0)")
    parser.add_argument("--known-width", type=float, default=KNOWN_WIDTH,
                        help=f"real width of the target object in cm (default: {KNOWN_WIDTH})")
    parser.add_argument("--focal-length", type=float, default=FOCAL_LENGTH,
                        help=f"camera focal length in pixels (default: {FOCAL_LENGTH})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.camera, args.known_width, args.focal_length)
