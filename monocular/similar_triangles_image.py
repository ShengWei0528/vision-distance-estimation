"""Monocular distance estimation on a still image using similar triangles.

Pipeline: grayscale -> Canny edges -> contours -> bounding boxes.
For every bounding box the distance is estimated with

    D = (W * F) / P

where W is the known real-world width of the object (cm), F is the focal
length (pixels) and P is the perceived width of the box (pixels). The three
contours with the smallest estimated distance are boxed and labelled.

Original file: 距離判斷_圖片2.py (v2).
In v1 (距離判斷_圖片.py) the three contours with the largest *area* were
labelled instead of the three closest ones.

Usage:
    python similar_triangles_image.py path/to/image.jpg --known-width 20 --focal-length 130
The result window can be resized freely. Press 'q' to quit.
"""
import argparse
import sys

import cv2

# Known real width of the object (cm); adjust to the actual object size
KNOWN_WIDTH = 20.0

# Focal length in pixels (from camera calibration, or an empirical value)
FOCAL_LENGTH = 130


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


def annotate_closest_objects(image, known_width, focal_length, top_k=3):
    """Draw boxes and distance labels for the top_k closest contours (in place)."""
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny edge detection to find the object edges
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Store each contour together with its estimated distance
    distances_and_contours = []

    # Compute the distance for every contour
    for contour in contours:
        # Bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)

        # Estimated distance of the object
        distance = calculate_distance(known_width, focal_length, w)

        distances_and_contours.append((distance, contour, (x, y, w, h)))

    # Sort by distance and keep the closest top_k objects
    closest_objects = sorted(distances_and_contours, key=lambda x: x[0])[:top_k]

    # Box and label the closest objects
    for distance, contour, (x, y, w, h) in closest_objects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"Distance: {distance:.2f} cm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image


def show_resizable(image, window_name="Object Detection and Distance Estimation"):
    """Show the image in a freely resizable window until 'q' is pressed."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # window can be resized
    cv2.resizeWindow(window_name, 800, 600)  # initial window size

    while True:
        # Current window size
        current_width = cv2.getWindowImageRect(window_name)[2]
        current_height = cv2.getWindowImageRect(window_name)[3]

        # Resize the image to fit the window
        resized_image = cv2.resize(image, (current_width, current_height))

        cv2.imshow(window_name, resized_image)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monocular distance estimation (similar triangles) on a single image.")
    parser.add_argument("image", help="path to the input image")
    parser.add_argument("--known-width", type=float, default=KNOWN_WIDTH,
                        help=f"real width of the target object in cm (default: {KNOWN_WIDTH})")
    parser.add_argument("--focal-length", type=float, default=FOCAL_LENGTH,
                        help=f"camera focal length in pixels (default: {FOCAL_LENGTH})")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the image
    image = cv2.imread(args.image)
    if image is None:
        sys.exit("Error: could not find or open the image.")

    annotate_closest_objects(image, args.known_width, args.focal_length)
    show_resizable(image)


if __name__ == "__main__":
    main()
