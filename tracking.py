import cv2
import numpy as np
import pandas as pd
import screeninfo
from scipy.interpolate import splprep, splev
from PIL import Image, ImageTk, ImageSequence

from enums import *
from exceptions import error_popup, warning_popup

def scale_frame(frame, scale_factor=0.9):
    """Sometimes resolution of video is larger than resolution of monitor this software is running on.
    This function scales the resolution of the frame so the entire frame can be seen for selecting markers or just general analysis.
    Scaling preseves aspect ratio, so it determines which dimension is most cut off from viewing (width or height),
    and determines caling ratio for the other dimension based on that.

    Distances of objects being tracked are scaled back up when recording data, so movement of tracked objects are recorded in the original resolution.

    Args:
        frame (numpy.ndarray): indv frame of video being tracked that will be scaled and returned
        scale_factor (float, optional): fraction of monitor resolution to scale image. Defaults to 0.9.

    Returns:
        scaled_frame (numpy.ndarray): scaled version of frame that was passed in
        min_scale_factor (float): Determing scale factor, used to scale values back up before recording data.
    """    
    monitor = screeninfo.get_monitors()[0] # get primary monitor resolution

    # get indv scale factors for width and height
    scale_factor_height = scale_factor * (monitor.height / frame.shape[0])
    scale_factor_width = scale_factor * (monitor.width / frame.shape[1])

    min_scale_factor = min(scale_factor_width, scale_factor_height)

    # resize based on scale factors
    scaled_frame = cv2.resize(frame, (int(frame.shape[1] * min_scale_factor), int(frame.shape[0] * min_scale_factor)))
    return scaled_frame, min_scale_factor

def find_furthest_left_average_y(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Threshold the image to binarize it (convert to black and white)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Find contours in the image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    leftmost_points = []

    for contour in contours:
        # Find the leftmost point in each contour
        leftmost_point = tuple(contour[contour[:, :, 0].argmin()][0])
        leftmost_points.append(leftmost_point)

    # Calculate the average y-value of the leftmost points
    if leftmost_points:
        avg_y = np.mean([point[1] for point in leftmost_points])
        return avg_y
    else:
        return None

def detect_edges(frame, smoothing_factor=500):
    if frame.dtype != np.uint8:
        # Normalize if necessary
        if np.issubdtype(frame.dtype, np.floating):
            frame = (frame * 255).astype(np.uint8)
        elif frame.dtype == np.uint16:
            frame = (frame / 256).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)

    # Convert to grayscale if needed
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(gray_frame)

    # Apply Gaussian Blurring to reduce noise
    blurred_image = cv2.GaussianBlur(contrast_enhanced, (0, 5), 5)

    # Calculate the mean pixel intensity
    mean_intensity = np.mean(blurred_image)

    # Apply thresholding using the mean intensity as the threshold value
    _, thresholded_image = cv2.threshold(blurred_image, mean_intensity * 1.15, 255, cv2.THRESH_BINARY)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(thresholded_image, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Smooth the contours
    smoothed_contours_img = np.zeros_like(edges)

    for contour in contours:
        contour = contour.reshape(-1, 2)

        if len(contour) > 3:
            try:
                # Spline approximation
                tck, _ = splprep([contour[:, 0], contour[:, 1]], s=smoothing_factor)
                x_new, y_new = splev(np.linspace(0, 1, 1000), tck)
                smooth_contour = np.vstack((x_new, y_new)).T
                smooth_contour = np.round(smooth_contour).astype(int)

                # Draw the smooth contour on the smoothed_contours_img
                for i in range(len(smooth_contour) - 1):
                    cv2.line(smoothed_contours_img, tuple(smooth_contour[i]), tuple(smooth_contour[i + 1]), 255, 1)
            except Exception as e:
                print(f"Error fitting spline to contour: {e}")

    return smoothed_contours_img

def analyze_edges(frame):
    # Get the edges from the core function
    edges = detect_edges(frame)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the leftmost points
    leftmost_points = []

    # Buffer distance to ignore points that are too close to previously detected ones
    buffer_distance = 100

    # List to store all potential leftmost points before filtering
    all_leftmost_points = []

    for contour in contours:
        # Get the leftmost point of the contour
        leftmost_point = tuple(contour[contour[:, :, 0].argmin()][0])
        all_leftmost_points.append(leftmost_point)

    # Sort points by their x-coordinate to ensure left-to-right processing
    all_leftmost_points.sort(key=lambda pt: pt[0])

    # Filter points to keep only those that are sufficiently far from the last detected one
    for point in all_leftmost_points:
        if not leftmost_points or abs(point[0] - leftmost_points[-1][0]) > buffer_distance:
            leftmost_points.append(point)
    print(leftmost_points)
    # Return the list of filtered leftmost points
    return leftmost_points

def display_edges(frame):

    # Normalize the frame to the correct range and type if needed
    if frame.dtype != np.uint8:
        # Normalize to 0-255 range if it's a floating point type
        if np.issubdtype(frame.dtype, np.floating):
            frame = (frame * 255).astype(np.uint8)
        # Convert 16-bit to 8-bit by normalizing
        elif frame.dtype == np.uint16:
            frame = (frame / 256).astype(np.uint8)
        else:
            frame = frame.astype(np.uint8)  # Fallback to direct conversion


    smoothed_edges = detect_edges(frame)

    # Call analyze_edges to get the leftmost points
    leftmost_points = analyze_edges(frame)

    # Convert smoothed edges to BGR for display
    edges_colored = cv2.cvtColor(smoothed_edges, cv2.COLOR_GRAY2BGR)

    # Convert frame to BGR if it's a single channel (grayscale)
    if len(frame.shape) == 2:  # Grayscale image
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # Ensure both images are the same size
    if frame.shape[:2] != edges_colored.shape[:2]:
        edges_colored_resized = cv2.resize(edges_colored, (frame.shape[1], frame.shape[0]))
    else:
        edges_colored_resized = edges_colored

    # Check that edges_colored_resized is also uint8
    if edges_colored_resized.dtype != np.uint8:
        edges_colored_resized = edges_colored_resized.astype(np.uint8)

    # Normalize the frame to the range 0-255
    frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)

    # Blend the smoothed edges with the original frame
    edge_blended_frame = cv2.addWeighted(frame, 0.7, edges_colored_resized, 0.3, 10)  # Try different gamma values

    # Draw vertical lines at the leftmost points
    for point in leftmost_points:
        if 0 <= point[0] < frame.shape[1]:  # Ensure the point is within the image width
            cv2.line(edge_blended_frame, (point[0], 0), (point[0], frame.shape[0]), (0, 0, 255), 2)

    # Convert the result to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(edge_blended_frame, cv2.COLOR_BGR2RGB))

    return pil_image