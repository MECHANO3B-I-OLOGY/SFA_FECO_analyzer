import cv2
import csv
import pprint
import numpy as np
import pandas as pd
import screeninfo
from scipy.interpolate import splprep, splev
from scipy.optimize import curve_fit
from PIL import Image, ImageTk, ImageSequence

# temp?
import matplotlib.pyplot as plt

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

def rough_approx_poi(frame):
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
    # print(leftmost_points)
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

    # Call rough_approx_poi to get the leftmost points
    leftmost_points = rough_approx_poi(frame)

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

def save_data_as_csv(data, filename):
    """
    Save data to a CSV file.

    :param data: List of dictionaries, each containing data for a frame
    :param filename: Desired filename (without extension)
    """
    # Get the field names from the first dictionary in the list
    fieldnames = data[0].keys()

    # Write data to CSV file
    with open(f"{filename}.csv", mode="w", newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()  # Write column headers
        for row in data:
            writer.writerow(row)

    print(f"Data saved to {filename}.csv")

def fine_approx_poi(tiff_file_path, y_start, y_end, filename=None, visualize=False):
    """
    Fine approximation of the points of interest using Y crop coordinates.
    
    :param tiff_file_path: Path to the TIFF file.
    :param y_start: Y-coordinate start of the crop area.
    :param y_end: Y-coordinate end of the crop area.
    :param filename: Name of generated CSV file. Default is None.
    :param visualize: If True, display the frame and ROI for debugging purposes.
    """
    visualize = True 
    # visualize = False
    # Convert y_start and y_end to integers
    y_start = int(y_start)
    y_end = int(y_end)
    
    # Open the TIFF file
    tiff = Image.open(tiff_file_path)
    num_frames = tiff.n_frames  # Number of frames in the TIFF file

    # Prepare list to hold data for all frames
    data = []

    for frame_index in range(num_frames):
        # Select the current frame
        tiff.seek(frame_index)
        frame = np.array(tiff)

        # Crop the frame using the Y crop coordinates
        cropped_frame = frame[y_start:y_end, :]

        # Run rough_approx_poi to get leftmost points
        leftmost_points = rough_approx_poi(cropped_frame)

        # Approximate the point of interest using a bounding box
        for point in leftmost_points:
            x, y = point
            # Define the region of interest around the leftmost point
            box_size = 20  # Define the size of the box for collapsing intensity
            # Define new ROI coordinates
            x_start = x - int(.5 * box_size)   # Left edge starts at the detected x point
            x_end = x + int(1.5 * box_size)  # Extend to the right by 2 * box_size

            # Center the ROI vertically around y
            y_start_roi = max(0, y - box_size)  # Start box_size above the detected y point
            y_end_roi = min(cropped_frame.shape[0], y + box_size)  # End box_size below the detected y point

            # Skip ROI if it extends beyond the cropped frame dimensions
            if x_end > cropped_frame.shape[1] or y_end_roi > cropped_frame.shape[0] or x_start < 0:
                print(f"Skipping invalid or empty ROI at frame {frame_index + 1}, point ({x}, {y})")
                continue

            # Crop the region of interest
            roi = cropped_frame[y_start_roi:y_end_roi, x_start:x_end]

            # Normalize the ROI to enhance contrast
            roi = normalize_hist_equalization(roi)

            # Collapse intensities along X and Y axes
            collapsed_x = np.sum(roi, axis=0)
            collapsed_y = np.sum(roi, axis=1)

            y_values = np.arange(len(collapsed_y))
            y_center = find_center_of_intensity(y_values, collapsed_y)

            x_values = np.arange(len(collapsed_x))
            x_center = find_center_of_intensity(x_values, collapsed_x)

            # Visualize the frame and the ROI
            if visualize:
                plt.figure(figsize=(10, 5))

                # Display the full cropped frame
                plt.subplot(1, 2, 1)
                plt.imshow(cropped_frame, cmap='gray')
                plt.title(f"Frame {frame_index + 1} - Cropped")
                plt.scatter(x_center + x_start, y_center + y_start_roi, color='red', marker='x', label='Center of Intensity')  # Corrected scatter coordinates

                # Display the ROI
                plt.subplot(1, 2, 2)
                plt.imshow(roi, cmap='gray')
                plt.title(f"ROI for point ({x}, {y}) in Frame {frame_index + 1}")
                plt.scatter(x_center, y_center, color='red', marker='x', label='Center of Intensity')  # Corrected scatter in ROI coordinates

                # Add legends to the plots
                plt.subplot(1, 2, 1)
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.legend()

                plt.show()

            # Store the results in the data list
            data.append({
                "Frame": frame_index + 1,
                "X Center": x_center,
                "Y Center": y_center
            })

    # If the filename is not provided, ask the user
    if not filename:
        filename = input("Enter the desired filename for the CSV (without extension): ") + '.csv'

    # Save the data to CSV
    save_data_as_csv(data, filename)
    print(f"Data saved to {filename}")

def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

# Define your fitting function
def find_center_of_intensity(values, collapsed_values):
    initial_amplitude = np.max(collapsed_values)
    initial_mean = np.argmax(collapsed_values)
    initial_sigma = np.std(values)  # Use the standard deviation of the y-values as an initial guess

    print(initial_amplitude)

    try:
        popt_y, _ = curve_fit(
            gaussian,
            values,
            collapsed_values,
            p0=[initial_amplitude, initial_mean, initial_sigma],
            maxfev=2000
        )
        y_center = popt_y[1]
        if y_center > 300 or y_center < 0:
            raise RuntimeError
    except RuntimeError:
        print("Gaussian fitting failed; using average value as fallback.")
        y_center = np.mean(values)

    return y_center

def normalize_min_max(roi):
    """
    Normalize the pixel values of the ROI to the range [0, 1].
    
    :param roi: The region of interest (2D numpy array).
    :return: Normalized ROI.
    """
    roi_min = np.min(roi)
    roi_max = np.max(roi)
    normalized_roi = (roi - roi_min) / (roi_max - roi_min)
    return normalized_roi

def normalize_hist_equalization(roi):
    """
    Apply histogram equalization to enhance contrast.
    
    :param roi: The region of interest (2D numpy array).
    :return: Contrast-enhanced ROI.
    """
    
    if roi.dtype != np.uint8:
        roi = (255 * normalize_min_max(roi)).astype(np.uint8)
        
    equalized_roi = cv2.equalizeHist(roi)
    return equalized_roi