
import cv2
import csv
import pprint
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.optimize import curve_fit
from scipy import stats
from PIL import Image, ImageTk, ImageSequence
from pprint import pprint
# temp?
import matplotlib.pyplot as plt

from enums import *
from exceptions import error_popup, warning_popup

def scale_frame(frame, scale_factor=0.9):
    """Scales a PIL image based on monitor resolution and provided scale factor.
    
    Args:
        frame (PIL.Image): Frame of the video (PIL image) to be scaled.
        scale_factor (float, optional): Fraction of monitor resolution to scale the image. Defaults to 0.9.

    Returns:
        scaled_frame (PIL.Image): Scaled version of the PIL image.
    """    

    # Get width and height of the original PIL image
    width, height = frame.size

    # Calculate scale factors for height and width
    scale_factor_height = scale_factor * height
    scale_factor_width = scale_factor * width

    # Resize the image using the minimum scale factor
    new_width = int(scale_factor_width)
    new_height = int(scale_factor_height)

    # Resize the image using PIL's resize method
    scaled_frame = frame.resize((new_width, new_height))

    return scaled_frame

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

def generate_motion_profile(file_path, y_start, y_end, filename):
    """
    Processes the TIFF video by performing vertical summing and normalization,
    then builds the final "timelapse" image. Can display progress after each frame.

    Args:
        file_path (str): Path to the input TIFF file.
        y_start (int): Start y-coordinate of the RoI.
        y_end (int): End y-coordinate of the RoI.
        filename (str): Output filename for the final processed image.
    """
    visualize = False
    i = 1
    # Open the TIFF file
    tiff = Image.open(file_path)
    num_frames = tiff.n_frames  # Number of frames in the TIFF file

    # Initialize an empty array to store the final "timelapse" image
    timelapse_image = []

    plt.ion()  # Turn on interactive plotting

    for frame_index in range(num_frames):
        # Get the current frame
        tiff.seek(frame_index)
        frame = np.array(tiff)  # Convert the frame to a numpy array

        # Crop the frame to the Region of Interest (RoI)
        cropped_frame = frame[y_start:y_end, :]

        # Sum the brightness values vertically (across the y-axis)
        vertical_sum = np.sum(cropped_frame, axis=0)

        # Normalize the summed values to the range [0, 255]
        norm_sum = np.interp(vertical_sum, (vertical_sum.min(), vertical_sum.max()), (0, 255))

        # Filter out all pixels below intensity 150
        norm_sum[norm_sum < 150] = 0

        # Add the normalized line to the timelapse image
        timelapse_image.append(norm_sum)

        # Convert the timelapse array to a numpy array (for visualization)
        timelapse_array = np.array(timelapse_image)

        if visualize and i % 5 == 0:
            # Visualization: display progress after each frame
            plt.figure(figsize=(10, 5))
            plt.imshow(timelapse_array, cmap='gray', aspect='auto')
            plt.title(f"Timelapse Progress - Frame {frame_index + 1}/{num_frames}")
            plt.colorbar()
            plt.show()

            # Wait for a key press before continuing to the next frame
            plt.waitforbuttonpress()
            plt.close()
        
        i += 1

    # Once all frames are processed, save the final timelapse image
    final_image = Image.fromarray(timelapse_array.astype(np.uint8))
    final_image.save(filename)
    print(f"Timelapse saved to {filename}")

def analyze_and_append_waves(image, 
                             wave_threshold=0, 
                             min_wave_gap=15, 
                             horizontal_proximity_threshold=15, 
                             vertical_proximity_threshold=1, 
                             max_missing_rows=2,
                             min_points_per_wave=50):
    """
    Analyze waves in the image, calculate the center of mass for each wave, 
    and append them to the correct wave line based on proximity to previously detected waves.
    
    Args:
    - image: 2D array representing the RoI of the motion profile.
    - wave_threshold: Intensity threshold for wave detection.
    - min_wave_gap: Minimum distance to separate different waves within a row.
    - horizontal_proximity_threshold: Maximum allowable horizontal distance to append a new center point to an existing wave.
    - vertical_proximity_threshold: Maximum allowable vertical distance between rows for a wave to be considered continuous.
    - max_missing_rows: Maximum number of consecutive rows where a wave can be missing before terminating it.
    - min_points_per_wave: Minimum number of points for a wave line to be considered valid.
    
    Returns:
    - wave_lines: List of wave lines, each being a list of (y, x_center) points.
    """
    height, width = image.shape
    wave_lines = []  # Initialize list to store wave lines
    wave_missing_counts = []  # Track how many rows each wave has been missing for

    # Loop through each row
    for y in range(height):
        row = image[y, :]
        
        # Detect wave positions in the row (above threshold)
        wave_positions = np.where(row > wave_threshold)[0]

        if len(wave_positions) > 0:
            # Cluster wave points based on proximity (min_wave_gap)
            waves = []
            current_wave = [wave_positions[0]]

            for i in range(1, len(wave_positions)):
                if wave_positions[i] - wave_positions[i - 1] > min_wave_gap:
                    waves.append(current_wave)
                    current_wave = [wave_positions[i]]
                else:
                    current_wave.append(wave_positions[i])

            # Append the last wave
            waves.append(current_wave)

            # Calculate center of mass for each wave
            center_of_mass_points = []
            for wave in waves:
                wave_intensities = row[wave]
                total_intensity = np.sum(wave_intensities)

                if total_intensity > 0:
                    positions = np.array(wave)
                    center_of_mass = np.sum(positions * wave_intensities) / total_intensity
                    center_of_mass_points.append((y, center_of_mass))

            # Append center of mass points to the closest wave line
            for (y, x_center) in center_of_mass_points:
                added = False

                # Compare to existing wave lines
                for idx, wave_line in enumerate(wave_lines):
                    last_y, last_x_center = wave_line[-1]

                    # Check both horizontal and vertical proximity
                    if abs(x_center - last_x_center) < horizontal_proximity_threshold and abs(y - last_y) <= vertical_proximity_threshold:
                        wave_line.append((y, x_center))  # Append to existing wave line
                        wave_missing_counts[idx] = 0  # Reset the missing row count for this wave
                        added = True
                        break

                # If no match is found, start a new wave line
                if not added:
                    wave_lines.append([(y, x_center)])
                    wave_missing_counts.append(0)  # Initialize missing row count for the new wave line

        else:
            # If no wave positions were found, increment the missing row count for each active wave line
            for i in range(len(wave_missing_counts)):
                wave_missing_counts[i] += 1

        # Remove wave lines that have been missing for too many rows
        wave_lines = [wave_line for idx, wave_line in enumerate(wave_lines) if wave_missing_counts[idx] < max_missing_rows]
        wave_missing_counts = [count for count in wave_missing_counts if count < max_missing_rows]

    # Remove wave lines that have fewer than the minimum required points
    wave_lines = [wave_line for wave_line in wave_lines if len(wave_line) >= min_points_per_wave]
    # pprint(wave_lines)
    # Remove data too close to the edge
    edge_threshold = 5
    wave_lines = [[(y, x) for (y, x) in wave_line if edge_threshold <= x <= width - edge_threshold and edge_threshold <= y <= height - edge_threshold] for wave_line in wave_lines]

    return wave_lines

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

def perform_turnaround_estimation(motion_profile_file_path, centerline_csv_path, y_offset = 0):
    """
    Estimate the turnaround points for each wave by performing leftward and rightward 
    linear fits, calculating their intersection points, and plotting the results.
    
    Args:
    - motion_profile_file_path: Path to the motion profile image file (e.g., .tiff, .png).
    - centerline_csv_path: Path to the CSV file that stores wave centerline coordinates.
    - y_offset: offset in y (frames), results from cropping image
    
    Returns:
    - Average of the estimated intersection points' y location, taking offset into account
    """

    # Load the motion profile image
    motion_profile_image = plt.imread(motion_profile_file_path)
    
    # Read the wave lines (centerlines) from the CSV
    wave_lines = []
    with open(centerline_csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        header = next(csvreader)  # Skip the first row, which is likely the header
        for row in csvreader:
            wave_lines.append([(int(row[0]), int(row[1]), float(row[2]))])

    # Dictionary to store intersection points for each wave line
    intersection_points = {}

    # Create a figure for the overlay
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size for clarity
    ax.imshow(motion_profile_image, cmap='gray')

    # Replot the wave lines (limit for performance testing)
    colors = plt.cm.rainbow(np.linspace(0, 1, min(10, len(wave_lines))))  # Limit to 10 lines for testing

    unique_wave_numbers = sorted(set(point[0] for wave_line in wave_lines for point in wave_line))
    # print(1, range(unique_wave_numbers[-1]))

    for wavenum in range(1, unique_wave_numbers[-1] + 1): 
        # Find the points in wave_lines corresponding to the current wave number
        wave_data = [point for wave_line in wave_lines for point in wave_line if point[0] == wavenum]

        # pprint(wave_data)

        x_coords = [point[2] for point in wave_data]  # Extract the x-coordinates (third value in each tuple)
        y_coords = [point[1] for point in wave_data]  # Extract the y-coordinates (second value in each tuple)
    
        # Appending leftward points
        pointNum = 5

        # Initialize lists for leftward and rightward motion points
        leftward_points = [(x_coords[pointNum - 2], y_coords[pointNum - 2])]

        while pointNum < len(x_coords) and (leftward_points[-1][0]- x_coords[pointNum]) >= 5:
            leftward_points.append((x_coords[pointNum], y_coords[pointNum]))
            pointNum += 1

        # Appending rightward points
        pointNum = len(x_coords) - 5
        rightward_points = [(x_coords[len(x_coords) - 4], y_coords[len(x_coords) - 4])]
        while pointNum >= 0 and (rightward_points[-1][0] - x_coords[pointNum]) >= 5:
            rightward_points.append((x_coords[pointNum], y_coords[pointNum]))
            pointNum -= 1
        # print(wavenum)

        # Separate Y and X coordinates for leftward and rightward points
        left_x_coords = [point[0] for point in leftward_points]
        left_y_coords = [point[1] for point in leftward_points]
        right_x_coords = [point[0] for point in rightward_points]
        right_y_coords = [point[1] for point in rightward_points]

        # Perform linear regression on leftward and rightward points
        left_slope, left_intercept = None, None
        right_slope, right_intercept = None, None
        
        if len(left_x_coords) > 1:
            left_slope, left_intercept, _, _, _ = stats.linregress(left_x_coords, left_y_coords)
        
        if len(right_x_coords) > 1:
            right_slope, right_intercept, _, _, _ = stats.linregress(right_x_coords, right_y_coords)

        # Find intersection of the two lines: Solve for x in y = mx + b
        intersection_point = None
        if left_slope is not None and right_slope is not None:
            # x-coordinate of intersection
            intersection_x = (right_intercept - left_intercept) / (left_slope - right_slope)
            # y-coordinate of intersection
            intersection_y = left_slope * intersection_x + left_intercept
            intersection_point = (intersection_x, intersection_y)
            intersection_points[wavenum] = intersection_point
            # print(intersection_point)

        # Plot each wave line
        ax.plot(x_coords, y_coords, color=colors[wavenum])
        # Plot leftward and rightward linear approximations
        if left_slope is not None:
            left_fit_x = np.array(left_x_coords)
            left_fit_y = left_slope * left_fit_x + left_intercept
            ax.plot(left_fit_x, left_fit_y, 'b--')

        if right_slope is not None:
            right_fit_x = np.array(right_x_coords)
            right_fit_y = right_slope * right_fit_x + right_intercept
            ax.plot(right_fit_x, right_fit_y, 'r--')

        # Plot the intersection point
        if intersection_point:
            ax.plot(intersection_point[0], intersection_point[1], 'go')

    # Reapply title, labels, and limit the legend for clarity
    ax.set_title("Turnaround Estimation")
    ax.set_xlabel("X (columns)")
    ax.set_ylabel("Y (rows)")
    ax.legend(loc='upper left', fontsize='small')

    # Save the resulting overlay as a PDF
    output_pdf_path = centerline_csv_path.replace(".csv", "_turnaround_estimation.pdf")
    plt.savefig(output_pdf_path, format='pdf')

    # Display the plot
    plt.show()

    # Extract all y-values from the dictionary
    y_values = [point[1] for point in intersection_points.values()]

    # Calculate the average y-value
    # print(y_offset)
    estimated_turnaround = sum(y_values) / len(y_values) + y_offset

    # print(estimated_turnaround)

    return estimated_turnaround
