
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
import matplotlib.pyplot as plt
import os

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

        cropped_frame = cv2.medianBlur(cropped_frame, ksize=3)
        
        # Sum the brightness values vertically (across the y-axis)
        vertical_sum = np.sum(cropped_frame, axis=0)

        # Normalize the summed values to the range [0, 255]
        norm_sum = np.interp(vertical_sum, (vertical_sum.min(), vertical_sum.max()), (0, 255)).astype(np.uint8)
            
        med = cv2.medianBlur(norm_sum, ksize=3)

        # morphological opening to knock out any little islands
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        opened = cv2.morphologyEx(med, cv2.MORPH_OPEN, kernel)

        # light closing to solidify image
        clean = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

        # Filter out all pixels below intensity 150
        clean[clean < 150] = 0

        # Add the normalized line to the timelapse image
        timelapse_image.append(clean)

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
    cv2.imwrite(filename, timelapse_array)  
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

def perform_turnaround_estimation(motion_profile_file_path, centerline_csv_path, x_offset = 0, y_offset = 0):
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
        _ = next(csvreader)  # Skip the first row, which is likely the header
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

        x_coords = [point[2] - x_offset for point in wave_data]  # Extract the x-coordinates (third value in each tuple)
        y_coords = [point[1] - y_offset for point in wave_data]  # Extract the y-coordinates (second value in each tuple)
    
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

    # Ensure the Output folder exists
    output_folder = os.path.join(os.getcwd(), "Output")
    os.makedirs(output_folder, exist_ok=True)

    # Construct the PDF file path in the Output folder
    output_pdf_name = os.path.basename(centerline_csv_path).replace(".csv", "_turnaround_estimation.pdf")
    output_pdf_path = os.path.join(output_folder, output_pdf_name)

    # Save the resulting overlay as a PDF
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

def denoise_frame_saltpep(frame):
    # Check if frame is already grayscale
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame
    
    # Apply median filter
    median_filtered = cv2.medianBlur(gray_frame, 11)
    
    # Apply bilateral filter (optional)
    bilateral_filtered = cv2.bilateralFilter(median_filtered, 9, 50, 50)
    
    gaussian = cv2.GaussianBlur(bilateral_filtered, (0, 0), 5)

    # Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(gaussian, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)
    # adaptive_thresh = gaussian
    
    # Morphological operations to connect blobs
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Dilation followed by erosion
    dilated = cv2.dilate(closing, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    return eroded

def preprocess_frame(frame, preprocessing_vals, advanced):
    """_summary_

    Args:
        frame (frame): frame to preprocess
        preprocessing_vals (dict): dictionary containing the values
        advanced (bool): whether to apply advanced preprocessing

    Returns:
        _type_: _description_
    """    
    # print("Preprocessing...")
    # Initialize a variable to store the modified frame
    modified_frame = frame.copy()

    # Apply improved smooth

    if preprocessing_vals["Smoothness"] != 0 and advanced:
        modified_frame = improve_smoothing(modified_frame, preprocessing_vals["Smoothness"]/50+.1)

    # Apply sharpening
    if preprocessing_vals["Blur/Sharpness"] != 0:
        modified_frame = sharpen_frame(modified_frame, preprocessing_vals["Blur/Sharpness"])

    # Apply contrast enhancement
    if preprocessing_vals["Contrast"] > 0:
        modified_frame = enhance_contrast(modified_frame, preprocessing_vals["Contrast"])
    
    # Apply brightness adjustment
    if preprocessing_vals["Brightness"] != 0:
        modified_frame = adjust_gamma(modified_frame, preprocessing_vals["Brightness"])

    if preprocessing_vals["Binarize"]  and advanced:
        modified_frame = improve_binarization(modified_frame)

    if preprocessing_vals["Denoise SP"]  and advanced:
        modified_frame = denoise_frame_saltpep(modified_frame)

    # modified_frame = cv2.adaptiveThreshold(modified_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # intermediate_frame_check("Preprocessing done", modified_frame)
    # print("Preprocessing done")

    return modified_frame

def enhance_contrast(frame, strength=50):
    # Define parameters for contrast enhancement
    clip_limit = 3.0
    tile_grid_size = (8, 8)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_frame = clahe.apply(frame)
    
    # Adjust contrast strength
    enhanced_frame = cv2.addWeighted(frame, 1 + strength / 2, enhanced_frame, 0.0, 0.0)
    
    return enhanced_frame

def sharpen_frame(frame, strength=1.0):
    if strength > 0:
        # Sharpening
        scaled_strength = strength / 100
        kernel = np.array([[0, -0.2, 0],
                           [-0.2, 2 + 3 * scaled_strength, -0.2],
                           [0, -0.2, 0]])
    else:
        # Blurring
        scaled_strength = abs(strength) / 10
        kernel_size = int(1 + 2 * scaled_strength)
        if kernel_size % 2 == 0:  # Ensure the kernel size is odd
            kernel_size += 1
        blurred = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        return blurred

    # Apply the kernel to the image
    result = cv2.filter2D(frame, -1, kernel)
    return result

def adjust_gamma(frame, gamma=50.0):
    # Apply gamma correction
    if gamma > 0:
        gamma=1-gamma/100
    elif gamma < 0:
        gamma=1-gamma/10
    gamma_corrected = np.array(255 * (frame / 255) ** gamma, dtype='uint8')
    return gamma_corrected

def improve_binarization(frame):    
    """
    Enhances the binarization of a grayscale image using various image processing techniques. This function applies
    CLAHE for contrast enhancement, background subtraction to highlight foreground objects, morphological operations
    to refine the image, and edge detection to further define object boundaries.

    Steps:
        1. Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to boost the contrast of the image.
        2. Perform background subtraction using a median blur to isolate foreground features.
        3. Apply morphological closing to close small holes within the foreground objects.
        4. Detect edges using the Canny algorithm, and dilate these edges to enhance their visibility.
        5. Optionally adjust the edge thickness with additional morphological operations like dilation or erosion
           depending on specific requirements (commented out in the code but can be adjusted as needed).

    Note:
        - This function is designed to work with grayscale images and expects a single-channel input.
        - Adjustments to parameters like CLAHE limits, kernel sizes for morphological operations, and Canny thresholds
          may be necessary depending on the specific characteristics of the input image.


    Args:
        frame (np.array): A single-channel (grayscale) image on which to perform binarization improvement.

    Returns:
        np.array: The processed image with enhanced binarization and clearer object definitions.

    """

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # boosts contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
    equalized = clahe.apply(frame)
    
    # Perform Background Subtraction
    # (Assuming a relatively uniform background)
    background = cv2.medianBlur(equalized, 13)
    subtracted = cv2.subtract(equalized, background)
    
    # Use morphological closing to close small holes inside the foreground
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(subtracted, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Use Canny edge detector to find edges and use it as a mask
    edges = cv2.Canny(closing, 30, 140)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    result = cv2.bitwise_or(closing, edges_dilated)

    # if edges too fine after result
    #result = cv2.dilate(result, kernel, iterations=1)
    # if edges too thick after result
    kernel = np.ones((3, 3), np.uint8)
    result = cv2.erode(result, kernel, iterations=1)

    return result

def improve_smoothing(frame, strength=0.9):
    """
    Enhances the input frame by applying Non-Local Means (NLM) denoising followed by a high-pass filter.
    
    NLM averages pixel intensity s.t. similar patches of the image (even far apart) contribute more to the average
    

    Args:
    frame (numpy.ndarray): The input image in grayscale.

    Returns:
    numpy.ndarray: The processed image with improved smoothing and enhanced details.
    """
    noise_level = np.std(frame)
    denoised = cv2.fastNlMeansDenoising(frame, None, h=noise_level*strength, templateWindowSize=7, searchWindowSize=25)
    
    # highlights cental pixel and reduces neighboring pixels
    # passes high frequencies and attenuates low frequencies
    # this kernel represents a discrete ver of the laplacian operator, approximating 2nd order derivative of image
    laplacian_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])

    # Apply the high-pass filter using convolution
    high_pass = cv2.filter2D(denoised, -1, laplacian_kernel)
    return high_pass
