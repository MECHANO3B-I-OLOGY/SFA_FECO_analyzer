import cv2
import numpy as np
import pandas as pd
import screeninfo
from scipy.interpolate import splprep, splev

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

def necking_point_y_axis(
        cap,
        frame_start,
        frame_end,
        percent_crop_top,
        percent_crop_bottom,
        binarize_intensity_thresh,
        frame_record_interval,
        frame_interval,
        time_units,
        file_mode,
        video_file_name,
        data_label
    ):
    """necking point detection loop on the x-axis
    Detects and records the necking point in each frame of a video. The necking point is defined as the shortest horizontal distance
    between two vertical edges within a specified region of interest. The function preprocesses the frames, performs edge detection,
    and identifies the left and right edges to calculate the necking point, highlighting it in red on the visual output.

    Args:
        cap (cv2.VideoCapture): Video capture object loaded with the video.
        frame_start (int): Frame number to start processing.
        frame_end (int): Frame number to end processing.
        percent_crop_top (float): Percentage of the frame's top side to exclude from processing.
        percent_crop_bottom (float): Percentage of the frame's bottom side to exclude from processing.
        binarize_intensity_thresh (int): Threshold for binarization of the frame to facilitate edge detection.
        frame_record_interval (int): Interval at which frames are processed.
        frame_interval (float): Real-world time interval between frames, used in time calculations.
        time_units (str): Units of time (e.g., 'seconds', 'minutes') for the output data.
        file_mode (FileMode): Specifies whether to overwrite or append data in the output file.
        video_file_name (str): Name of the video file being processed.
        data_label (str): Unique identifier for the data session.
    """
    y_interval = 50  # interval for how many blue line visuals to display
    frame_num = frame_start
    dist_data = {'1-Frame': [], f'1-Time({time_units})': [], '1-y at necking point (px)': [], '1-x necking distance (px)': [], '1-video_file_name': video_file_name, '1-detection_method': 'min_distance', '1-data_label': data_label}
    percent_crop_top *= 0.01
    percent_crop_bottom *= 0.01

    while True:  # read frame by frame until the end of the video
        ret, frame = cap.read()
        frame_num += frame_record_interval
        if frame_record_interval != 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        if not ret:
            break

        scaled_frame, scale_factor = scale_frame(frame)  # scale the frame
        gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)  # convert frame to gray
        _, binary_frame = cv2.threshold(gray_frame, binarize_intensity_thresh, 255, cv2.THRESH_BINARY)  # threshold to binarize image

        # error checking for appropriate binarization threshold
        if np.all(binary_frame == 255):
            msg = "Binarization threshold too low,\nfound no pixels below the threshold.\n\nPlease adjust the threshold (default is 120)"
            error_popup(msg)
        if np.all(binary_frame == 0):
            msg = "Binarization threshold too high,\nfound no pixels above the threshold.\n\nPlease adjust the threshold (default is 120)"

        edges = cv2.Canny(binary_frame, 0, 2)  # edge detection, nums are gradient thresholds

        y_samples = []
        x_distances = []
        x_line_values = []

        frame_draw = scaled_frame.copy()
        frame_draw[edges > 0] = [0, 255, 0]  # draw edges

        # remove x% of edges from consideration of detection
        vertical_pixels_top = 0
        vertical_pixels_bottom = scaled_frame.shape[0]
        if percent_crop_top != 0.:
            top_pixels_removed = int(percent_crop_top * scaled_frame.shape[0])
            vertical_pixels_top = max(0, top_pixels_removed)
        if percent_crop_bottom != 0.:
            bottom_pixels_removed = int(percent_crop_bottom * scaled_frame.shape[0])
            vertical_pixels_bottom = min(scaled_frame.shape[0], scaled_frame.shape[0] - bottom_pixels_removed)

        for y in range(vertical_pixels_top, vertical_pixels_bottom):
            edge_pixels = np.nonzero(edges[y, :])[0]  # find x coord of edge pixels in cur row

            if edge_pixels.size > 0:  # if edge pixels in cur row,
                dist = np.abs(edge_pixels[0] - edge_pixels[-1])  # find distance of left and right edges
                y_samples.append(y)
                x_line_values.append((edge_pixels[0], edge_pixels[-1]))
                x_distances.append(dist)

                if y % y_interval == 0:  # draw visualization lines at every y_interval pixels
                    # draw horizontal lines connecting edges for visualization
                    cv2.line(frame_draw, (edge_pixels[0], y), (edge_pixels[-1], y), (200, 0, 0), 1)

        # find index of smallest distance
        necking_distance = np.min(x_distances)
        necking_pt_indices = np.where(x_distances == necking_distance)[0]
        necking_pt_ind = int(np.median(necking_pt_indices))

        # record and save data using original resolution
        if frame_interval == 0:
            dist_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) / cap.get(5)))
        else:
            dist_data[f'1-Time({time_units})'].append(np.float16((frame_num - frame_start) * frame_interval))
        dist_data['1-Frame'].append(frame_num - frame_start)
        dist_data['1-y at necking point (px)'].append(int(y_samples[necking_pt_ind] / scale_factor))
        dist_data['1-x necking distance (px)'].append(int(necking_distance / scale_factor))

        cv2.line(frame_draw, (x_line_values[necking_pt_ind][0], y_samples[necking_pt_ind]), (x_line_values[necking_pt_ind][1], y_samples[necking_pt_ind]), (0, 0, 255), 2)

        cv2.imshow('Necking Point Visualization', frame_draw)

        if cv2.waitKey(1) == 27 or frame_end <= frame_num:
            break

        # record_data(file_mode, dist_data, "output/Necking_Point_Output.csv")
        cap.release()
        cv2.destroyAllWindows()

def detect_edges(frame, smoothing_factor=500):
    # Convert to grayscale if needed
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    # Convert to 8-bit image if not already
    if gray_frame.dtype != np.uint8:
        gray_frame = cv2.convertScaleAbs(gray_frame)

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
    smoothed_edges = detect_edges(frame)

    # Call analyze_edges to get the leftmost points
    leftmost_points = analyze_edges(frame)

    # Convert smoothed edges to BGR for display
    edges_colored = cv2.cvtColor(smoothed_edges, cv2.COLOR_GRAY2BGR)

    # Blend the smoothed edges with the original frame
    edge_blended_frame = cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)

    # Draw vertical lines at the leftmost points
    for point in leftmost_points:
        cv2.line(edge_blended_frame, (point[0], 0), (point[0], frame.shape[0]), (0, 0, 255), 2)

    return edge_blended_frame