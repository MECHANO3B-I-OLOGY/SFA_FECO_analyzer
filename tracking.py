import cv2
import numpy as np
import pandas as pd
import screeninfo

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