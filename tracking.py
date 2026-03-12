import csv
import cv2
import pandas as pd
import pprint
import matplotlib.pyplot as plt
import numpy as np
import os

from deprecation import deprecated
from PIL import Image, ImageTk, ImageSequence, ImageFilter
from pprint import pprint 
from scipy import stats
from scipy.interpolate import splprep, splev, UnivariateSpline
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit, minimize
from scipy.signal import savgol_filter, medfilt, find_peaks
from sklearn.mixture import GaussianMixture

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

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def generate_motion_profile(
    input_file_path,
    output_file_name,
    slope,
    intercept,
    padding=3,
    noiseFloor=150,
    set_status=None
):
    """
    Vectorized motion profile generation.
    Samples along y = slope*x + intercept with vertical padding.
    """

    visualize = False

    if set_status:
        set_status("Generating Motion Profile: 0%")

    tiff = Image.open(input_file_path)
    num_frames = tiff.n_frames

    timelapse_image = []

    plt.ion()

    next_update = 5  # next 5% threshold

    for frame_index in range(num_frames):
        tiff.seek(frame_index)
        frame = np.array(tiff)

        height, width = frame.shape

        x_vals = np.arange(width)

        y_center = slope * x_vals + intercept
        y_center = np.rint(y_center).astype(int)

        offsets = np.arange(-padding, padding + 1)

        y_samples = y_center[None, :] + offsets[:, None]
        y_samples = np.clip(y_samples, 0, height - 1)

        sampled_values = frame[y_samples, x_vals]
        sampled_profile = sampled_values.mean(axis=0)

        if sampled_profile.max() > sampled_profile.min():
            norm_profile = np.interp(
                sampled_profile,
                (sampled_profile.min(), sampled_profile.max()),
                (0, 255)
            )
        else:
            norm_profile = np.zeros_like(sampled_profile)

        norm_profile[norm_profile < noiseFloor] = 0

        timelapse_image.append(norm_profile)

        # ---- Progress Reporting ----
        if set_status:
            percent = int(((frame_index + 1) / num_frames) * 100)

            if percent >= next_update:
                set_status(f"Generating Motion Profile: {next_update}%")
                next_update += 5

        # Optional visualization
        if visualize and frame_index % 5 == 0:
            timelapse_array = np.array(timelapse_image)
            plt.figure(figsize=(10, 5))
            plt.imshow(timelapse_array, cmap='gray', aspect='auto')
            plt.title(f"Timelapse Progress - Frame {frame_index + 1}/{num_frames}")
            plt.colorbar()
            plt.show()
            plt.waitforbuttonpress()
            plt.close()

    # Final output
    timelapse_array = np.array(timelapse_image)
    final_image = Image.fromarray(timelapse_array.astype(np.uint8))
    final_image.save(output_file_name)

    if set_status:
        set_status("Generating Motion Profile: 100%")

    print(f"Timelapse saved to {output_file_name}")

def fix_endpoints(xs_clean, xs_smooth, k_frac=0.06, min_k=3):
    """
    Repair endpoints of xs_smooth using local linear fits on xs_clean.
    - k_frac: fraction of total length to treat as 'end region'
    - min_k: minimum number of points in end region
    """
    n = len(xs_clean)
    k = max(min_k, int(np.ceil(n * k_frac)))
    if n <= 2*k:
        # Too short — return cleaned values (no smoothing) as safest fallback
        return xs_clean.copy()

    out = xs_smooth.copy()

    # Left endpoint: fit linear on first 2*k points of the cleaned signal
    left_fit_n = min(max(3, k), n//2)
    pL = np.polyfit(np.arange(left_fit_n), xs_clean[:left_fit_n], 1)
    left_lin = np.polyval(pL, np.arange(left_fit_n))

    # Right endpoint: fit linear on last 2*k points
    right_fit_n = left_fit_n
    pR = np.polyfit(np.arange(n-right_fit_n, n), xs_clean[-right_fit_n:], 1)
    right_x = np.arange(n-right_fit_n, n)
    right_lin = np.polyval(pR, right_x)

    # Replace first k and last k values with linear fits (or a blend)
    # Blend weights linearly from 1 (use linear) to 0 (use smooth) over k points
    for i in range(k):
        alpha = 1.0 - (i / float(k))  # alpha=1 at the very edge, 0 at boundary
        out[i] = alpha * left_lin[i] + (1 - alpha) * xs_smooth[i]
        j = n - 1 - i
        out[j] = alpha * right_lin[-(i+1)] + (1 - alpha) * xs_smooth[j]

    return out

def robust_smooth_1d(xs, smooth=True, base_window=21, polyorder=3):
    """
    Robust smoothing for 1D traces:
    - small median filter to remove spikes
    - MAD outlier replacement with local median
    - adaptive window smoothing (adjusts window by local slope)
    - fallback to smoothing spline if too few points
    """
    xs = np.asarray(xs, dtype=float)
    n = len(xs)
    if n == 0:
        return xs

    # 1) small median filter to suppress single-frame spikes
    k_med = 3 if n >= 3 else 1
    xs_med = medfilt(xs, kernel_size=k_med) if k_med > 1 else xs.copy()

    # 2) MAD-based outlier detection
    resid = xs - xs_med
    mad = np.median(np.abs(resid - np.median(resid)))
    if mad <= 0:
        mad = np.std(resid) + 1e-8
    threshold = 4.5 * mad
    outliers = np.abs(resid) > threshold
    if np.any(outliers):
        xs_clean = xs.copy()
        for i in np.where(outliers)[0]:
            lo = max(0, i - 2)
            hi = min(n, i + 3)
            xs_clean[i] = np.median(xs[lo:hi])
    else:
        xs_clean = xs_med

    if not smooth:
        return xs_clean

    # 3) Adaptive smoothing
    if n <= 5:
        try:
            spl = UnivariateSpline(np.arange(n), xs_clean, s=0.0, k=min(3, n-1))
            return spl(np.arange(n))
        except Exception:
            return xs_clean

    # --- Adaptive window logic ---
    dy = np.gradient(xs_clean)
    abs_dy = np.abs(dy)
    if np.max(abs_dy) == 0:
        abs_dy += 1e-8

    # Window shrinks where slope is high, expands where slope is flat
    min_w, max_w = 5, base_window
    local_windows = np.clip(max_w - (abs_dy / np.max(abs_dy)) * (max_w - min_w), min_w, max_w)
    local_windows = (2 * (local_windows // 2) + 1).astype(int)  # ensure odd

    xs_smooth = np.zeros_like(xs_clean)
    for i in range(n):
        w = int(local_windows[i])
        half = w // 2
        start = max(0, i - half)
        end = min(n, i + half + 1)
        xs_smooth[i] = np.mean(xs_clean[start:end])

    xs_smooth = fix_endpoints(xs_clean, xs_smooth, k_frac=0.06, min_k=3)

    return xs_smooth

def enforce_monotonic_wave(xs):
    """
    Given a 1D array of x positions along the wave (ordered by y),
    enforce monotonic decrease before the minimum and monotonic increase after,
    replacing violations with linear interpolation between surrounding points.
    """
    xs = np.array(xs, dtype=float)
    n = len(xs)
    if n < 3:
        return xs  # too few points

    # Find index of minimum (minima)
    min_idx = np.argmin(xs)

    # --- Before the minima: enforce decreasing (x[i] <= x[i-1]) ---
    i = 1
    while i < min_idx:
        if xs[i] > xs[i - 1]:
            # find next j > i where xs[j] < xs[i - 1]
            j = i + 1
            while j < min_idx and xs[j] >= xs[i - 1]:
                j += 1
            if j < n:
                # Linear interpolate between (i-1) and j
                for k in range(i, j):
                    t = (k - (i - 1)) / (j - (i - 1))
                    xs[k] = xs[i - 1] + t * (xs[j] - xs[i - 1])
            i = j
        else:
            i += 1

    # --- After the minima: enforce increasing (x[i] >= x[i-1]) ---
    i = min_idx + 1
    while i < n:
        if xs[i] < xs[i - 1]:
            # find next j > i where xs[j] > xs[i - 1]
            j = i + 1
            while j < n and xs[j] <= xs[i - 1]:
                j += 1
            if j < n:
                for k in range(i, j):
                    t = (k - (i - 1)) / (j - (i - 1))
                    xs[k] = xs[i - 1] + t * (xs[j] - xs[i - 1])
            i = j
        else:
            i += 1

    return xs

def newer_analyze_and_append_waves(
    image,
    userPeaks,
    edgeBound=2,
    inertia=0.5,
    prominence=10,
    minSep=6,
    noiseFloor=100,
    modality="singlet"
):
    """
    userPeaks: [(col, row)] length 1 or 2
    returns: list of waves, each wave = [(row, col), ...]
    """

    height, width = image.shape
    n = len(userPeaks)
    assert n in (1, 2)

    # --- denoise ---
    for _ in range(3):
        image = median_filter(image, size=3)

    # --- midpoint ---
    midpoint = int(sum(p[1] for p in userPeaks) / n)

    # --- midline for seeding ---
    midline = np.zeros(width)
    for x in range(width):
        for k in range(-edgeBound + 1, edgeBound):
            midline[x] += image[midpoint + k, x]
    midline /= (2 * edgeBound - 1)
    midline[midline < noiseFloor] = 0

    mid_peaks = find_peaks(midline, prominence=prominence)[0]

    # --- seed matching ---
    seeds = []
    for col, _ in userPeaks:
        seeds.append(min(mid_peaks, key=lambda p: abs(p - col)))
    seeds = sorted(seeds)

    # ---------- prediction ----------
    def predict(track):
        if len(track) < 2:
            return track[-1][1]
        (_, c1), (_, c0) = track[-1], track[-2]
        return c1 + inertia * (c1 - c0)

    # ---------- initialize ----------
    waves_up   = [[(midpoint, s)] for s in seeds]
    waves_down = [[(midpoint, s)] for s in seeds]

    # ============================================================
    # TRACK UPWARD
    # ============================================================
    for r in range(midpoint - 1, edgeBound, -1):
        waves_up = _step(image, r, waves_up, predict,
                         edgeBound, prominence, minSep, noiseFloor)

    # ============================================================
    # TRACK DOWNWARD (fresh state)
    # ============================================================
    for r in range(midpoint + 1, height - edgeBound):
        waves_down = _step(image, r, waves_down, predict,
                           edgeBound, prominence, minSep, noiseFloor)

    # ============================================================
    # MERGE + POSTPROCESS
    # ============================================================
    waves = []
    for w_up, w_dn in zip(waves_up, waves_down):
        wave = w_up[::-1] + w_dn[1:]
        wave = postprocess_wave(wave, midpoint)
        waves.append(wave)

    # ============================================================
    # DOUBLET MODE: append average wave
    # ============================================================
    if modality == "doublet" and len(waves) == 2:
        w1, w2 = waves

        # defensive alignment
        L = min(len(w1), len(w2))

        avg_wave = []
        for i in range(L):
            r1, c1 = w1[i]
            r2, c2 = w2[i]

            # rows should match; trust w1
            avg_col = 0.5 * (c1 + c2)
            avg_wave.append((r1, avg_col))

        waves.append(avg_wave)

    return waves

def _step(image, row, waves, predict, edgeBound, prominence, minSep, noiseFloor):
    width = image.shape[1]
    n = len(waves)

    # --- build row profile ---
    line = np.zeros(width)
    for x in range(width):
        for k in range(-edgeBound + 1, edgeBound):
            line[x] += image[row + k, x]
    line /= (2 * edgeBound - 1)
    line[line < noiseFloor] = 0

    peaks = find_peaks(line, prominence=prominence)[0]
    if len(peaks) == 0:
        return waves

    # ============================================================
    # SINGLE WAVE (trivial assignment)
    # ============================================================
    if n == 1:
        pred = predict(waves[0])
        best = min(peaks, key=lambda p: abs(p - pred))
        waves[0].append((row, best))
        return waves

    # ============================================================
    # TWO WAVES (joint assignment)
    # ============================================================
    p0 = predict(waves[0])
    p1 = predict(waves[1])

    best = None
    best_cost = np.inf

    for a in peaks:
        for b in peaks:
            if a >= b:
                continue

            c0 = abs(a - p0)
            c1 = abs(b - p1)

            sep = b - a
            sep_penalty = 0
            if sep < minSep:
                sep_penalty = 1000 * (1 - sep / minSep)

            cost = c0 + c1 + sep_penalty

            if cost < best_cost:
                best_cost = cost
                best = (a, b)

    if best is not None:
        waves[0].append((row, best[0]))
        waves[1].append((row, best[1]))

    return waves


def postprocess_wave(wave, midpoint):
    rows = np.array([r for r, _ in wave])
    cols = np.array([c for _, c in wave], dtype=float)

    # --- strong smoothing ---
    win = min(11, len(cols) // 2 * 2 + 1)
    cols = savgol_filter(cols, win, 2)

    # --- midpoint index ---
    mid_idx = np.argmin(np.abs(rows - midpoint))

    # =====================================================
    # Directional monotonic enforcement (assumption #1)
    # =====================================================

    # above midpoint: move left (cols non-increasing)
    for i in range(mid_idx - 1, -1, -1):
        cols[i] = max(cols[i], cols[i + 1])

    # below midpoint: move right (cols non-decreasing)
    for i in range(mid_idx + 1, len(cols)):
        cols[i] = max(cols[i], cols[i - 1])

    # --- re-smooth to remove clamping kinks ---
    win = min(7, len(cols) // 2 * 2 + 1)
    cols = savgol_filter(cols, win, 2)

    # =====================================================
    # Optional curvature limiter (assumption #2)
    # =====================================================
    '''
    max_delta = 4.0  # pixels per row

    for i in range(1, len(cols)):
        d = cols[i] - cols[i - 1]
        if abs(d) > max_delta:
            cols[i] = cols[i - 1] + np.sign(d) * max_delta
            '''

    return list(zip(rows, cols))


def new_analyze_and_append_waves(
        image, 
        wave_threshold=0, 
        min_wave_gap=45, 
        horizontal_proximity_threshold=45, 
        vertical_proximity_threshold=1, 
        max_missing_rows=2,
        min_points_per_wave=50,
        edge_bound=10,
        modality="singlet",
        smooth=True,
        smooth_window=21,
        smooth_polyorder=3):
    """
    Analyze waves in the image, calculate the center of mass for each wave, 
    and append them to the correct wave line based on proximity to previously detected waves.

    Args:
        image: 2D array representing the RoI of the motion profile.
        wave_threshold: Intensity threshold for wave detection.
        min_wave_gap: Minimum distance to separate different waves within a row.
        horizontal_proximity_threshold: Maximum allowable horizontal distance 
            to append a new center point to an existing wave.
        vertical_proximity_threshold: Maximum allowable vertical distance between rows 
            for a wave to be considered continuous.
        max_missing_rows: Maximum number of consecutive rows where a wave can be missing 
            before terminating it.
        min_points_per_wave: Minimum number of points for a wave line to be considered valid.
        smooth: Whether to apply Savitzky-Golay smoothing to wave lines.
        smooth_window: Base window length for the smoothing filter (auto-adjusts per wave).
        smooth_polyorder: Polynomial order for smoothing.

    Returns:
        wave_lines: List of wave lines, each being a list of (y, x_center) points.
    """

    if modality != "singlet":
        horizontal_proximity_threshold = 25

    height, width = image.shape
    wave_lines = []  # Initialize list to store wave lines
    wave_missing_counts = []  # Track how many rows each wave has been missing for

    if height < 50:
        min_points_per_wave = height / 2

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

            waves.append(current_wave)

            # Calculate center of mass for each wave
            center_of_mass_points = []
            new_waves = []

            if modality != "singlet":
                for wave in waves:
                    new_waves.append(wave[:int(0.5 * len(wave))])
                    new_waves.append(wave[int(0.5 * len(wave)):])
            else:
                new_waves = waves

            for wave in new_waves:
                wave_intensities = row[wave]
                total_intensity = np.sum(wave_intensities)

                if total_intensity > 0:
                    positions = np.array(wave)
                    center_of_mass = np.sum(positions * wave_intensities) / total_intensity
                    center_of_mass_points.append((y, center_of_mass))

            # Append center of mass points to the closest wave line
            for (y, x_center) in center_of_mass_points:
                added = False

                for idx, wave_line in enumerate(wave_lines):
                    last_y, last_x_center = wave_line[-1]

                    if abs(x_center - last_x_center) < horizontal_proximity_threshold and abs(y - last_y) <= vertical_proximity_threshold:
                        wave_line.append((y, x_center))
                        wave_missing_counts[idx] = 0
                        added = True
                        break

                if not added:
                    wave_lines.append([(y, x_center)])
                    wave_missing_counts.append(0)

        else:
            # Increment missing row counts
            for i in range(len(wave_missing_counts)):
                wave_missing_counts[i] += 1

        # Remove wave lines missing too long
        wave_lines = [wave_line for idx, wave_line in enumerate(wave_lines) if wave_missing_counts[idx] < max_missing_rows]
        wave_missing_counts = [count for count in wave_missing_counts if count < max_missing_rows]

    # Filter out short wave lines
    wave_lines = [wave_line for wave_line in wave_lines if len(wave_line) >= min_points_per_wave]

    
    
    # --- Apply adaptive Savitzky-Golay smoothing ---
    if smooth and len(wave_lines) > 0:
        smoothed_wave_lines = []
        for wave_line in wave_lines:
            ys, xs = zip(*wave_line)
            ys = np.array(ys)
            xs = np.array(xs)
            n = len(xs)

            # Apply smoothing only if enough data points exist
            if n >= (smooth_polyorder + 3):
                # Choose adaptive window length (odd, ≤ n)
                window_len = min(smooth_window, n // 2 * 2 + 1)
                window_len = max(3, window_len)  # Ensure at least 3
                polyorder = min(smooth_polyorder, window_len - 2)

                try:
                    xs_smooth = robust_smooth_1d(xs, smooth=True, base_window=smooth_window, polyorder=smooth_polyorder)
                except ValueError:
                    xs_smooth = xs  # Fallback if filter fails
            else:
                xs_smooth = xs  # Too few points, skip smoothing

            smoothed_wave_lines.append(list(zip(ys, xs_smooth)))

        wave_lines = smoothed_wave_lines
    
    '''
    height, width = image.shape
    clipped_wave_lines = []
    for wave_line in wave_lines:
        clipped = [(y, x) for (y, x) in wave_line if 0 <= y < height and 0 <= x < width]
        if len(clipped) > 0:
            clipped_wave_lines.append(clipped)

    wave_lines = clipped_wave_lines
    '''

    if smooth:
        corrected_wave_lines = []
        for wave_line in wave_lines:
            ys, xs = zip(*wave_line)
            xs_corrected = enforce_monotonic_wave(xs)
            corrected_wave_lines.append(list(zip(ys, xs_corrected)))

        wave_lines = corrected_wave_lines

    # --- Apply adaptive Savitzky-Golay smoothing ---
    if smooth and len(wave_lines) > 0:
        smoothed_wave_lines = []
        for wave_line in wave_lines:
            ys, xs = zip(*wave_line)
            ys = np.array(ys)
            xs = np.array(xs)
            n = len(xs)

            # Apply smoothing only if enough data points exist
            if n >= (smooth_polyorder + 3):
                # Choose adaptive window length (odd, ≤ n)
                window_len = min(smooth_window, n // 2 * 2 + 1)
                window_len = max(3, window_len)  # Ensure at least 3
                polyorder = min(smooth_polyorder, window_len - 2)

                try:
                    xs_smooth = robust_smooth_1d(xs, smooth=True, base_window=smooth_window, polyorder=smooth_polyorder)
                except ValueError:
                    xs_smooth = xs  # Fallback if filter fails
            else:
                xs_smooth = xs  # Too few points, skip smoothing

            smoothed_wave_lines.append(list(zip(ys, xs_smooth)))

        wave_lines = smoothed_wave_lines

    return wave_lines


@deprecated("Use new_analyze_and_append_waves instead, has modality check")
def analyze_and_append_waves(image, 
                             wave_threshold=0, 
                             min_wave_gap=15, 
                             horizontal_proximity_threshold=15, 
                             vertical_proximity_threshold=1, 
                             max_missing_rows=2,
                             min_points_per_wave=50,
                             modality = ""):
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

    print(wave_lines)

    return wave_lines

def gaussian(x, A, mu, sigma, offset):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset


def best_shift_for_row(row_data, template_x, template_gauss):
    """
    Compute best horizontal shift of a Gaussian template to match the row data.
    Returns the shift and the resulting peak position (mu + shift).
    """

    def loss(shift):
        shifted_x = template_x + shift
        # Since interpolation is not used, clip to bounds
        valid = (shifted_x >= 0) & (shifted_x < len(row_data))
        if not np.any(valid):
            return np.inf
        interp_template = np.zeros_like(template_gauss)
        interp_template[valid] = template_gauss[valid]
        return np.sum((row_data - interp_template)**2)

    res = minimize(loss, x0=0.0, method="Nelder-Mead")
    shift = float(res.x[0])

    return shift

def track_wave(seed_y, mu0, x_template, y_template, image, direction=1, window=15):
    """
    Track a wave starting from seed_y and mu0.
    direction = +1 for downward, -1 for upward
    """

    H, W = image.shape
    wave_points = [(seed_y, mu0)]
    prev_mu = mu0

    y_range = range(seed_y + direction, H if direction > 0 else -1, direction)

    for y in y_range:
        row = image[y, :].astype(float)

        # Define local horizontal window
        x_start = max(0, int(prev_mu - window))
        x_end   = min(W, int(prev_mu + window))
        row_local = row[x_start:x_end]

        # Interpolate template to match local window size
        t_len = len(row_local)
        x_dense_local = np.linspace(0, t_len - 1, len(x_template))
        y_template_local = np.interp(np.arange(t_len), x_dense_local, y_template)

        # Compute cross-correlation
        corr = np.correlate(row_local - row_local.mean(),
                            y_template_local - y_template_local.mean(),
                            mode='valid')
        best_shift = np.argmax(corr) - (len(y_template_local) // 2)

        new_mu = prev_mu + best_shift
        new_mu = max(0, min(W - 1, new_mu))

        wave_points.append((y, new_mu))
        prev_mu = new_mu

    if direction < 0:
        wave_points = list(reversed(wave_points))
    return wave_points

def gaussianWaveDetection(image, seed_points, search_window=10):
    """
    Track FECO waves from user-selected seed points.
    image: 2D numpy array
    seed_points: list of (x, y)
    search_window: pixels left/right to look in next row
    """
    H, W = image.shape
    waves = []

    for seed_x, seed_y in seed_points:
        wave_coords = []

        # Start at seed
        prev_x = seed_x

        for y in range(seed_y, -1, -1):  # upward
            row = image[y, :]
            left = max(0, int(prev_x - search_window))
            right = min(W, int(prev_x + search_window))
            local_row = row[left:right]

            # Find local maxima
            peaks, _ = find_peaks(local_row)
            if len(peaks) == 0:
                x_new = prev_x
            else:
                # pick peak closest to previous
                peak_positions = peaks + left
                x_new = peak_positions[np.argmin(np.abs(peak_positions - prev_x))]

            wave_coords.append((y, x_new))
            prev_x = x_new

        wave_coords.reverse()

        # Track downward
        prev_x = seed_x
        for y in range(seed_y + 1, H):
            row = image[y, :]
            left = max(0, int(prev_x - search_window))
            right = min(W, int(prev_x + search_window))
            local_row = row[left:right]

            peaks, _ = find_peaks(local_row)
            if len(peaks) == 0:
                x_new = prev_x
            else:
                peak_positions = peaks + left
                x_new = peak_positions[np.argmin(np.abs(peak_positions - prev_x))]

            wave_coords.append((y, x_new))
            prev_x = x_new

        waves.append(wave_coords)

    print(waves)

    return waves

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
        print(x_coords)
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

        # Plot each wave line with a label
        ax.plot(x_coords, y_coords, color=colors[wavenum-1], label=f"Wave {wavenum}")

        # Plot leftward linear approximation with a label
        if left_slope is not None:
            left_fit_x = np.array(left_x_coords)
            left_fit_y = left_slope * left_fit_x + left_intercept
            ax.plot(left_fit_x, left_fit_y, 'b--', label=f"Wave {wavenum} Left Fit")

        # Plot rightward linear approximation with a label
        if right_slope is not None:
            right_fit_x = np.array(right_x_coords)
            right_fit_y = right_slope * right_fit_x + right_intercept
            ax.plot(right_fit_x, right_fit_y, 'r--', label=f"Wave {wavenum} Right Fit")

        # Plot the intersection point with a label
        if intersection_point:
            ax.plot(intersection_point[0], intersection_point[1], 'go', label=f"Wave {wavenum} Intersection")

    # Reapply title, labels, and limit the legend for clarity
    ax.set_title("Turnaround Estimation")
    ax.set_xlabel("X (columns)")
    ax.set_ylabel("Y (rows)")
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.,
        fontsize='small'
    )
    fig.tight_layout() 

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

def _sum_of_neg_gaussians(x, *params):
    """
    Sum of *positive-amplitude* Gaussians.
    params: [A1, mu1, sigma1, A2, mu2, sigma2, ...]
    """
    x = np.asarray(x)
    y = np.zeros_like(x)
    n = len(params) // 3
    for i in range(n):
        A = params[3*i + 0]
        mu = params[3*i + 1]
        sigma = params[3*i + 2]
        if sigma <= 0:
            continue
        y += A * np.exp(-0.5 * ((x - mu)/sigma)**2)
    return y

# --- NEW MODEL FUNCTION ---
def _linear_baseline_plus_gaussians(x, m, c, *gauss_params):
    """
    A linear baseline (m*x + c) minus a sum of Gaussians.
    """
    baseline = m * x + c
    # _sum_of_neg_gaussians returns positive dips, so we subtract them
    dips = _sum_of_neg_gaussians(x, *gauss_params) 
    return baseline - dips

def arbitrary_gaussian_fits(
    data, 
    plot=True, 
    min_gaussians=1, 
    max_gaussians=15, 
    prominence=1, 
    invert_peaks=False,
    forced_peaks=None
):
    """
    Fit Gaussian features (either dips or peaks) with a fitted linear baseline.

    Parameters
    ----------
    data : array-like
        List of (x,y) points.
    forced_peaks : list of x positions from the GUI (optional)
        If provided, these override automatic peak detection.
    """
    x = np.array([p[0] for p in data], dtype=float)
    y = np.array([p[1] for p in data], dtype=float)

    # --- Estimate linear baseline ---
    m_est, c_est = np.polyfit(x, y, 1)
    current_x, current_y = x, y
    for _ in range(3):
        baseline_guess = m_est * current_x + c_est
        residuals = current_y - baseline_guess
        std_dev = np.std(residuals)
        keep_indices = residuals > -1.5 * std_dev
        if np.sum(keep_indices) < 2:
            break
        current_x, current_y = current_x[keep_indices], current_y[keep_indices]
        m_est, c_est = np.polyfit(current_x, current_y, 1)
    y_baseline_est = m_est * x + c_est

    # --- Compute residual depending on dip/peak detection ---
    if invert_peaks:
        residual = y - y_baseline_est
    else:
        residual = y_baseline_est - y

    residual = np.clip(residual, 0, None)
    residual_smooth = savgol_filter(residual, window_length=21, polyorder=3)

    # ==========================================================
    # === NEW: Forced peak locations override peak detection ===
    # ==========================================================
    if forced_peaks is not None and len(forced_peaks) > 0:
        # Use forced peaks directly
        forced_peaks = np.array(forced_peaks, dtype=float)

        # Keep only those inside data range
        forced_peaks = forced_peaks[(forced_peaks >= x[0]) & (forced_peaks <= x[-1])]
        forced_peaks = np.sort(forced_peaks)

        n_gauss = len(forced_peaks)

        # Build initial guess: amplitude = local residual, sigma = small default
        p0 = [m_est, c_est]
        for mu0 in forced_peaks:
            idx = np.argmin(np.abs(x - mu0))
            A0 = residual[idx]
            sigma0 = max(3.0, (x[-1] - x[0]) * 0.01)
            p0 += [A0, mu0, sigma0]

        # Bounds: restrict mu near forced peaks (±5 pixels)
        lower = [-np.inf, -np.inf]
        upper = [np.inf, np.inf]
        for mu0 in forced_peaks:
            lower += [0.0, mu0 - 5, 1e-3]             # amplitude≥0, mu near forced
            upper += [np.inf, mu0 + 5, (x[-1]-x[0])/2]

    else:
        # ===================================================
        # === ORIGINAL peak detection section remains here ===
        # ===================================================
        peaks, _ = find_peaks(residual_smooth, prominence=prominence)
        if len(peaks) == 0:
            peaks = [np.argmax(residual)]

        if len(peaks) < min_gaussians:
            peaks = list(peaks)
            while len(peaks) < min_gaussians:
                peaks.append(peaks[-1])
        elif len(peaks) > max_gaussians:
            peaks = peaks[np.argsort(residual[peaks])[-max_gaussians:]]

        n_gauss = len(peaks)

        p0 = [m_est, c_est]
        lower, upper = [-np.inf, -np.inf], [np.inf, np.inf]
        for idx in peaks:
            A0 = residual[idx]
            mu0 = x[idx]
            sigma0 = max(3.0, (x[-1]-x[0]) * 0.01)
            p0 += [A0, mu0, sigma0]

            lower += [0.0, x[0], 1e-3]
            upper += [np.inf, x[-1], (x[-1]-x[0])*2]

    # --- Continue as before ---
    sigma_weights = np.ones_like(y)
    flat_indices = np.where(residual < prominence)[0]
    sigma_weights[flat_indices] = 0.1

    def model_func(x, m, c, *params):
        baseline = m * x + c
        peaksum = _sum_of_neg_gaussians(x, *params)
        if invert_peaks:
            return baseline + peaksum
        else:
            return baseline - peaksum

    popt, pcov = curve_fit(
        model_func,
        x, y,
        p0=p0,
        bounds=(lower, upper),
        sigma=sigma_weights,
        maxfev=20000
    )

    m_fit, c_fit = popt[0], popt[1]
    gauss_popt = popt[2:]

    amps, mus, sigs = [], [], []
    for i in range(n_gauss):
        amps.append(gauss_popt[3*i])
        mus.append(gauss_popt[3*i + 1])
        sigs.append(gauss_popt[3*i + 2])

    x_dense = np.linspace(np.min(x), np.max(x), 2000)
    y_fit = model_func(x_dense, *popt)

    # --- Plot remains unchanged ---
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(x, y, label='data', lw=1)
        plt.plot(x_dense, y_fit, label=f'fit (n={n_gauss})', lw=2)
        y_baseline_fit = m_fit * x_dense + c_fit
        plt.plot(x_dense, y_baseline_fit, '--', color='gray', label='baseline')
        for A, mu, sigma in zip(amps, mus, sigs):
            y_g = A * np.exp(-0.5 * ((x_dense - mu)/sigma)**2)
            if invert_peaks:
                plt.plot(x_dense, y_baseline_fit + y_g, '--', alpha=0.7)
            else:
                plt.plot(x_dense, y_baseline_fit - y_g, '--', alpha=0.7)
        plt.legend()
        plt.show()

    return {
        "baseline_slope": m_fit,
        "baseline_intercept": c_fit,
        "amplitudes": np.array(amps),
        "means": np.array(mus),
        "sigmas": np.array(sigs),
        "popt": popt,
        "pcov": pcov,
        "n_gauss": n_gauss,
        "x_dense": x_dense,
        "y_fit": y_fit,
    } 