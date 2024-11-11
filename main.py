import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import csv
from PIL import Image, ImageTk, ImageSequence
from enums import CalibrationValues

import tracking  
from exceptions import error_popup, warning_popup

class SFA_FECO_UI:
    """
        Main UI function for SFA FECO 
    """
    def __init__(self, root):
        self.root = root
        self.root.title("SFA FECO Analyzer")
        self.root.geometry("400x400+300+100") # Force the parent window to start at a set position

        # Constants for window sizing and positioning
        self.DEFAULT_WIDTH_RATIO = 0.35
        self.DEFAULT_HEIGHT_RATIO = 0.7

        self.raw_video_file_path = None
        self.motion_profile_file_path = None  # To store the chosen or generated data file path
        self.calibration_video_file_path = None
        self.split_file_path = None
        self.centerlines_csv_path = None

        self.split_frame_num = 0
        self.roi_offset= 0
        self.analysis_x_offset = 0
        self.analysis_y_offset = 0
        self.calibration_parameters = {}

        self.mica_thickness = '0'

        # Get screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Define scaling factors
        window_width = int(screen_width * self.DEFAULT_WIDTH_RATIO)
        window_height = int(screen_height * self.DEFAULT_HEIGHT_RATIO)
        
        self.root.geometry(f"{window_width}x{window_height}")

        # Configure validation to accept only numbers
        vcmd = (root.register(self.validate_numeric_input), '%P')

        # Setup styles
        self.setup_styles()

         # Configure grid layout for the root window
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=1)
        root.grid_rowconfigure(3, weight=1)
        root.grid_rowconfigure(4, weight=1)
        root.grid_rowconfigure(5, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=0)
        root.grid_columnconfigure(2, weight=1)

        # Prep Section: Add a column for the raw video select, crop/preprocess, and generate motion profile buttons
        prep_label = ttk.Label(root, text="STEP 1: Prep", style='Step.TLabel', font=20)
        prep_label.grid(row=0, column=0, sticky='ew', padx=10)

        # Raw video data select button
        self.select_raw_button = ttk.Button(root, text="Select Raw Video File", command=self.select_raw_video, style='Regular.TButton')
        self.select_raw_button.grid(row=1, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the selected file's name
        self.raw_file_label = ttk.Label(root, text="No file selected", style='Regular.TLabel')
        self.raw_file_label.grid(row=2, column=0, sticky='new', padx=10)

        # Crop/Preprocess button
        self.crop_button = ttk.Button(root, text="Crop/Preprocess", command=self.open_crop_preprocess_window, style='Regular.TButton')
        self.crop_button.grid(row=3, column=0, sticky='ew', padx=10, pady=5)

        # Generate motion profile button
        self.generate_motion_button = ttk.Button(root, text="Generate Motion Profile", command=self.generate_motion_profile, style='Regular.TButton')
        self.generate_motion_button.grid(row=4, column=0, sticky='ew', padx=10, pady=5)

        # Creating the frame to hold the calibration 
        self.calibration_subframe = ttk.Frame(root)
        self.calibration_subframe.grid(row=5, column=0, sticky='new')

        # Configure the column of the subframe to expand
        self.calibration_subframe.columnconfigure(0, weight=1)  

        # Adding a label to the subframe
        self.split_label = ttk.Label(self.calibration_subframe, text="Calibrate Wavelengths")
        self.split_label.grid(row=0, column=0, pady=(0, 5), sticky = 'w')

        # Select file for calibration
        self.select_calibration_file_button = ttk.Button(self.calibration_subframe, text="Select Calibration Video", command=self.select_calibration_file, style='Regular.TButton')
        self.select_calibration_file_button.grid(row=1, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the selected file's name
        self.calibration_file_label = ttk.Label(self.calibration_subframe, text="No file selected", style='Regular.TLabel')
        self.calibration_file_label.grid(row=2, column=0, sticky='new', padx=10)

        self.execute_wavelength_calibration = ttk.Button(self.calibration_subframe, text="Calibrate Wavelengths", command=self.run_wavelength_calibration, style='Regular.TButton')
        self.execute_wavelength_calibration.grid(row=3, column=0,sticky='ew', padx=10, pady=5)

        # Label to display the calibration status
        self.calibration_completion_label = ttk.Label(self.calibration_subframe, text="Calibration not completed", style='Regular.TLabel')
        self.calibration_completion_label.grid(row=4, column=0, sticky='new', padx=10)

        self.execute_thickness_calibration = ttk.Button(self.calibration_subframe, text="Calibrate Thickness", command=self.run_thickness_calibration, style='Regular.TButton')
        self.execute_thickness_calibration.grid(row=5, column=0,sticky='esw', padx=10, pady=(20, 5))

        # Label to display the thickness num
        self.calibration_thickness_label = ttk.Label(self.calibration_subframe, text="Mica thickness", style='Regular.TLabel')
        self.calibration_thickness_label.grid(row=6, column=0, sticky='sew', padx=10)

        # thickness display 
        self.thickness_display = tk.Text(self.calibration_subframe, height=1, width=10, wrap="none")
        self.thickness_display.grid(row=7, column=0, sticky="esw", padx=10, pady=(20, 5))

        # Insert the mica thickness value into the text widget
        self.thickness_display.insert("1.0", str(self.mica_thickness))

        # Set the text widget to be read-only
        self.thickness_display.config(state="disabled")


        # COLUMN TWO
        # Step 2: Analyze
        step3_label = ttk.Label(root, text="STEP 2: Analyze", style='Step.TLabel', font=20)
        step3_label.grid(row=0, column=2, sticky='ew', padx=10)

        # Button to choose an existing data file
        self.choose_data_button = ttk.Button(root, text="Choose Data File", command=self.choose_data_file, style='Regular.TButton')
        self.choose_data_button.grid(row=1, column=2, sticky='ew', padx=10)

        # File field for the output data
        self.data_file_label = ttk.Label(root, text="No file selected", style='Regular.TLabel')
        self.data_file_label.grid(row=2, column=2, columnspan=2, sticky='enw', padx=10)

        self.analyze_button = ttk.Button(root, text="Analyze", command=self.analyze, style='Regular.TButton')
        self.analyze_button.grid(row=3, column=2, sticky='ew')

        self.analyze_button = ttk.Button(root, text="Estimate Turnaround", command=self.estimate_turnaround, style='Regular.TButton')
        self.analyze_button.grid(row=4, column=2, sticky='ew')

        # Creating the frame to hold the button and the number-entry box
        self.split_subframe = ttk.Frame(root)
        self.split_subframe.grid(row=5, column=2, sticky='ew')

        # Adding a label to the frame
        self.split_label = ttk.Label(self.split_subframe, text="Frame turnaround: ")
        self.split_label.grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky = 'w')

        # Button to choose an existing data file
        self.choose_split_button = ttk.Button(self.split_subframe, text="Choose File to Split", command=self.choose_split_file, style='Regular.TButton')
        self.choose_split_button.grid(row=1, columnspan=2, sticky='ew', padx=10)

        # File field for the output data
        self.split_file_label = ttk.Label(self.split_subframe, text="No file selected", style='Regular.TLabel')
        self.split_file_label.grid(row=2, columnspan=2, sticky='enw', padx=10, pady=(0, 5))

        # Creating a StringVar to hold and control the value of the entry box
        self.split_var = tk.StringVar(value=str(self.split_frame_num))

        # Number-entry box
        self.split_entry = ttk.Entry(self.split_subframe, textvariable=self.split_var, validate='key', validatecommand=vcmd)
        self.split_entry.grid(row=3, column=0, padx=(0, 25), sticky='ew')

        # Split button
        self.analyze_button = ttk.Button(self.split_subframe, text="Split", command=self.split, style='Regular.TButton')
        self.analyze_button.grid(row=3, column=1, sticky='ew')

        # Make the columns expand correctly
        self.split_subframe.columnconfigure(0, weight=0)
        self.split_subframe.columnconfigure(1, weight=1)  

        # Add a vertical separator between columns
        vertical_separator = ttk.Separator(root, orient="vertical")
        vertical_separator.grid(row=0, column=1, rowspan=6, sticky='ns', padx=10)

    def setup_styles(self):
        self.btn_style = ttk.Style()
        self.btn_style.configure(
            "Regular.TButton",
            padding=(10, 5),
            relief="raised",
            width=10
        )

    def select_raw_video(self):
        # Open a file dialog to select a TIFF file
        """file_path = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd()),
            title='Browse for TIFF file',
            filetypes=[("TIFF Files", "*.tif *.tiff")]
        )"""
        file_path = "FR1-P1-bis.tif" # hardcoded
        if file_path:
            # Save the selected file path
            self.raw_video_file_path = file_path
            
            # Update the label to display the file name
            self.raw_file_label.config(text=f"Selected File: {os.path.basename(file_path)}")

            # Check if the file exists
            if not os.path.isfile(self.raw_video_file_path):
                msg = "Invalid file"
                error_popup(msg)

    def open_crop_preprocess_window(self):
        Frame_Prep_Window(self.raw_video_file_path, self.callback_handle_roi_selection)
        
    def callback_handle_roi_selection(self, roi_data):
        """
        Handle the ROI data returned from the Frame_Prep_Window.
        :param roi_data: Tuple containing (y_start, y_end, offset, frame).
        """
        self.y_start, self.y_end, self.roi_offset, cropped_frame = roi_data
        # print(f"ROI Selected: Y-Start: {self.y_start}, Y-End: {self.y_end}, Offset: {self.offset}")
    
    def generate_motion_profile(self):
        max_length = 15;

        # Ensure a file is selected before analyzing
        if self.raw_video_file_path:
            # Ask the user for a filename to save the data
            filename = filedialog.asksaveasfilename(defaultextension=".tiff", filetypes=[("tiff files", "*.tiff")])
            if filename:
                if hasattr(self, 'y_start') and hasattr(self, 'y_end'):
                    # Call the fine approximation function with the Y crop info
                    tracking.generate_motion_profile(self.raw_video_file_path, self.y_start, self.y_end, filename,)
                    self.motion_profile_file_path = filename

                    if len(self.motion_profile_file_path) > max_length:
                        data_file_text = '...' + self.motion_profile_file_path[len(self.motion_profile_file_path) - max_length:]
                        self.data_file_label.config(text=f"Using file: {data_file_text}")
                    else: 
                        self.data_file_label.config(text=f"Data saved: {self.motion_profile_file_path}")
                else:
                    msg = "Please select a region of interest in the crop/preprocess window"
                    error_popup(msg)

        else:
            msg = "No file selected, aborting"
            error_popup(msg)

    def select_calibration_file(self):
        # Open a file dialog to select a TIFF file
        """file_path = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd()),
            title='Browse for TIFF file',
            filetypes=[("TIFF Files", "*.tif *.tiff")]
        )"""
        file_path = "mica_gold.tif" # HARDCODED
        if file_path:
            # Save the selected file path
            self.calibration_video_file_path = file_path
            
            # Update the label to display the file name
            self.calibration_file_label.config(text=f"Selected File: {os.path.basename(file_path)}")

            # Check if the file exists
            if not os.path.isfile(self.calibration_video_file_path):
                msg = "Invalid file"
                error_popup(msg)

    def callback_get_calibration(self, values):
        self.calibration_completion_label.config(text="Calibration completed")
        self.calibration_parameters = values

    def run_wavelength_calibration(self):
        Wavelength_Calibration_Window(self.calibration_video_file_path, self.callback_get_calibration)
        return
    
    def run_thickness_calibration(self):
        Mica_Thickness_Calibration_Window(self.calibration_parameters, self.callback_get_thickness_value)

    def callback_get_thickness_value(self, thickness):
        self.mica_thickness = thickness       
        
        # Enable the widget to update the text
        self.thickness_display.config(state="normal")
        
        # Clear the current content and insert the new value
        self.thickness_display.delete("1.0", "end")
        self.thickness_display.insert("1.0", str(abs(thickness)) + 'um')
        
        # Disable the widget again to make it read-only
        self.thickness_display.config(state="disabled")

    # STEP 2

    def choose_data_file(self):
        max_length = 15;
        # Allow the user to choose an existing data file
        check_file = filedialog.askopenfilename(filetypes=[("Tiff files", "*.tiff")])
        if(check_file): 
            self.motion_profile_file_path = check_file
            if len(self.motion_profile_file_path) > max_length:
                data_file_text = '...' + self.motion_profile_file_path[len(self.motion_profile_file_path) - max_length:]
                self.data_file_label.config(text=f"Using file: {data_file_text}")
        else: 
            msg = "No file selected, aborting"
            error_popup(msg)

    def analyze(self):
        self.centerlines_csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not self.centerlines_csv_path:
            msg = "No output file selected, aborting"
            error_popup(msg)
        Motion_Analysis_Window(self.motion_profile_file_path, self.calibration_parameters, self.centerlines_csv_path, self.callback_handle_crop_offset)

    def callback_handle_crop_offset(self, offsets):
        """
        Handle the offset data returned from the Motion_Analysis_Window.
        :param offsets: int of y offset
        """
        self.analysis_x_offset = offsets[0]
        self.analysis_y_offset = offsets[1]

    def estimate_turnaround(self):
        # Ensure the output directory and filename components are handled separately
        # Assuming self.motion_profile_file_path holds the original file path
        if(self.motion_profile_file_path):
            original_path = self.motion_profile_file_path

            # Separate the file directory, base name, and extension
            file_dir = os.path.dirname(original_path)
            base_name, ext = os.path.splitext(os.path.basename(original_path))

            # Append "_cropped" to the base name and reassemble the path
            cropped_path = os.path.join(file_dir, f"{base_name}_cropped{ext}")

            self.split_frame_num = tracking.perform_turnaround_estimation(cropped_path, self.centerlines_csv_path, self.analysis_x_offset, self.analysis_y_offset) 
            self.split_var.set(str(self.split_frame_num))
        else: 
            msg = "No motion profile file selected"
            error_popup(msg)

    def split(self):
        file_to_split = self.split_file_path

        if(file_to_split):
            # Open the CSV file
            with open(file_to_split, 'r') as csv_file: 
                csv_reader = csv.reader(csv_file)
                header = next(csv_reader)  # Assuming the first row is the header

                # Prepare file paths
                base, ext = os.path.splitext(file_to_split)
                in_file_path = f"{base}_in{ext}"
                out_file_path = f"{base}_out{ext}"

                # Open new CSV files for writing
                with open(in_file_path, 'w', newline='') as in_file, open(out_file_path, 'w', newline='') as out_file:
                    in_writer = csv.writer(in_file)
                    out_writer = csv.writer(out_file)

                    # Write the header to both files
                    in_writer.writerow(header)
                    out_writer.writerow(header)

                    # Process each row and split based on the y value (second column)
                    for row in csv_reader:
                        y_value = float(row[1])  # Convert second column to a float
                        if y_value <= self.split_frame_num:
                            in_writer.writerow(row)
                        else:
                            out_writer.writerow(row)

            print(f"CSV files saved as {in_file_path} and {out_file_path}")
        else:
            msg = "No spltting file selected, aborting"
            error_popup(msg)
 
    def choose_split_file(self):
        max_length = 15;
        # Allow the user to choose an existing data file
        self.split_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if len(self.split_file_path) > max_length:
            data_file_text = '...' + self.split_file_path[len(self.split_file_path) - max_length:]
        if self.split_file_path:
            self.split_file_label.config(text=f"Using file: {data_file_text}")

    def update_entry(self, *args):
        # Automatically update split_frame_num when split_var changes
        try:
            self.split_frame_num = int(self.split_var.get())
        except ValueError:
            pass  # Ignore if the value isn't a valid integer
            
    def validate_numeric_input(self, value):
        # Allow only numbers (positive integers)
        return value.isdigit() or value == ""
        
class Frame_Prep_Window:
    SCALE_FACTOR = 0.75

    def __init__(self, file_path, roi_callback=None):
        self.raw_video_file_path = file_path
        self.roi_callback = roi_callback
        self.cropped_frame = None
        self.frames = []
        self.current_frame_index = 0
        self.crop_rectangle = None

        self.crop_start_y = 0
        self.crop_end_y = 0

        # Load the TIFF file using PIL
        try:
            self.tiff_image = Image.open(self.raw_video_file_path)
            self.frames = [frame.copy() for frame in ImageSequence.Iterator(self.tiff_image)]
        except Exception as e:
            error_popup(f"Failed to load TIFF file: {e}")
            return

        # Create the window
        self.window = tk.Toplevel()
        self.window.title("Frame Preparation Window")

        # Add an instruction label
        self.instruction_label = ttk.Label(self.window, text="Step 1: Select the region to crop (y-axis only). Press enter to accept.")
        self.instruction_label.pack(pady=5)

        # Create canvas and slider for frame selection
        self.canvas = tk.Canvas(self.window, bg="white")
        self.canvas.pack(fill="both", expand=True)
        
        self.slider = ttk.Scale(self.window, from_=0, to=len(self.frames) - 1, orient="horizontal", command=self.update_frame)
        self.slider.pack(fill="x")

        # Bind events
        self.canvas.bind("<Button-1>", self.start_crop)
        self.canvas.bind("<ButtonRelease-1>", self.end_crop)
        self.window.bind("<Escape>", self.cancel_crop)
        self.window.bind("<Return>", self.confirm_crop)

        # Display the initial frame
        self.update_frame(0)

    def update_frame(self, value):
        """Update the displayed frame based on the slider."""
        self.current_frame_index = int(float(value))
        frame = self.frames[self.current_frame_index]
        scaled_frame = tracking.scale_frame(frame, Frame_Prep_Window.SCALE_FACTOR)
        self.display_image(scaled_frame)

    def display_image(self, image):
        """Displays a PIL image on the canvas, scaling the window appropriately."""
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, self.crop_start_y, anchor="nw", image=self.photo)

        frame_width, frame_height = image.size
        self.canvas.config(width=frame_width, height=frame_height)
        self.window.geometry(f"{frame_width}x{frame_height + 80}")

    def start_crop(self, event):
        """Begins the crop selection (only y-axis)."""
        if(self.crop_start_y is not None):
            self.cancel_crop()
        self.crop_start_y = event.y
        self.crop_rectangle = self.canvas.create_rectangle(0, self.crop_start_y, self.canvas.winfo_width(), self.crop_start_y, outline="red", width=2)

    def end_crop(self, event):
        """Finalize the crop selection."""
        self.crop_end_y = event.y
        if(self.crop_rectangle):
            self.canvas.delete(self.crop_rectangle)
        self.crop_rectangle = self.canvas.create_rectangle(0, self.crop_start_y, self.canvas.winfo_width(), self.crop_end_y, outline="red", width=2)

    def cancel_crop(self, event = None):
        """Cancels the cropping selection."""
        if self.crop_rectangle:
            self.canvas.delete(self.crop_rectangle)
        self.crop_start_y = None
        self.crop_end_y = None

    def confirm_crop(self, event):
        """Confirms the crop and finalizes the cropped image."""
        if self.crop_start_y is not None and self.crop_end_y is not None:
            # Convert crop coordinates to original image scale
            y_start, y_end = sorted((int(self.crop_start_y / Frame_Prep_Window.SCALE_FACTOR), int(self.crop_end_y / Frame_Prep_Window.SCALE_FACTOR)))
            current_frame = self.frames[self.current_frame_index]
            cropped_frame = current_frame.crop((0, y_start, current_frame.width, y_end))

            # Scale the cropped frame and display it
            self.scaled_cropped_frame = tracking.scale_frame(cropped_frame, Frame_Prep_Window.SCALE_FACTOR)
            self.slider.pack_forget()
            self.instruction_label.config(text="Cropping complete. Ready for further processing. You may close the window.")
            self.display_image(self.scaled_cropped_frame)

            # Call the callback if provided
            if self.roi_callback:
                self.roi_callback((int(y_start/Frame_Prep_Window.SCALE_FACTOR), int(y_end/Frame_Prep_Window.SCALE_FACTOR), int(y_start/Frame_Prep_Window.SCALE_FACTOR), self.cropped_frame))
            else:
                print("No callback provided. ROI selection will not be returned.")
            
            # Unbind cropping events
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<ButtonRelease-1>")
        else: 
            msg = "No crop area selected"
            error_popup(msg)

class Motion_Analysis_Window:
    """
    A class to perform motion analysis on a TIFF image, allowing cropping, deletion, and wave analysis.
    Parameters:
        data_file_path (str): Path to the input data file (TIFF).
        calibration_parameters (dict): Dictionary containing slope and intercept for calibrated x-axis ticks.
        output_file_path (str): Path to save analysis results.
        offset_callback (function, optional): Callback function to handle offsets after cropping.
    """

    CROPPING_MODE = 'crop'
    DELETION_MODE = 'delete'
    FIGURE_SIZE = (12, 8)

    def __init__(self, data_file_path, calibration_parameters, output_file_path, offset_callback = None) -> None:
        self.y_offset = 0
        self.x_offset_start = 0
        self.x_offset_end = 0
        self.calibration_parameters = calibration_parameters
        self.offset_callback = offset_callback

        # Step 1: Request output file name
        self.output_filename = output_file_path

        # Step 2: Load the timelapse image (convert it to a NumPy array)
        self.timelapse_image = np.array(Image.open(data_file_path).convert('L'))  # Grayscale conversion
        self.file_path = data_file_path  # Store file path for saving

        # Step 3: Initialize mode (cropping or deleting)
        self.mode = Motion_Analysis_Window.CROPPING_MODE  # Start with cropping mode
        self.cropping_complete = False
        self.crop_area = None  # Store the crop area
        self.deletion_areas = []  # Store areas selected for deletion

        # Increase the figure size for larger display
        self.fig, self.ax = plt.subplots(figsize=Motion_Analysis_Window.FIGURE_SIZE)  # Set larger figure size
        self.ax.imshow(self.timelapse_image, cmap='gray')
        # self.ax.set_xlim( self.x_offset_start, self.x_offset_end)
        self.ax.set_title("Click and drag to crop the image, then press any key to confirm. Press escape to cancel selection.")

        # Create a RectangleSelector for cropping the image
        self.rect_selector = RectangleSelector(self.ax, self.on_select_crop, useblit=True, interactive=True)
        self.fig.canvas.mpl_connect('key_press_event', self.handle_key_press)

        plt.show(block=True)

        # Step 4: Run the analysis after cropping
        if self.cropping_complete:
            self.run_analysis()

    def handle_key_press(self, event):
        """Centralized key press handler based on mode."""
        if self.mode == self.CROPPING_MODE:
            self.handle_crop_keypress(event)
        elif self.mode == self.DELETION_MODE:
            self.handle_delete_keypress(event)

    def handle_crop_keypress(self, event):
        """Handle key presses specifically for cropping mode."""
        if event.key == 'enter':
            if self.crop_area:
                self.confirm_crop()
            else:
                print("No crop area selected.")
        elif event.key == 'escape':
            self.cancel_crop()

    def handle_delete_keypress(self, event):
        """Handle key presses specifically for deletion mode."""
        if event.key == 'enter':
            self.confirm_deletion()
        elif event.key == 'escape':
            self.cancel_deletion()

    def confirm_crop(self):
        print(f"Crop confirmed: {self.crop_area}")
        self.cropping_complete = True

        # Perform the crop
        x_start, x_end, y_start, y_end = self.crop_area
        self.x_offset_start = min(x_start, x_end)
        self.x_offset_end = max(x_start, x_end)
        self.y_offset = min(y_start, y_end)
        print(self.x_offset_start)
        print(self.x_offset_end)
        if self.offset_callback:
            offsets = (self.x_offset_start, self.y_offset)
            self.offset_callback(offsets)
        self.cropped_image = self.timelapse_image[y_start:y_end, x_start:x_end]
        
        # Get the base name and extension of the original file
        base_name, ext = os.path.splitext(self.file_path)

        # Create a new file name with "_cropped" appended to the base name
        cropped_file_path = f"{base_name}_cropped{ext}"

        # Convert NumPy array back to a PIL image and overwrite original file
        cropped_pil_image = Image.fromarray(self.cropped_image)
        cropped_pil_image.save(cropped_file_path)
        
        print(f"Image saved as {cropped_file_path} with the crop area {self.crop_area}")

        # Deactivate the cropping RectangleSelector
        self.rect_selector.set_active(False)

        # Switch to deletion mode after cropping is complete
        self.mode = Motion_Analysis_Window.DELETION_MODE
        plt.close(self.fig)  # Close the figure to proceed

    def cancel_crop(self):
        """Cancel the current cropping selection and reset the mode."""
        # Reset the crop area coordinates
        self.crop_area = None
        self.current_area = None
        self.cropping_complete = False

        # Clear any displayed crop rectangle on the plot
        self.ax.clear()
        self.ax.imshow(self.timelapse_image, cmap='gray')  # Redisplay the original image
        self.ax.set_title("Crop mode: Drag to select, Enter to confirm, Esc to cancel.")

        # Redraw the canvas
        plt.draw()

        # Ensure the RectangleSelector is active again for the next selection
        self.rect_selector.set_active(False)
        self.rect_selector.set_active(True)  # Reactivate to allow a new crop

    def confirm_deletion(self):
        """Apply deletions to the selected data and reapply calibrated axis ticks if applicable."""
        for area in self.deletion_areas:
            x_start, x_end, y_start, y_end = area
            for wave_line in self.wave_lines:
                wave_line[:] = [(y, x) for (y, x) in wave_line if not (x_start <= x <= x_end and y_start <= y <= y_end)]

        # Clear the deletion areas after applying deletions
        self.deletion_areas = []

        # Update the plot with the modified wave lines
        self.update_plot()

    def cancel_deletion(self):
        """Cancel the current deletion selection and reset any marked areas."""
        # Clear the list of deletion areas
        self.deletion_areas = []

        self.update_plot()

        # Ensure the RectangleSelector is active again for new selections
        self.rect_selector.set_active(True)

    def on_select_crop(self, eclick, erelease):
        """Callback for when the cropping rectangle is selected."""
        x_start, y_start = int(eclick.xdata), int(eclick.ydata)
        x_end, y_end = int(erelease.xdata), int(erelease.ydata)
        self.crop_area = (x_start, x_end, y_start, y_end)
        print(f"Crop area selected: {self.crop_area}")

    def on_select_delete(self, eclick, erelease):
        """Callback for when the deletion rectangle is selected."""
        x_start, y_start = int(eclick.xdata), int(eclick.ydata)
        x_end, y_end = int(erelease.xdata), int(erelease.ydata)
        deletion_area = (x_start, x_end, y_start, y_end)
        self.deletion_areas.append(deletion_area)
        print(f"Deletion area selected: {deletion_area}")

        # Draw a rectangle on the plot to show the selected area
        rect = plt.Rectangle((x_start, y_start), x_end - x_start, y_end - y_start,
                            linewidth=1, edgecolor='r', facecolor='none')
        self.ax.add_patch(rect)
        plt.draw()

        # Reset the RectangleSelector
        self.rect_selector.set_active(False)
        self.rect_selector.set_active(True)

    def run_analysis(self):
        """Run the analysis on the cropped image."""
        # Perform analysis on the cropped image
        self.wave_lines = tracking.analyze_and_append_waves(self.cropped_image)

        # Visualize the results and enable data deletion
        self.visualize_wave_centerlines(self.cropped_image, self.wave_lines)

    def visualize_wave_centerlines(self, image, wave_lines):
        """Visualize and enable deletion on the results with calibrated x-axis ticks if calibration is available."""
        # Store the image and wave lines
        self.cropped_image = image
        self.wave_lines = wave_lines

        # Use the new update_plot method to plot everything
        self.update_plot()
        
        plt.show()

    def update_plot(self):
        """Update the plot with the current wave lines and apply calibration if necessary."""
        # Check if the figure and axis already exist
        if not hasattr(self, 'fig') or not hasattr(self, 'ax') or not plt.fignum_exists(self.fig.number):
            # Create the figure and axis with the desired frame size
            self.fig, self.ax = plt.subplots(figsize=(10, 4))
            
            # Connect event handlers
            self.fig.canvas.mpl_connect('key_press_event', self.handle_key_press)
            self.fig.canvas.mpl_connect('close_event', self.on_close)
            
            # Initialize RectangleSelector for deletion
            self.rect_selector = RectangleSelector(self.ax, self.on_select_delete, useblit=True, interactive=False)
        
        self.ax.clear()
        
        # Replot the image
        self.ax.imshow(self.cropped_image, cmap='gray')

        # Replot the wave lines
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.wave_lines)))

        for idx, wave_line in enumerate(self.wave_lines):
            if wave_line:  # Ensure the wave_line is not empty
                y_coords = [point[0] for point in wave_line]
                x_coords = [point[1] for point in wave_line]
                self.ax.plot(x_coords, y_coords, color=colors[idx], label=f"Wave {idx + 1}")

        # Reapply title, labels, and legend
        self.ax.set_title("Highlight data to delete it. Enter to accept, Esc to cancel, close window to save.")
        if self.calibration_parameters:
            self.ax.set_xlabel(r"Wavelength, $\lambda$ (nm)")
        else:
            self.ax.set_xlabel("Pixels")
        self.ax.set_ylabel("Frame Number")
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Apply calibrated x-axis ticks if calibration parameters exist
        if self.calibration_parameters:
            ticks = self.ax.get_xticks()  # Get original x-axis ticks

            # Apply calibration to display ticks only
            calibrated_ticks = [
                self.calibration_parameters['slope'] * (tick - self.x_offset_start) + self.calibration_parameters['intercept']
                for tick in ticks
            ]
            self.ax.set_xticks(ticks)  # Original ticks for data
            self.ax.set_xticklabels([f"{tick:.2f}" for tick in calibrated_ticks])  # Show calibrated labels

            self.ax.set_xlim(0, self.x_offset_end - self.x_offset_start)

        # Redraw the updated plot
        plt.draw()

        # Adjust the layout to include all elements
        self.fig.tight_layout()

    def on_close(self, event):
        """Save the modified wave lines and the figure when the window is closed."""
        # Save the wave centerlines to CSV
        self.save_wave_centerlines_to_csv(self.wave_lines, self.output_filename)
        
        # Update the plot title to a proper name
        self.ax.set_title("Final Wave Centerlines")
        
        # Force an immediate redraw of the figure
        self.fig.canvas.draw()
        
        # Save the figure as a PDF
        pdf_filename = "last_centerline_visualization.pdf"
        try:
            self.fig.savefig(pdf_filename, format='pdf', bbox_inches='tight')
            print(f"Figure saved as '{pdf_filename}'")
        except Exception as e:
            print(f"Error saving figure as PDF: {e}")

    def save_wave_centerlines_to_csv(self, wave_lines, output_filename):
        """Save the wave centerlines to a CSV file, with an optional calibrated CSV if parameters are available."""
        try:
            # Ensure the output directory and filename components are handled separately
            output_dir = os.path.dirname(output_filename)
            base_filename = os.path.basename(output_filename)

            # Save the original CSV
            with open(output_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Wave Index", "Frame Number", "Center of Mass X Coord"])

                for wave_idx, wave_line in enumerate(wave_lines):
                    for (y, x_center) in wave_line:
                        frame_number = y + self.y_offset  # Calculate once
                        writer.writerow([wave_idx + 1, frame_number, x_center + self.x_offset_start])

            print(f"Wave centerlines successfully saved to {output_filename}")

            # Generate calibrated CSV if calibration parameters exist
            if self.calibration_parameters:
                # Construct the calibrated filename in the same directory as the original file
                calibrated_filename = os.path.join(output_dir, f"calibrated_{base_filename}")
                with open(calibrated_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Wave Index", "Frame Number", "Calibrated Center of Mass X Coord"])

                    for wave_idx, wave_line in enumerate(wave_lines):
                        for (y, x_center) in wave_line:
                            frame_number = y + self.y_offset  # Ensure consistency
                            calibrated_x = self.calibration_parameters['slope'] * (x_center - self.x_offset_start) + self.calibration_parameters['intercept']
                            writer.writerow([wave_idx + 1, frame_number, calibrated_x])

                print(f"Calibrated wave centerlines successfully saved to {calibrated_filename}")

        except Exception as e:
            print(f"Error saving wave centerlines to CSV: {e}")

class Wavelength_Calibration_Window:
    def __init__(self, file_path, callback):
        self.file_path = file_path
        self.callback = callback
        self.window = tk.Toplevel()
        self.window.title("Calibration Window")

        # Load the image as a PIL image
        self.image = Image.open(self.file_path)

        # State variables
        self.crop_start = None
        self.crop_rectangle = None
        self.waves = None
        self.stage = 1
        self.scale_factor = 0.75
        self.cropped_image = None
        self.wave_x_avgs = []
        self.selected_waves = [] 
        self.num_waves = 3

        # Create instruction label
        self.instruction_label = ttk.Label(self.window, text="Step 1: Select the region to crop (y-axis only).")
        self.instruction_label.pack(pady=5)

        # Create UI elements
        self.canvas = tk.Canvas(self.window)
        self.canvas.pack(fill="both", expand=True)
        self.slider = ttk.Scale(self.window, from_=0, to=self.image.n_frames - 1, orient="horizontal", command=self.update_frame)
        self.slider.pack(fill="x")

        # Bind events
        self.canvas.bind("<Button-1>", self.click_start_crop)
        self.canvas.bind("<ButtonRelease-1>", self.end_crop)
        self.window.bind("<Escape>", self.cancel_crop)
        self.window.bind("<Return>", self.confirm_crop)

        # Load the initial frame
        self.update_frame(0)

    def update_frame(self, value):
        """Updates the displayed frame based on the slider value."""
        current_frame_index = int(float(value))
        self.image.seek(current_frame_index)
        scaled_frame = tracking.scale_frame(self.image, self.scale_factor)
        self.display_image(scaled_frame)

    def display_image(self, image):
        """Displays a PIL image on the canvas."""
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        
        frame_width, frame_height = image.size
        self.canvas.config(width=frame_width, height=frame_height)
        self.window.geometry(f"{frame_width}x{frame_height + 80}")

    def click_start_crop(self, event):
        """Begins the crop selection (only y-axis)."""
        if self.stage == 1:  # Only allow cropping in step 1    
            self.crop_start_y = event.y
            if self.crop_rectangle:
                self.canvas.delete(self.crop_rectangle)
        elif self.stage == 2:
            self.select_wave_click(event)

    def end_crop(self, event):
        """Draws the cropping line (only y-axis)."""
        
        if self.stage == 1:  # Only allow cropping in step 1 
            self.crop_end_y = event.y
            width = self.image.width  # Use the entire width of the image
            self.crop_rectangle = self.canvas.create_rectangle(0, self.crop_start_y, width, self.crop_end_y, outline="red", width=2)

    def cancel_crop(self, event):
        """Cancels the cropping selection."""
        if self.stage == 1:  # Only allow cropping in step 1 
            if self.crop_rectangle:
                self.canvas.delete(self.crop_rectangle)
            self.crop_start_y = None
            self.crop_end_y = None
        elif self.stage == 2:
            self.selected_waves = []
            self.update_overlay

    def confirm_crop(self, event):
        """Confirms the crop and proceeds to analyze waves."""
        if self.stage == 1 and self.crop_start_y is not None and self.crop_end_y is not None:
            y1 = min(self.crop_start_y, self.crop_end_y)
            y2 = max(self.crop_start_y, self.crop_end_y)
            cropped_frame = self.image.crop((0, y1, self.image.width, y2))
            self.canvas.delete(self.crop_rectangle)
            self.cropped_image = cropped_frame

            # Hide the slider after step 1 is complete
            self.slider.pack_forget()
            self.stage = 2
        
            # Update instructions
            self.instruction_label.config(text="Step 2: Select 3 wave lines.")
        
            self.run_wave_analysis(cropped_frame)

    def run_wave_analysis(self, image):
        """Runs the wave analysis on the cropped image."""
        self.waves = tracking.analyze_and_append_waves(np.array(image), wave_threshold=110)
        self.display_waves()

    def display_waves(self):
        """Displays the waves over the cropped image."""
        if self.waves and self.cropped_image is not None:
            # Ensure the cropped image is in RGB mode
            if self.cropped_image.mode != 'RGB':
                self.cropped_image = self.cropped_image.convert('RGB')
                
            # Convert the PIL image to a numpy array for OpenCV processing
            overlay = np.array(self.cropped_image).copy()

            # Print out the waves for debugging
            print(f"Number of waves: {len(self.waves)}")
            for wave_index, wave in enumerate(self.waves):
                # Calculate the average x position for the wave
                average_x = int(np.mean([point[1] for point in wave]))
                self.wave_x_avgs.append(average_x)
                print(f"Wave {wave_index} average x-position: {average_x}")

                # Ensure the average_x is within the image boundaries
                if 0 <= average_x < overlay.shape[1]:
                    # Draw a vertical line at the average x position
                    cv2.line(overlay, (average_x, 0), (average_x, overlay.shape[0]), (0, 255, 0), 2)
                else:
                    print(f"Wave {wave_index} average x-position is out of bounds and will not be drawn.")

            # Convert back to a PIL image if needed or continue processing with OpenCV
            overlay_pil = Image.fromarray(overlay)
            scaled_overlay = tracking.scale_frame(overlay_pil, self.scale_factor)
            self.display_image(scaled_overlay)

            # DEBUG: Display the frame using OpenCV for debugging
            debug_frame = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
            # self.display_image_with_pixel_scale(debug_frame)

            # Convert the numpy array back to a PIL image
            overlay_pil = Image.fromarray(debug_frame)
            
            # Scale the overlay image
            scaled_overlay = tracking.scale_frame(overlay_pil, self.scale_factor)
            
            # Display the scaled image
            self.display_image(scaled_overlay)
            
            # Bind the return key for wave selection confirmation
            self.window.bind("<Return>", self.confirm_wave_selection)

    def confirm_wave_selection(self, event):
        """Allows the user to select 3 wave lines and calculate the calibration equation."""
        if len(self.waves) < 3:
            print("Select at least 3 wave lines")
            return
        self.select_waves(3)

    def select_waves(self, num_waves):
        """Allows the user to select a number of wave lines by clicking on the canvas."""
        self.num_waves = num_waves
        self.canvas.bind("<Button-1>", self.select_wave_click)

        # Instructions for the user
        self.instructions_label = ttk.Label(self.window, text=f"Click to select {self.num_waves} wave lines.")
        self.instructions_label.pack(pady=5)

    def select_wave_click(self, event):
        """Handles click events to select wave lines based on the average x-coordinate."""
        x = event.x / self.scale_factor

        # Find the closest wave based on the x-coordinate clicked
        closest_wave = None
        min_distance = float('inf')
        for wave_x in self.wave_x_avgs:
            distance = abs(wave_x - x)
            # print(wave_x, " wave pixel")
            # print(distance, " distance")
            # print()
            if distance < min_distance:
                min_distance = distance
                closest_wave = wave_x

        if closest_wave is not None:
            # Check if the closest wave has already been selected; if not, add it
            if closest_wave not in self.selected_waves:
                self.selected_waves.append(closest_wave)

            # Redraw the overlay to highlight all waves (green for unselected, red for selected)
            self.update_overlay()

            # Check if enough waves have been selected
            if len(self.selected_waves) >= self.num_waves:
                self.canvas.unbind("<Button-1>")
                self.instruction_label.config(text="Wave selection complete. Press Enter to proceed.")
                self.window.bind("<Return>", self.calculate_transformation)

    def update_overlay(self):
        """Updates the overlay to highlight selected and unselected waves."""
        # Create an overlay image
        overlay = np.array(self.cropped_image).copy()

        # Draw all waves: green for unselected and red for selected
        for wave_x in self.wave_x_avgs:
            color = (0, 255, 0)  # Default color is green
            if wave_x in self.selected_waves:
                color = (255, 0, 0)  # Change to red for selected waves
            cv2.line(overlay, (int(wave_x), 0), (int(wave_x), overlay.shape[0]), color, 2)

        # Convert the overlay back to PIL format and scale it
        overlay_pil = Image.fromarray(overlay)
        scaled_overlay = tracking.scale_frame(overlay_pil, self.scale_factor)
        self.display_image(scaled_overlay)


    def calculate_transformation(self, event):
        """Calculates the calibration equation using the selected waves."""
        x_values = self.wave_x_avgs
        y_values = [CalibrationValues.HG_GREEN.value, CalibrationValues.HG_YELLOW_1.value, CalibrationValues.HG_YELLOW_2.value]
        coefficients = np.polyfit(x_values, y_values, 1)
        calibration_equation = {"slope": coefficients[0], "intercept": coefficients[1]}
        print(calibration_equation)
        self.callback(calibration_equation)
        self.window.destroy() 

class Mica_Thickness_Calibration_Window: 
    def __init__(self, calibration_parameters, callback):
        self.calibration_parameters = calibration_parameters
        self.callback = callback
        self.selected_waves = []
        
        # Prompt the user to select a TIFF file
        file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif")])
        if not file_path:
            print("No file selected. Aborting.")
            return

        # Set up crop area variables
        self.crop_area = None
        self.cropping_complete = False
        self.draw_rectangle = True
        self.mode = 'crop'
        self.cropped_frame = None
        self.cropped_frame_display = None
        self.scale_factor = .75

        self.selected_wavelengths = []
        
        # Load and scale the TIFF file
        self.tiff_image = Image.open(file_path)
        self.frames = [tracking.scale_frame(frame.copy(), self.scale_factor) for frame in ImageSequence.Iterator(self.tiff_image)]
        
        # Create a window
        self.window = tk.Toplevel()
        self.window.title("Calibrate Mica Thickness")

        # Create instructional label above the canvas
        self.instruction_label = tk.Label(self.window, text="Step 1: Select the region to crop (y-axis only). Use mouse drag, press Enter to confirm, or Esc to reset.")
        self.instruction_label.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 0))

        # Create canvas and slider for frame selection
        self.canvas = tk.Canvas(self.window, bg="white")
        self.canvas.grid(row=1, column=0, columnspan=2, sticky="nsew")
        
        self.slider = ttk.Scale(self.window, from_=0, to=len(self.frames) - 1, orient="horizontal", command=self.update_frame)
        self.slider.grid(row=2, column=0, columnspan=2, sticky="ew")

        # Show the first frame
        self.current_frame_index = 0
        self.display_frame()

        # Bind keyboard events
        self.window.bind("<Escape>", self.reset_crop)
        self.window.bind("<Return>", self.confirm_crop)
        self.canvas.bind("<ButtonPress-1>", self.start_crop)

    def update_instructions(self, text):
        """Update the instructions label with the provided text."""
        self.instruction_label.config(text=text)

    def update_frame(self, value):
        """Update the displayed frame based on the slider."""
        self.current_frame_index = int(float(value))
        self.display_frame()

    def display_frame(self):
        """Display the current frame on the canvas."""
        if self.cropped_frame is None:
            frame = self.frames[self.current_frame_index]
        else:
            frame = self.cropped_frame
        # Ensure `frame` is a PIL Image
        if isinstance(frame, np.ndarray):  # If frame is a NumPy array, convert it to a PIL image
            frame = Image.fromarray(frame)

        self.tk_image = ImageTk.PhotoImage(frame)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.image = self.tk_image

        frame_width, frame_height = frame.size
        self.canvas.config(width=frame_width, height=frame_height)
        self.window.geometry(f"{frame_width}x{frame_height + 100}")  # Adjust for label height

    def start_crop(self, event):
        """Begins the crop selection if in cropping mode."""
        if self.mode == 'crop' and not self.cropping_complete:
            self.crop_start_y = event.y
            self.canvas.bind("<B1-Motion>", self.drag_crop)
            self.canvas.bind("<ButtonRelease-1>", self.end_crop)

    def drag_crop(self, event):
        """Draw a rectangle to indicate the crop area."""
        if not self.cropping_complete:
            self.crop_end_y = event.y
            self.canvas.delete("crop_rectangle")
            self.canvas.create_rectangle(0, self.crop_start_y, self.tk_image.width(), self.crop_end_y, outline="red", width=2, tags="crop_rectangle")

    def end_crop(self, event):
        """Finalize the crop area."""
        self.crop_end_y = event.y
        self.crop_area = (min(self.crop_start_y, self.crop_end_y), max(self.crop_start_y, self.crop_end_y))
        print(f"Crop area selected: {self.crop_area}")

    def reset_crop(self, event=None):
        """Reset the crop selection."""
        self.cropping_complete = False
        self.crop_area = None
        self.cropped_frame = None
        self.canvas.delete("crop_rectangle")
        self.update_instructions("Step 1: Select the region to crop (y-axis only). Use mouse drag, press Enter to confirm, or Esc to reset.")
        print("Crop selection reset.")

    def confirm_crop(self, event=None):
        """Confirm the crop selection, finalize the cropped image, and disable further cropping."""
        if self.crop_area:
            self.cropping_complete = True
            y_start, y_end = self.crop_area
            self.cropped_frame = np.array(self.frames[self.current_frame_index])[y_start:y_end, :]
        
            min_val = self.cropped_frame.min()
            max_val = self.cropped_frame.max()

            # Normalize the array to the range 0-255
            normalized_frame = 255 * (self.cropped_frame - min_val) / (max_val - min_val)

            # Convert to uint8 for display purposes
            self.cropped_frame = normalized_frame.astype(np.uint8)

            # Update the display with the cropped image
            self.display_frame()
            
            # Disable further cropping by unbinding events and switching mode
            self.canvas.unbind("<Button-1>")
            self.canvas.unbind("<ButtonRelease-1>")
            self.canvas.unbind("<B1-Motion>")
            self.mode = 'select'  # Update the mode to stop cropping

            # Process the cropped frame
            self.slider.grid_forget()  # Hide the slider after cropping
            self.update_instructions("Step 2: Select two wave lines by clicking on them.")
            self.run_wave_detection()

    def run_wave_detection(self):
        """Detect and filter wave lines, then allow user to select two for calibration."""
        self.wave_lines = tracking.analyze_and_append_waves(self.cropped_frame, wave_threshold=40, min_points_per_wave=10, min_wave_gap=10)
        wave_averages = [np.mean([x for _, x in wave]) for wave in self.wave_lines]
        
        # Filter out clustered wave averages
        filtered_averages = []
        for avg in sorted(wave_averages):
            if not filtered_averages or abs(filtered_averages[-1] - avg) > 5:
                filtered_averages.append(avg)
        
        self.filtered_averages = filtered_averages
        self.display_filtered_waves()

    def display_filtered_waves(self):
        """Display the cropped image with filtered wave averages overlaid for selection."""
        
        # Create an RGB version of the cropped frame to serve as the base for overlay
        overlay_base = self.cropped_frame.copy()
        if len(overlay_base.shape) == 2:  # If grayscale, convert to RGB
            overlay_base = cv2.cvtColor(overlay_base, cv2.COLOR_GRAY2RGB)

        # Overlay the lines on a copy of overlay_base, leaving self.cropped_frame unaltered
        overlay_image = overlay_base.copy()
        for avg_x in self.filtered_averages:
            cv2.line(overlay_image, (int(avg_x), 0), (int(avg_x), overlay_image.shape[0]), (0, 255, 0), 2)

        # Convert the overlay image to PIL format for Tkinter display
        overlay_pil = Image.fromarray(overlay_image)
        self.tk_overlay = ImageTk.PhotoImage(overlay_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_overlay)
        self.canvas.image = self.tk_overlay

        # Bind click event for wave selection
        self.canvas.bind("<Button-1>", self.select_wave_click)

    def select_wave_click(self, event):
        """Handles click events to select wave lines based on the average x-coordinate."""
        x = event.x

        # Find the closest wave based on the x-coordinate clicked
        closest_wave = None
        min_distance = float('inf')
        for wave_x in self.filtered_averages:
            distance = abs(wave_x - x)
            if distance < min_distance:
                min_distance = distance
                closest_wave = wave_x

        if closest_wave is not None:
            # Check if the closest wave has already been selected; if not, add it
            if closest_wave not in self.selected_waves:
                self.selected_waves.append(closest_wave)

            # Redraw the overlay to highlight all waves (green for unselected, red for selected)
            self.update_overlay()

            # Check if enough waves have been selected
            if len(self.selected_waves) >= 2:
                # Unbind click event to prevent further selections
                self.canvas.unbind("<Button-1>")
                self.update_instructions("Press Enter to confirm and calculate thickness.")
                # Bind Enter key for wavelength conversion confirmation
                self.window.bind("<Return>", self.convert_to_wavelengths)

    def update_overlay(self):
        """Redraw overlay with selected and unselected wave lines."""
        overlay_image = self.cropped_frame.copy()
        
        # Convert to RGB if grayscale
        if len(overlay_image.shape) == 2:
            overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_GRAY2RGB)

        # Draw all wave lines, red for selected and green for unselected
        for avg_x in self.filtered_averages:
            color = (0, 255, 0)  # Default to green for unselected waves
            if avg_x in self.selected_waves:
                color = (255, 0, 0)  # Change to red for selected waves
            cv2.line(overlay_image, (int(avg_x), 0), (int(avg_x), overlay_image.shape[0]), color, 2)

        overlay_pil = Image.fromarray(overlay_image.astype(np.uint8))
        self.tk_overlay = ImageTk.PhotoImage(overlay_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_overlay)
        self.canvas.image = self.tk_overlay

    def convert_to_wavelengths(self, event=None):
        """Convert selected x-coordinates to wavelengths using calibration parameters."""
        # Ensure calibration parameters are defined
        if not self.calibration_parameters:
            print("Calibration parameters not provided.")
            return
        
        # Calculate wavelengths using slope and intercept
        slope = self.calibration_parameters['slope']
        intercept = self.calibration_parameters['intercept']
        self.selected_wavelengths = [slope * (x * 4 / 3) + intercept for x in self.selected_waves]
        
        print(f"Selected wavelengths: {self.selected_wavelengths}")

        thick = self.calculate_thickness()
        self.callback(thick)
        self.window.destroy()

    def calculate_thickness(self):
        """
        Calculates mica thickness (T) using selected wavelengths and calibration parameters.
        Requires exactly two selected wavelengths stored in `self.selected_waves`.
        """
        if len(self.selected_waves) != 2:
            print("Please select exactly two wave points for calibration.")
            return None

        lambda_n_nm, lambda_n_minus_1_nm = self.selected_wavelengths
        lambda_n_angstrom = lambda_n_nm * 10
        lambda_n_minus_1_angstrom = lambda_n_minus_1_nm * 10

        mu_mica = 1.5757 + (5.89 * 10**5) / (lambda_n_angstrom ** 2)

        try:
            T = (lambda_n_angstrom * lambda_n_minus_1_angstrom) / (4 * (lambda_n_minus_1_angstrom - lambda_n_angstrom) * mu_mica)
            T_um = T / 10000
            print(f"Calculated mica thickness (T): {T_um}")
            return T_um
        except ZeroDivisionError:
            print("Error: The selected wavelengths are too close, leading to division by zero.")
            return None

if __name__ == "__main__":
    root = tk.Tk()
    app = SFA_FECO_UI(root)
    root.mainloop()