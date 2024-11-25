import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Slider
import csv
from PIL import Image, ImageTk, ImageSequence
from enums import CalibrationValues
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tracking  
from exceptions import error_popup, warning_popup

class SFA_FECO_UI:
    """
        Main UI function for SFA FECO 
    """
    def __init__(self, root):
        self.root = root
        self.root.title("SFA FECO Analyzer")

        # Constants for window sizing and positioning
        self.DEFAULT_WIDTH_RATIO = 0.5
        self.DEFAULT_HEIGHT_RATIO = 0.5

        self.MAX_FILE_DISP_LENGTH = 20

        # Initialize file paths and parameters
        self.raw_video_file_path = None
        self.motion_output_file_path = None
        self.wavelength_calibration_video_file_path = None
        self.thickness_input_file_path = None
        self.split_file_path = None
        self.data_file_path = None
        self.analyze_output_file_path = None

        self.split_frame_num = 0
        self.roi_offset = 0
        self.analysis_x_offset = None
        self.analysis_y_offset = None
        self.calibration_parameters = {}
        self.mica_thickness = '0'

        # Set protocol for window close to ensure full exit
        self.root.protocol("WM_DELETE_WINDOW", self.exit_application)

        # Get screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Define window size
        window_width = int(screen_width * self.DEFAULT_WIDTH_RATIO)
        window_height = int(screen_height * self.DEFAULT_HEIGHT_RATIO)

        self.root.geometry(f"{window_width}x{window_height}+300+100")  # Start at a set position

        # Configure validation to accept only numbers
        vcmd = (root.register(self.validate_numeric_input), '%P')

        # Setup styles
        self.setup_styles()

        # Configure grid layout for the root window
        for i in range(7):
            self.root.grid_rowconfigure(i, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=0)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=0)
        self.root.grid_columnconfigure(4, weight=1)


        # region Subframe for Calibration
        prep_label = ttk.Label(self.root, text="STEP 0: Calibrate", style='Step.TLabel', font=20)
        prep_label.grid(row=0, column=0, sticky='new', padx=10)

        self.calibration_subframe = ttk.Frame(self.root)
        self.calibration_subframe.grid(row=1, column=0, rowspan=2, sticky='ew')

        # Configure the column of the subframe to expand
        self.calibration_subframe.columnconfigure(0, weight=1)

        # Select wavelength calibration video file
        self.select_calibration_file_button = ttk.Button(self.calibration_subframe, text="Select Wavelength Calibration Video", command=self.select_wavelength_calibration_file, style='Regular.TButton')
        self.select_calibration_file_button.grid(row=1, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the selected wavelength calibration video file's name
        self.wavelength_calibration_file_label = ttk.Label(self.calibration_subframe, text="No file selected", style='Regular.TLabel')
        self.wavelength_calibration_file_label.grid(row=2, column=0, sticky='ew', padx=10)

        # Calibrate Wavelengths button
        self.execute_wavelength_calibration = ttk.Button(self.calibration_subframe, text="Calibrate Wavelengths", command=self.run_wavelength_calibration, style='Regular.TButton')
        self.execute_wavelength_calibration.grid(row=3, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the wavelength calibration status
        self.calibration_completion_label = ttk.Label(self.calibration_subframe, text="Calibration not completed", style='Regular.TLabel')
        self.calibration_completion_label.grid(row=4, column=0, sticky='new', padx=10, pady=(0, 20))

        # Select Thickness File button
        self.select_thickness_file_button = ttk.Button(self.calibration_subframe, text="Select Thickness Calibration Video", command=self.select_thickness_file, style='Regular.TButton')
        self.select_thickness_file_button.grid(row=5, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the selected thickness file's name
        self.thickness_file_label = ttk.Label(self.calibration_subframe, text="No file selected", style='Regular.TLabel')
        self.thickness_file_label.grid(row=6, column=0, sticky='new', padx=10)

        # Calibrate Thickness button
        self.execute_thickness_calibration = ttk.Button(self.calibration_subframe, text="Calibrate Thickness", command=self.run_thickness_calibration, style='Regular.TButton')
        self.execute_thickness_calibration.grid(row=7, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the thickness
        self.calibration_thickness_label = ttk.Label(self.calibration_subframe, text="Mica thickness:", style='Regular.TLabel')
        self.calibration_thickness_label.grid(row=8, column=0, sticky='sew', padx=10)

        # Thickness display
        self.thickness_display = tk.Text(self.calibration_subframe, height=1, width=10, wrap="none")
        self.thickness_display.grid(row=9, column=0, sticky="esw", padx=10, pady=(5, 5))

        # Insert the mica thickness value into the text widget
        self.thickness_display.insert("1.0", str(self.mica_thickness))

        # Set the text widget to be read-only
        self.thickness_display.config(state="disabled")
        # endregion
        
        # Add a vertical separator between columns
        vertical_separator = ttk.Separator(self.root, orient="vertical")
        vertical_separator.grid(row=0, column=1, rowspan=7, sticky='ns', padx=10)

        # region Step 1: Prep
        prep_label = ttk.Label(self.root, text="STEP 1: Prep", style='Step.TLabel', font=20)
        prep_label.grid(row=0, column=2, sticky='ew', padx=10)

        # Subframe for Raw Video selection
        self.raw_video_subframe = ttk.Frame(self.root)
        self.raw_video_subframe.grid(row=1, column=2, sticky='ew')

        self.raw_video_subframe.columnconfigure(0, weight=1)

        # Raw video data select button
        self.select_raw_button = ttk.Button(self.raw_video_subframe, text="Select Raw Video File", command=self.select_raw_video, style='Regular.TButton')
        self.select_raw_button.grid(row=0, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the selected file's name
        self.raw_file_label = ttk.Label(self.raw_video_subframe, text="No file selected", style='Regular.TLabel')
        self.raw_file_label.grid(row=1, column=0, sticky='ew', padx=10)

        # Crop/Preprocess button
        self.crop_button = ttk.Button(self.raw_video_subframe, text="Crop/Preprocess", command=self.open_crop_preprocess_window, style='Regular.TButton')
        self.crop_button.grid(row=2, column=0, sticky='ew', padx=10, pady=5)

        # Subframe for Generate Motion Profile
        self.motion_profile_subframe = ttk.Frame(self.root)
        self.motion_profile_subframe.grid(row=2, column=2, sticky='ew')

        self.motion_profile_subframe.columnconfigure(0, weight=1)

        # Output file selection button for Generate Motion Profile
        self.select_motion_output_button = ttk.Button(self.motion_profile_subframe, text="Select Motion Profile Output File", command=self.select_motion_output_file, style='Regular.TButton')
        self.select_motion_output_button.grid(row=0, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the selected output file's name
        self.motion_output_file_label = ttk.Label(self.motion_profile_subframe, text="No output file selected", style='Regular.TLabel')
        self.motion_output_file_label.grid(row=1, column=0, sticky='ew', padx=10)

        # Generate motion profile button
        self.generate_motion_button = ttk.Button(self.motion_profile_subframe, text="Generate Motion Profile", command=self.generate_motion_profile, style='Regular.TButton')
        self.generate_motion_button.grid(row=2, column=0, sticky='ew', padx=10, pady=5)
        # endregion

        # Add a vertical separator between columns
        vertical_separator = ttk.Separator(self.root, orient="vertical")
        vertical_separator.grid(row=0, column=3, rowspan=7, sticky='ns', padx=10)

        # region Step 2: Analyze
        step3_label = ttk.Label(self.root, text="STEP 2: Analyze", style='Step.TLabel', font=20)
        step3_label.grid(row=0, column=4, sticky='ew', padx=10)

        # region Subframe for Data File Selection
        self.motion_profile_file_subframe = ttk.Frame(self.root)
        self.motion_profile_file_subframe.grid(row=1, column=4, sticky='ew')

        self.motion_profile_file_subframe.columnconfigure(0, weight=1)

        # Button to choose an existing data file
        self.choose_motion_profile_file_button = ttk.Button(self.motion_profile_file_subframe, text="Choose Motion Profile File", command=self.select_analysis_input_image_file, style='Regular.TButton')
        self.choose_motion_profile_file_button.grid(row=0, column=0, sticky='ew', padx=10, pady=5)

        # File field for the data file
        self.motion_profile_file_label = ttk.Label(self.motion_profile_file_subframe, text="No file selected", style='Regular.TLabel')
        self.motion_profile_file_label.grid(row=1, column=0, sticky='enw', padx=10)
        # endregion

        # region Subframe for Analyze
        self.analyze_subframe = ttk.Frame(self.root)
        self.analyze_subframe.grid(row=2, column=4, sticky='ew')

        self.analyze_subframe.columnconfigure(0, weight=1)

        # Output file selection button for Analyze
        self.select_analyze_output_button = ttk.Button(self.analyze_subframe, text="Select Output File", command=self.select_analyze_output_file, style='Regular.TButton')
        self.select_analyze_output_button.grid(row=0, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the selected output file's name
        self.analyze_output_file_label = ttk.Label(self.analyze_subframe, text="No output file selected", style='Regular.TLabel')
        self.analyze_output_file_label.grid(row=1, column=0, sticky='ew', padx=10)

        # Analyze button
        self.analyze_button = ttk.Button(self.analyze_subframe, text="Analyze", command=self.analyze, style='Regular.TButton')
        self.analyze_button.grid(row=2, column=0, sticky='ew', padx=10, pady=5)

        # Estimate Turnaround button
        self.estimate_turnaround_button = ttk.Button(self.analyze_subframe, text="Estimate Turnaround of Output", command=self.estimate_turnaround, style='Regular.TButton')
        self.estimate_turnaround_button.grid(row=3, column=0, sticky='ew', padx=10, pady=5)
        # endregion

        # region Subframe for Split functionality
        self.split_subframe = ttk.Frame(self.root)
        self.split_subframe.grid(row=3, column=4, sticky='ew')

        self.split_subframe.columnconfigure(0, weight=1)

        # Adding a label to the frame
        self.split_label = ttk.Label(self.split_subframe, text="Split output on Turnaround:")
        self.split_label.grid(row=0, column=0, columnspan=2, pady=(0, 5), sticky='w')

        # Button to choose an existing data file to split
        self.choose_split_file_button = ttk.Button(self.split_subframe, text="Choose File to Split", command=self.select_split_file, style='Regular.TButton')
        self.choose_split_file_button.grid(row=1, column=0, columnspan=2, sticky='ew', padx=10)

        # File field for the file to split
        self.split_file_label = ttk.Label(self.split_subframe, text="No file selected", style='Regular.TLabel')
        self.split_file_label.grid(row=2, column=0, columnspan=2, sticky='enw', padx=10, pady=(0, 5))

        # Creating a StringVar to hold and control the value of the entry box
        self.split_var = tk.StringVar(value=str(self.split_frame_num))

        # Turnaround frame label
        self.split_frame_num_label = ttk.Label(self.split_subframe, text="Turnaround frame #: ", style='Regular.TLabel')
        self.split_frame_num_label.grid(row=2, column=0, columnspan=2, sticky='e', padx=10, pady=(0, 5))

        # Number-entry box
        self.split_entry = ttk.Entry(self.split_subframe, textvariable=self.split_var, validate='key', validatecommand=vcmd)
        self.split_entry.grid(row=3, column=1, padx=(10, 5), sticky='ew')

        # Split button
        self.split_button = ttk.Button(self.split_subframe, text="Split", command=self.split, style='Regular.TButton')
        self.split_button.grid(row=3, column=0, sticky='ew', padx=(5, 10))

        # Configure columns in split_subframe
        self.split_subframe.columnconfigure(0, weight=1)
        self.split_subframe.columnconfigure(1, weight=1)
        # endregion
        # endregion
    
    def exit_application(self):
        """Cleanly exit the application."""
        # Close all matplotlib figures
        plt.close('all')

        # Destroy the Tkinter root window
        self.root.quit()  # This stops the Tkinter main loop
        self.root.destroy()

        # Exit the program forcefully to ensure no lingering processes
        sys.exit()

    def setup_styles(self):
        """
            function for defining the styles to be used in the base UI
        """
        self.btn_style = ttk.Style()
        self.btn_style.configure(
            "Regular.TButton",
            padding=(10, 5),
            relief="raised",
            width=10
        )

    def select_wavelength_calibration_file(self):
        """
            Function for selecting wavelength calibration input file. Checks for validity and updates label. 
        """
        # Open a file dialog to select a TIFF file
        file_path = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd()),
            title='Browse for TIFF file',
            filetypes=[("TIFF Files", "*.tif *.tiff")]
        )
        # file_path = "mica_gold.tif" # HARDCODED
        if file_path:
            # Save the selected file path
            self.wavelength_calibration_video_file_path = file_path
            
            # Update the label to display the file name
            if len(self.wavelength_calibration_video_file_path) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.wavelength_calibration_video_file_path[len(self.wavelength_calibration_video_file_path) - self.MAX_FILE_DISP_LENGTH:]
                self.wavelength_calibration_file_label.config(text=data_file_text)
            else: 
                self.wavelength_calibration_file_label.config(text=self.wavelength_calibration_video_file_path)


            # Check if the file exists
            if not os.path.isfile(self.wavelength_calibration_video_file_path):
                msg = "Invalid file"
                error_popup(msg)

    def run_wavelength_calibration(self):
        """
            Function to open the wavelength calibration window. Checks for input file. 
        """
        if(not self.wavelength_calibration_video_file_path): 
            msg = "Please select an input file"
            error_popup(msg)
            return
        Wavelength_Calibration_Window(self.wavelength_calibration_video_file_path, self.callback_get_wavelength_calibration)

    def callback_get_wavelength_calibration(self, parameters):
        """
        Handles the callback for retrieving the parameters of the pixel -> wavelength conversion
        
        Args:
            values (float, float): a tuple containing the slope and intercept of the linear equation converting pixels to um
        """
        self.calibration_completion_label.config(text="Calibration completed")
        self.calibration_parameters = parameters
    
    def select_thickness_file(self):
        """Select input file for Calibrate Thickness. Updates label."""
        self.thickness_input_file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif")])
        if self.thickness_input_file_path:
            if len(self.thickness_input_file_path) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.thickness_input_file_path[len(self.thickness_input_file_path) - self.MAX_FILE_DISP_LENGTH:]
                self.thickness_file_label.config(text=data_file_text)
            else:
                self.thickness_file_label.config(text=self.thickness_input_file_path) 

    def run_thickness_calibration(self):
        """
            Function for running the thickness calibration window. Checks for input first. 
        """
        if(not self.thickness_input_file_path): 
            msg = "Please select an input file"
            error_popup(msg)
            return
        Mica_Thickness_Calibration_Window(self.calibration_parameters, self.thickness_input_file_path, self.callback_get_thickness_value)

    def callback_get_thickness_value(self, thickness):
        """
            Function for retrieving the thickness of the mica in um

        Args:
            thickness (float): thickness of the mica in um
        """
        self.mica_thickness = thickness       
        
        # Enable the widget to update the text
        self.thickness_display.config(state="normal")
        
        # Clear the current content and insert the new value
        self.thickness_display.delete("1.0", "end")
        self.thickness_display.insert("1.0", str(abs(thickness)) + 'um')
        
        # Disable the widget again to make it read-only
        self.thickness_display.config(state="disabled")

    def select_raw_video(self):
        """
            Function for the user to select a file for the input. Updates the label and checks for validity.
        """
        # Open a file dialog to select a TIFF file
        file_path = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd()),
            title='Browse for TIFF file',
            filetypes=[("TIFF Files", "*.tif *.tiff")]
        )
        # file_path = "FR1-P1-bis.tif" # hardcoded
        if file_path:
            # Save the selected file path
            self.raw_video_file_path = file_path
            
            # Update the label to display the file name
            if len(self.raw_video_file_path) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.raw_video_file_path[len(self.raw_video_file_path) - self.MAX_FILE_DISP_LENGTH:]
                self.raw_file_label.config(text=data_file_text)
            else:
                self.raw_file_label.config(text=self.raw_video_file_path)

            # Check if the file exists
            if not os.path.isfile(self.raw_video_file_path):
                msg = "Invalid file"
                error_popup(msg)

    def open_crop_preprocess_window(self):
        """
            Function to open crop/preprocess window. Checks for valid input first.         
        """
        if(not self.raw_video_file_path): 
            msg = "Please select an input file"
            error_popup(msg)
            return
        Frame_Prep_Window(self.raw_video_file_path, self.callback_handle_roi_selection)
        
    def callback_handle_roi_selection(self, roi_data):
        """
        Handle the ROI data returned from the Frame_Prep_Window.
        :param roi_data: Tuple containing (y_start, y_end, offset, frame).
        """
        self.y_start, self.y_end, self.roi_offset, cropped_frame = roi_data 
    
    def select_motion_output_file(self):
        """Select output file for Generate Motion Profile. Also updates the label."""
        self.motion_output_file_path = filedialog.asksaveasfilename(defaultextension=".tif", filetypes=[("tiff files", "*.tif")])
        if self.motion_output_file_path:
            if len(self.motion_output_file_path) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.motion_output_file_path[len(self.motion_output_file_path) - self.MAX_FILE_DISP_LENGTH:]
                self.motion_output_file_label.config(text=data_file_text)
            else: 
                self.motion_output_file_label.config(text=self.motion_output_file_path) 

    def generate_motion_profile(self):
        """
            Function for calling tracking.generate_motion_profile. Checks for valid input and output files. 
        """
        # Ensure a file is selected before analyzing
        if self.raw_video_file_path:
            # Ask the user for a filename to save the data
            filename = self.motion_output_file_path
            if filename:
                if hasattr(self, 'y_start') and hasattr(self, 'y_end'):
                    # Call the fine approximation function with the Y crop info
                    tracking.generate_motion_profile(self.raw_video_file_path, self.y_start, self.y_end, filename,)

                    if len(self.motion_output_file_path) > self.MAX_FILE_DISP_LENGTH:
                        data_file_text = '...' + self.motion_output_file_path[len(self.motion_output_file_path) - self.MAX_FILE_DISP_LENGTH :]
                        self.motion_profile_file_label.config(text=f"Using file: {data_file_text}")
                    else: 
                        self.motion_profile_file_label.config(text=f"Data saved: {self.motion_output_file_path}")
                else:
                    msg = "Please select a region of interest in the crop/preprocess window"
                    error_popup(msg)
            else: 
                msg = "Please select an output file"
                error_popup(msg)
        else:
            msg = "Please select an input file"
            error_popup(msg)
        return

    # STEP 2

    def select_analysis_input_image_file(self):
        """
            Function for the user to select a file for analysis. Updates label accordingly. 
        """
        # Allow the user to choose an existing data file
        check_file = filedialog.askopenfilename(filetypes=[("Tiff files", "*.tiff")])
        if(check_file): 
            self.motion_output_file_path = check_file
            if len(self.motion_output_file_path) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.motion_output_file_path[len(self.motion_output_file_path) - self.MAX_FILE_DISP_LENGTH:]
                self.motion_profile_file_label.config(text=data_file_text)
            else:
                self.motion_profile_file_label.config(text=self.motion_output_file_path)
        else: 
            msg = "No file selected, aborting"
            error_popup(msg)

    def select_analyze_output_file(self):
        """Select output file for Analyze. Updates label as well."""
        self.analyze_output_file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if self.analyze_output_file_path:
            if len(self.analyze_output_file_path) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.analyze_output_file_path[len(self.analyze_output_file_path) - self.MAX_FILE_DISP_LENGTH:]
                self.analyze_output_file_label.config(text=data_file_text)
            else: 
                self.analyze_output_file_label.config(text=self.analyze_output_file_path) 

    def analyze(self):
        """
            Function for opening the analysis window. Checks for input and output file validity. 
        """
        if not self.motion_output_file_path:
            msg = "Please select an input file"
            error_popup(msg)
            return
        if not self.analyze_output_file_path:
            msg = "Please select an output file"
            error_popup(msg)
            return
        Motion_Analysis_Window(self.motion_output_file_path, self.calibration_parameters, self.analyze_output_file_path, self.callback_handle_crop_offset)

    def callback_handle_crop_offset(self, offsets):
        """
            Handle the offset data returned from the Motion_Analysis_Window.
            :param offsets: int of y offset
        """
        self.analysis_x_offset = offsets[0]
        self.analysis_y_offset = offsets[1]

    def estimate_turnaround(self):
        """
            Function to call tracking.perform_turnaround_estimation. Checks for previous step w/ warning and input w/ error
        """

        # Ensure the output directory and filename components are handled separately
        # Assuming self.motion_profile_file_path holds the original file path
        if(not self.analysis_x_offset):
            msg = "No offsets declared. May cause display errors. Would you like to continue?"
            if(warning_popup(msg)):
                return
        if(self.motion_output_file_path):
            original_path = self.motion_output_file_path

            # Separate the file directory, base name, and extension
            file_dir = os.path.dirname(original_path)
            base_name, ext = os.path.splitext(os.path.basename(original_path))

            # Append "_cropped" to the base name and reassemble the path
            cropped_path = os.path.join(file_dir, f"{base_name}_cropped{ext}") 

            self.split_frame_num = tracking.perform_turnaround_estimation(cropped_path, self.analyze_output_file_path, self.analysis_x_offset, self.analysis_y_offset) 
            self.split_var.set(str(self.split_frame_num))
        else: 
            msg = "No motion profile file selected"
            error_popup(msg)

    def split(self):
        """
            Simple function to split a given CSV file along the centerline given. Checks for input.
        """
        file_to_split = self.split_file_path

        if(not self.split_frame_num):
            msg = "No split frame selected, aborting"
            error_popup(msg)
            return

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
            msg = "No splitting file selected, aborting"
            error_popup(msg)
 
    def select_split_file(self):
        """
            Select output file for splitting. Updates label as well.
        """
        max_length = 15;
        # Allow the user to choose an existing data file
        self.split_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if len(self.split_file_path) > max_length:
            data_file_text = '...' + self.split_file_path[len(self.split_file_path) - max_length:]
            self.split_file_label.config(text=f"Using file: {data_file_text}")
        else: 
            self.split_file_label.config(text=f"Using file: {self.split_file_path}")

    def update_split_frame_entry(self, *args):
        """Automatically update split_frame_num when split_var changes"""
        try:
            self.split_frame_num = int(self.split_var.get())
        except ValueError:
            pass  # Ignore if the value isn't a valid integer
            
    def validate_numeric_input(self, value):
        # Allow only numbers (positive integers)
        return value.isdigit() or value == ""
    
class Frame_Prep_Window:
    """
        A GUI-based tool for preparing and processing frames from a TIFF file.

        This class provides functionality to:
        - Load and display frames from a multi-frame TIFF file.
        - Dynamically crop a region of interest (ROI) using mouse interaction.
        - Scale frames for display and processing.
        - Navigate through frames using a slider.

        Attributes:
            raw_video_file_path (str): Path to the input TIFF file.
            roi_callback (function): Callback function to handle the selected ROI.
            cropped_frame (PIL.Image or None): The cropped frame after ROI selection.
            frames (list): List of frames extracted from the TIFF file.
            current_frame_index (int): Index of the currently displayed frame.
            crop_start_y (int): Starting y-coordinate of the crop area.
            self.crop_rectangle = rectangle for crop display
            self.motion_event_id: motion event handler

        Methods:
            on_key_press(self, event): routing function for button functionality
            update_frame(value): Updates the displayed frame based on the slider value.
            display_frame(): Resets and displays the current frame on the canvas.
            start_crop(event): Handles the start of ROI selection.
            drag_crop(event): Dynamically draws a rectangle for the ROI during dragging.
            end_crop(event): Finalizes the ROI selection.
            cancel_crop(event): Resets the ROI selection.
            confirm_crop(event): Confirms the ROI selection and processes the cropped frame.
    """
    SCALE_FACTOR = 0.75

    def __init__(self, file_path, roi_callback=None):
        self.raw_video_file_path = file_path
        self.roi_callback = roi_callback
        self.cropped_frame = None
        self.frames = []
        self.current_frame_index = 0
        self.crop_start_y = None
        self.crop_end_y = None
        self.crop_rectangle = None
        self.motion_event_id = None

        # Load the TIFF file using PIL
        try:
            self.tiff_image = Image.open(self.raw_video_file_path)
            self.frames = [
                np.array(tracking.scale_frame(frame.copy(), Frame_Prep_Window.SCALE_FACTOR))
                for frame in ImageSequence.Iterator(self.tiff_image)
            ]
        except Exception as e:
            msg = "Load failed. Check console for details."
            error_popup(msg)
            print(f"Failed to load TIFF file: {e}")
            return

        # Create Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.2, top=0.85)  # Leave space for the slider and instructions

        # Add instruction text to the figure
        self.instruction_text = self.fig.text(
            0.5, 0.95,  # x, y in figure coordinates
            "Step 1: Select the region to crop (y-axis only). Drag to select, Enter to confirm.",
            ha='center', va='center', fontsize=10
        )

        # Create slider for frame selection using Matplotlib's Slider widget
        slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])  # position of the slider in figure coordinates
        self.slider = Slider(
            slider_ax, 'Frame', 0, len(self.frames) - 1, valinit=0, valstep=1
        )
        self.slider.on_changed(self.update_frame)

        # Display the initial frame
        self.update_frame(0)

        # Bind Matplotlib events for cropping
        self.fig.canvas.mpl_connect("button_press_event", self.start_crop)
        self.fig.canvas.mpl_connect("button_release_event", self.end_crop)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

        # Show the plot
        plt.show()

    def on_key_press(self, event):
        """Handle key press events."""
        if event.key == 'escape':
            self.cancel_crop()
            self.display_frame()  # Redraw the frame without the rectangle
        elif event.key == 'enter':
            self.confirm_crop()

    def update_frame(self, value):
        """Update the displayed frame based on slider value."""
        self.current_frame_index = int(value)
        self.display_frame()

    def display_frame(self):
        """Display the current frame in Matplotlib."""
        frame = self.frames[self.current_frame_index]
        self.ax.clear()
        self.ax.imshow(frame, cmap="gray")
        self.ax.set_title("Select region to crop (y-axis only).")
        self.ax.axis("off")
        self.fig.canvas.draw()

    def start_crop(self, event):
        """Begin cropping by recording the starting y-coordinate."""
        if event.inaxes == self.ax:
            self.cancel_crop()
            self.crop_start_y = event.ydata
            # Remove any existing rectangle
            if self.crop_rectangle:
                self.crop_rectangle.remove()
                self.crop_rectangle = None

            # Connect the motion event handler
            if not self.motion_event_id:
                self.motion_event_id = self.fig.canvas.mpl_connect("motion_notify_event", self.drag_crop)

    def drag_crop(self, event):
        """Draw a dynamic rectangle as the user drags."""
        if event.inaxes == self.ax and self.crop_start_y is not None:
            current_y = event.ydata

            # Remove the existing rectangle if present
            if self.crop_rectangle:
                self.crop_rectangle.remove()

            y_start = self.crop_start_y
            y_end = current_y
            height = y_end - y_start

            # Draw a new rectangle
            self.crop_rectangle = self.ax.add_patch(
                plt.Rectangle(
                    (0, y_start),
                    self.frames[self.current_frame_index].shape[1],  # Full width of the frame
                    height,
                    edgecolor="red",
                    facecolor="none",
                    linestyle="--",
                    linewidth=2,
                )
            )
            self.fig.canvas.draw()

    def end_crop(self, event):
        """Finalize the crop area by recording the ending y-coordinate."""
        if event.inaxes == self.ax and self.crop_start_y is not None:
            self.crop_end_y = event.ydata 

            # Disconnect the motion event handler to stop updating the rectangle
            if self.motion_event_id:
                self.fig.canvas.mpl_disconnect(self.motion_event_id)
                self.motion_event_id = None

            # Redraw the figure to ensure the rectangle stays
            self.fig.canvas.draw()

    def cancel_crop(self):
        """Cancel the cropping selection."""
        self.crop_start_y = None
        self.crop_end_y = None
        if self.crop_rectangle:
            self.crop_rectangle.remove()
            self.crop_rectangle = None
            self.fig.canvas.draw() 

    def confirm_crop(self):
        """Confirm the crop selection and finalize the cropped image."""
        if self.crop_start_y is not None and self.crop_end_y is not None:
            # Convert crop coordinates to the original image scale
            y_start, y_end = sorted((int(self.crop_start_y), int(self.crop_end_y)))
            current_frame = self.frames[self.current_frame_index]
            self.cropped_frame = current_frame[y_start:y_end, :]

            # Display the cropped frame
            self.ax.clear()
            self.ax.imshow(self.cropped_frame, cmap="gray")
            self.ax.set_title("Cropping complete. You may close the window.")
            self.ax.axis("off")
            self.fig.canvas.draw()

            # Hide the slider after cropping
            self.slider.ax.set_visible(False)
            self.fig.canvas.draw()

            # Call the ROI callback if provided
            if self.roi_callback:
                self.roi_callback((y_start, y_end, y_start, self.cropped_frame))
            else:
                print("No callback provided. ROI selection will not be returned.")

            # Unbind cropping events
            if self.motion_event_id:
                self.fig.canvas.mpl_disconnect(self.motion_event_id)
                self.motion_event_id = None

            # Optionally, you can close the figure here if desired
            # plt.close(self.fig)
        else:
            msg = "No crop area selected."
            error_popup(msg)

class Wavelength_Calibration_Window:
    """
    A tool for calibrating wavelength data by allowing users to crop a region of interest (ROI)
    and select specific wave lines for calibration.

    Purpose:
        This class provides an interactive interface for users to:
        1. Select a region of interest in an image by cropping (y-axis only).
        2. Analyze and display detected wave lines in the cropped region.
        3. Select wave lines for calibration and compute a calibration equation.

    parameters: 
        file_path (str): The path to the input image file (e.g., TIFF format).
        callback (function): A function to handle the computed calibration equation.

    returns: 
        Uses callback to return tuple of:
            - slope (float): The slope of the calibration line.
            - intercept (float): The intercept of the calibration line.
    """

    def __init__(self, input_file_path, callback):
        self.input_file_path = input_file_path
        self.callback = callback

        # Load the image as a PIL image
        self.image = Image.open(self.input_file_path)

        # State variables
        self.crop_start_y = None
        self.crop_end_y = None
        self.temp_crop_rectangle = None
        self.waves = None
        self.stage = 1
        self.scale_factor = 1
        self.cropped_image = None
        self.wave_x_avgs = []
        self.selected_waves = []
        self.num_waves = 3

        # Set up the Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.2, top=.85)  # Leave space for the slider

        # Add instruction text above the plot
        self.instruction_text = self.fig.text(
            0.5, 0.95,  # Centered horizontally, near the top of the figure
            "Step 1: Select the region to crop by clicking and dragging. Press Enter to confirm.",
            ha="center", va="center", fontsize=10
        )

        # Load and display the initial frame
        self.current_frame_index = 0
        self.update_frame(0)

        # Add a slider for frame selection if the image has multiple frames
        if hasattr(self.image, "n_frames"):
            slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
            self.slider = Slider(slider_ax, "Frame", 0, self.image.n_frames - 1, valinit=0, valstep=1)
            self.slider.on_changed(self.update_frame)
        else:
            self.slider = None

        # Connect Matplotlib events
        self.fig.canvas.mpl_connect("button_press_event", self.handle_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.drag_crop)
        self.fig.canvas.mpl_connect("button_release_event", self.end_crop)
        self.fig.canvas.mpl_connect("key_press_event", self.handle_key_press)

        # Show the plot
        plt.show()

    def handle_key_press(self, event):
        """Handles key press events for crop confirmation or cancellation."""
        if event.key == "enter":
            if self.stage == 1:
                self.confirm_crop()
            elif self.stage == 2 and len(self.selected_waves) == self.num_waves:
                self.calculate_transformation()
        elif event.key == "escape":
            if self.stage == 1:
                self.cancel_crop()
            elif self.stage == 2:
                self.cancel_selection()

    def handle_click(self, event):
        """Routes click events based on the current stage."""
        if self.stage == 1:
            self.click_start_crop(event)
        elif self.stage == 2:
            self.select_wave_click(event)

    def update_frame(self, value):
        """Updates the displayed frame based on the slider value."""
        self.current_frame_index = int(value)
        self.image.seek(self.current_frame_index)
        scaled_frame = self.scale_image(self.image)
        self.display_image(scaled_frame)

    def update_instructions(self, text):
        """Updates the instruction text dynamically."""
        self.instruction_text.set_text(text)
        self.fig.canvas.draw()

    def scale_image(self, image):
        """Scales the image by the specified factor."""
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        width, height = image.size
        return image.resize((int(width * self.scale_factor), int(height * self.scale_factor)), Image.LANCZOS)

    def display_image(self, image):
        """Displays the current frame in the Matplotlib axes."""
        self.ax.clear()
        self.ax.imshow(np.array(image), cmap="gray")
        self.ax.set_title(f"Frame {self.current_frame_index}")
        self.ax.axis("off")
        self.update_instructions("Step 1: Select the region to crop by clicking and dragging. Press Enter to confirm.")
        self.fig.canvas.draw()

    def click_start_crop(self, event):
        """Handles the start of crop selection."""
        if self.stage == 1 and event.inaxes == self.ax:
            self.cancel_crop()
            self.crop_start_y = event.ydata
            self.temp_crop_rectangle = None  # Clear any existing temporary crop
            self.fig.canvas.draw()

    def drag_crop(self, event):
        """Dynamically draws a rectangle to indicate the crop area during mouse movement."""
        if self.stage == 1 and self.crop_start_y is not None and self.crop_end_y is None and event.inaxes == self.ax:
            # Remove the previous temporary rectangle, if any
            if self.temp_crop_rectangle:
                self.temp_crop_rectangle.remove()
            temp_crop_end_y = event.ydata
            self.temp_crop_rectangle = self.ax.add_patch(
                plt.Rectangle(
                    (0, min(self.crop_start_y, temp_crop_end_y)),
                    self.image.width,  # Full image width
                    abs(temp_crop_end_y - self.crop_start_y),
                    edgecolor="red",
                    facecolor="none",
                    linestyle="-",
                    linewidth=1.5,
                )
            )
            self.fig.canvas.draw()

    def end_crop(self, event):
        """Finalizes the crop selection."""
        if self.stage == 1 and self.crop_start_y is not None and event.inaxes == self.ax:
            self.update_instructions("Press Enter to confirm or Esc to reset.")
            self.crop_end_y = event.ydata
            self.ax.axhline(y=self.crop_start_y, color="red", linestyle="-")
            self.ax.axhline(y=self.crop_end_y, color="red", linestyle="-")
            self.fig.canvas.draw() 

    def confirm_crop(self):
        """Confirms the crop selection and proceeds to wave analysis."""
        if self.crop_start_y is not None and self.crop_end_y is not None:
            self.update_instructions("Select 3 lines for calibration. Press enter when finished, or escape to restart.")
            y1, y2 = sorted((int(self.crop_start_y), int(self.crop_end_y)))
            self.image.seek(self.current_frame_index)
            cropped_frame = self.image.crop((0, y1, self.image.width, y2))
            self.cropped_image = cropped_frame
            self.stage = 2

            # Hide the slider
            if self.slider:
                self.slider.ax.set_visible(False)
                self.fig.canvas.draw()

            # Run wave analysis
            self.run_wave_detection(cropped_frame)
        else:
            self.update_instructions("No crop area selected. Please try again.") 

    def cancel_crop(self):
        """Cancels the cropping selection."""
        self.crop_start_y = None
        self.crop_end_y = None
        if self.temp_crop_rectangle:
            try:
                self.temp_crop_rectangle.remove()
            except ValueError:
                pass
            finally:
                self.temp_crop_rectangle = None
        self.ax.clear()
        self.update_frame(self.current_frame_index)

    def run_wave_detection(self, image):
        """Runs the wave analysis on the cropped image."""
        self.waves = tracking.analyze_and_append_waves(np.array(image), wave_threshold=110)
        self.display_waves()

    def display_waves(self):
        """Displays the waves over the cropped image using Matplotlib."""
        if self.waves and self.cropped_image is not None:
            self.ax.clear()
            self.ax.imshow(np.array(self.cropped_image), cmap="gray")

            for wave_index, wave in enumerate(self.waves):
                average_x = int(np.mean([point[1] for point in wave]))
                if average_x not in self.wave_x_avgs:
                    self.wave_x_avgs.append(average_x) 
                if 0 <= average_x < self.cropped_image.width:
                    self.ax.axvline(x=average_x, color="lime", linestyle="-")

            self.fig.canvas.draw()

    def cancel_selection(self):
        """Cancels the wave selection"""
        self.selected_waves = []
        self.update_overlay()

    def select_wave_click(self, event):
        """Handles wave selection via mouse clicks."""
        if len(self.selected_waves) < self.num_waves:
            x = event.xdata
            closest_wave = None
            min_distance = float("inf")

            for wave_x in self.wave_x_avgs:
                distance = abs(wave_x - x)
                if distance < min_distance:
                    min_distance = distance
                    closest_wave = wave_x

            if closest_wave is not None and closest_wave not in self.selected_waves:
                self.selected_waves.append(closest_wave)
                self.update_overlay()

    def update_overlay(self):
        """Redraws the waves and highlights selected waves."""
        self.ax.clear()
        self.ax.imshow(np.array(self.cropped_image), cmap="gray")
        for wave_x in self.wave_x_avgs:
            color = "red" if wave_x in self.selected_waves else "lime"
            self.ax.axvline(x=wave_x, color=color, linestyle="-")
        self.fig.canvas.draw()

    def calculate_transformation(self):
        """Calculates the calibration equation using the selected waves."""
        if len(self.selected_waves) < 3: 
            return
        x_values = sorted(self.selected_waves)
        y_values = [
            CalibrationValues.HG_GREEN.value,
            CalibrationValues.HG_YELLOW_1.value,
            CalibrationValues.HG_YELLOW_2.value,
        ]
        coefficients = np.polyfit(x_values, y_values, 1)
        calibration_equation = {"slope": coefficients[0], "intercept": coefficients[1]}
        print(calibration_equation)
        self.callback(calibration_equation)
        self.close_figure()

    def close_figure(self):
        """Closes the Matplotlib figure and cleans up resources."""
        plt.close(self.fig)  # Close the specific figure 

class Mica_Thickness_Calibration_Window:
    """
    A GUI-based tool for calibrating mica thickness using wave analysis from a TIFF file.

    This class allows users to:
    - Load a multi-frame TIFF file and navigate through its frames.
    - Dynamically crop a region of interest (ROI) using mouse interaction.
    - Analyze wave lines in the cropped region.
    - Select specific wave lines for calibration.
    - Calculate mica thickness using calibration parameters.

    Attributes:
        calibration_parameters (dict): Parameters for calibration, including slope and intercept.
        callback (function): Callback function to handle the calculated thickness value.
        selected_waves (list): List of x-coordinates for user-selected wave lines.
        crop_start_y (float or None): Starting y-coordinate of the crop area.
        crop_end_y (float or None): Ending y-coordinate of the crop area.
        temp_crop_rectangle (matplotlib.patches.Rectangle or None): Temporary rectangle for dynamic crop visualization.
        mode (str): Current mode of the tool, e.g., 'crop'.
        cropped_frame (numpy.ndarray or None): The cropped frame after ROI selection.
        scale_factor (float): Scaling factor for frame display.
        stage (int): Current stage of the workflow (1 for cropping, 2 for wave selection).
        selected_wavelengths (list): List of wavelengths corresponding to selected wave lines.
        image (PIL.Image): Loaded TIFF image for analysis.
        fig (matplotlib.figure.Figure): Matplotlib figure for displaying frames and ROI.
        ax (matplotlib.axes.Axes): Matplotlib axes for rendering the current frame.

    Methods:
        handle_key_press(event): Handles keyboard events for crop confirmation or cancellation.
        handle_click(event): Routes mouse click events based on the current stage.
        update_frame(value): Updates the displayed frame based on the slider value.
        update_instructions(text): Updates the instruction text displayed above the plot.
        scale_image(image): Scales the image by the specified factor.
        display_image(image): Displays the current frame in the Matplotlib axes.
        click_start_crop(event): Handles the start of ROI selection.
        drag_crop(event): Dynamically draws a rectangle for the ROI during dragging.
        end_crop(event): Finalizes the ROI selection.
        confirm_crop(): Confirms the crop selection and proceeds to wave analysis.
        cancel_crop(): Cancels the cropping selection.
        run_wave_detection(): Detects and analyzes wave lines in the cropped frame.
        display_filtered_waves(): Displays the cropped image with detected wave lines for selection.
        select_wave_click(event): Handles user selection of specific wave lines.
        update_overlay(): Redraws the overlay to highlight selected and unselected wave lines.
        convert_to_wavelengths(event): Converts selected x-coordinates to wavelengths using calibration parameters.
        calculate_thickness(): Calculates mica thickness based on selected wavelengths and calibration parameters.
    """
    def __init__(self, calibration_parameters, input_file_path, callback):
        self.calibration_parameters = calibration_parameters
        self.callback = callback
        self.selected_waves = []
        file_path = input_file_path

        # Set up crop area variables
        self.crop_start_y = None
        self.crop_end_y = None
        self.temp_crop_rectangle = None
        self.mode = 'crop'
        self.cropped_frame = None
        self.scale_factor = 0.75
        self.stage = 1

        self.selected_wavelengths = []
        
        # Load the image as a PIL image
        self.image = Image.open(file_path)
        
        # Set up the Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.2, top=0.85)  # Leave space for the slider

        # Add instruction text above the plot
        self.instruction_text = self.fig.text(
            0.5, 0.95,  # Centered horizontally, near the top of the figure
            "Step 1: Select the region to crop by clicking and dragging. Press Enter to confirm.",
            ha="center", va="center", fontsize=10
        )

        # Load and display the initial frame
        self.current_frame_index = 0
        self.update_frame(0)

        # Add a slider for frame selection if the image has multiple frames
        if hasattr(self.image, "n_frames") and self.image.n_frames > 1:
            slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
            self.slider = Slider(slider_ax, "Frame", 0, self.image.n_frames - 1, valinit=0, valstep=1)
            self.slider.on_changed(self.update_frame)
        else:
            self.slider = None

        # Connect Matplotlib events
        self.fig.canvas.mpl_connect("button_press_event", self.handle_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.drag_crop)
        self.fig.canvas.mpl_connect("button_release_event", self.end_crop)
        self.fig.canvas.mpl_connect("key_press_event", self.handle_key_press)

        # Show the plot
        plt.show()

    def handle_key_press(self, event):
        """Handles key press events for crop confirmation or cancellation."""
        if event.key == "enter":
            if self.stage == 1:
                self.confirm_crop()
            elif self.stage == 2 and len(self.selected_waves) == 2:
                self.convert_to_wavelengths()
        elif event.key == "escape":
            if self.stage == 1:
                self.cancel_crop()
            elif self.stage == 2:
                self.cancel_selection()

    def handle_click(self, event):
        """Routes click events based on the current stage."""
        if self.stage == 1:
            self.click_start_crop(event)
        elif self.stage == 2:
            self.select_wave_click(event)

    def update_frame(self, value):
        """Updates the displayed frame based on the slider value."""
        self.current_frame_index = int(value)
        self.image.seek(self.current_frame_index)
        scaled_frame = self.scale_image(self.image)
        self.display_image(scaled_frame)

    def update_instructions(self, text):
        """Updates the instruction text dynamically."""
        self.instruction_text.set_text(text)
        self.fig.canvas.draw()

    def scale_image(self, image):
        """Scales the image by the specified factor."""
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        width, height = image.size
        return image.resize((int(width * self.scale_factor), int(height * self.scale_factor)), Image.LANCZOS)

    def display_image(self, image):
        """Displays the current frame in the Matplotlib axes."""
        self.ax.clear()
        self.ax.imshow(np.array(image), cmap="gray")
        self.ax.set_title(f"Frame {self.current_frame_index}")
        self.ax.axis("off")
        self.update_instructions("Step 1: Select the region to crop by clicking and dragging. Press Enter to confirm.")
        self.fig.canvas.draw()

    def click_start_crop(self, event):
        """Handles the start of crop selection."""
        if self.stage == 1 and event.inaxes == self.ax:
            self.cancel_crop()
            self.crop_start_y = event.ydata
            self.temp_crop_rectangle = None  # Clear any existing temporary crop
            self.fig.canvas.draw()

    def drag_crop(self, event):
        """Dynamically draws a rectangle to indicate the crop area during mouse movement."""
        if self.stage == 1 and self.crop_start_y is not None and self.crop_end_y is None and event.inaxes == self.ax:
            # Remove the previous temporary rectangle, if any
            if self.temp_crop_rectangle:
                self.temp_crop_rectangle.remove()
            temp_crop_end_y = event.ydata
            self.temp_crop_rectangle = self.ax.add_patch(
                plt.Rectangle(
                    (0, min(self.crop_start_y, temp_crop_end_y)),
                    self.image.width,  # Full image width
                    abs(temp_crop_end_y - self.crop_start_y),
                    edgecolor="red",
                    facecolor="none",
                    linestyle="-",
                    linewidth=1.5,
                )
            )
            self.fig.canvas.draw()

    def end_crop(self, event):
        """Finalizes the crop selection."""
        if self.stage == 1 and self.crop_start_y is not None and event.inaxes == self.ax:
            self.update_instructions("Press Enter to confirm or Esc to reset.")
            self.crop_end_y = event.ydata
            self.ax.axhline(y=self.crop_start_y, color="red", linestyle="-")
            self.ax.axhline(y=self.crop_end_y, color="red", linestyle="-")
            self.fig.canvas.draw() 

    def confirm_crop(self):
        """Confirms the crop selection and proceeds to wave analysis."""
        if self.crop_start_y is not None and self.crop_end_y is not None:
            self.update_instructions("Select lines for calibration.")
            y1, y2 = sorted((int(self.crop_start_y), int(self.crop_end_y)))
            self.image.seek(self.current_frame_index)
            cropped_frame = self.image.crop((0, y1, self.image.width, y2))
            self.cropped_frame = cropped_frame
            self.stage = 2

            # Hide the slider
            if self.slider:
                self.slider.ax.set_visible(False)
                self.fig.canvas.draw()

            # Run wave analysis
            self.run_wave_detection(cropped_frame)
        else:
            self.update_instructions("No crop area selected. Please try again.") 

    def cancel_crop(self):
        """Cancels the cropping selection."""
        self.crop_start_y = None
        self.crop_end_y = None
        if self.temp_crop_rectangle:
            try:
                self.temp_crop_rectangle.remove()
            except ValueError:
                pass
            finally:
                self.temp_crop_rectangle = None
        self.ax.clear()
        self.update_frame(self.current_frame_index)

    def run_wave_detection(self, image):
        """Detect and filter wave lines, then allow user to select two for calibration."""
        # Convert the PIL image to a NumPy array
        image_array = np.array(image)
        
        # Normalize the image
        normalized_image = cv2.normalize(image_array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        normalized_image = normalized_image.astype(np.uint8)
        
        # Run wave detection on the normalized image
        self.waves = tracking.analyze_and_append_waves(
            normalized_image,
            wave_threshold=40,
            min_points_per_wave=10,
            min_wave_gap=10
        )
        
        # Proceed with filtering and displaying waves
        wave_averages = [np.mean([x for _, x in wave]) for wave in self.waves]
        
        # Filter out clustered wave averages
        filtered_averages = []
        for avg in sorted(wave_averages):
            if not filtered_averages or abs(filtered_averages[-1] - avg) > 5:
                filtered_averages.append(avg)
        
        self.filtered_averages = filtered_averages
        self.display_filtered_waves()

    def display_filtered_waves(self):
        """Display the cropped image with filtered wave averages overlaid for selection."""
        # Convert the cropped frame to a NumPy array
        image_array = np.array(self.cropped_frame)

        # Display the image using Matplotlib
        self.ax.clear()
        self.ax.imshow(image_array, cmap='gray')
        self.ax.axis('off')

        # Draw vertical lines at the positions of the wave averages
        for avg_x in self.filtered_averages:
            self.ax.axvline(x=avg_x, color='lime', linestyle='-')

        self.fig.canvas.draw()

    def select_wave_click(self, event):
        """Handles click events to select wave lines based on the average x-coordinate."""
        if event.inaxes != self.ax:
            return

        x = event.xdata

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
                self.update_instructions("Press Enter to confirm and calculate thickness.")
    
    def cancel_selection(self):
        """Cancels the wave selection."""
        self.selected_waves = []
        self.update_overlay()

    def update_overlay(self):
        """Redraw overlay with selected and unselected wave lines."""
        # Convert the cropped frame to a NumPy array
        image_array = np.array(self.cropped_frame)

        # Display the image using Matplotlib
        self.ax.clear()
        self.ax.imshow(image_array, cmap='gray')
        self.ax.axis('off')

        # Draw all wave lines, red for selected and green for unselected
        for avg_x in self.filtered_averages:
            color = 'lime'  # Green for unselected
            if avg_x in self.selected_waves:
                color = 'red'  # Red for selected
            self.ax.axvline(x=avg_x, color=color, linestyle='-')

        self.fig.canvas.draw()

    def convert_to_wavelengths(self, event=None):
        """Convert selected x-coordinates to wavelengths using calibration parameters."""
        # Ensure calibration parameters are defined
        if not self.calibration_parameters:
            msg = "Calibration parameters not provided."
            error_popup(msg)
            return
        
        # Calculate wavelengths using slope and intercept
        slope = self.calibration_parameters['slope']
        intercept = self.calibration_parameters['intercept']
        self.selected_wavelengths = [slope * (x) + intercept for x in self.selected_waves] 

        thickness = self.calculate_thickness()
        if thickness:
            self.callback(thickness)
        self.close_figure()

    def calculate_thickness(self):
        """
        Calculates mica thickness (T) using selected wavelengths and calibration parameters.
        Requires exactly two selected wavelengths stored in `self.selected_waves`.
        """
        if len(self.selected_wavelengths) != 2:
            msg = "Please select exactly two wave points for calibration."
            error_popup(msg)
            return None

        lambda_n_nm, lambda_n_minus_1_nm = self.selected_wavelengths
        lambda_n_angstrom = lambda_n_nm * 10
        lambda_n_minus_1_angstrom = lambda_n_minus_1_nm * 10

        mu_mica = 1.5757 + (5.89 * 10**5) / (lambda_n_angstrom ** 2)

        try:
            T = (lambda_n_angstrom * lambda_n_minus_1_angstrom) / (4 * (lambda_n_minus_1_angstrom - lambda_n_angstrom) * mu_mica)
            T_um = T / 10000 
            return T_um
        except ZeroDivisionError:
            msg = "Error: The selected wavelengths are too close, leading to division by zero."
            error_popup(msg)
            return None

    def close_figure(self):
        """Closes the Matplotlib figure and cleans up resources."""
        plt.close(self.fig)  # Close the specific figure 

class Motion_Analysis_Window:
    """
    A class to perform motion analysis on a TIFF image, allowing cropping, deletion, and wave analysis.
    Parameters:
        motion_profile_file_path (str): Path to the input data file (TIFF).
        calibration_parameters (dict): Dictionary containing slope and intercept for calibrated x-axis ticks.
        output_file_path (str): Path to save analysis results.
        offset_callback (function, optional): Callback function to handle offsets after cropping.
    """

    CROPPING_MODE = 'crop'
    DELETION_MODE = 'delete'
    FIGURE_SIZE = (12, 4)

    def __init__(self, motion_profile_file_path, calibration_parameters, output_file_path, offset_callback = None) -> None:
        self.y_offset = 0
        self.x_offset_start = 0
        self.x_offset_end = 0
        self.calibration_parameters = calibration_parameters
        self.offset_callback = offset_callback

        # Step 1: Request output file name
        self.output_filename = output_file_path

        # Step 2: Load the timelapse image (convert it to a NumPy array)
        self.timelapse_image = np.array(Image.open(motion_profile_file_path).convert('L'))  # Grayscale conversion
        self.file_path = motion_profile_file_path  # Store file path for saving

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
                msg = "No crop area selected."
                error_popup(msg)
        elif event.key == 'escape':
            self.cancel_crop()

    def handle_delete_keypress(self, event):
        """Handle key presses specifically for deletion mode."""
        if event.key == 'enter':
            self.confirm_deletion()
        elif event.key == 'escape':
            self.cancel_deletion()

    def confirm_crop(self): 
        """Confirms the crop selection and proceeds to wave analysis."""
        self.cropping_complete = True

        # Perform the crop
        x_start, x_end, y_start, y_end = self.crop_area
        self.x_offset_start = min(x_start, x_end)
        self.x_offset_end = max(x_start, x_end)
        self.y_offset = min(y_start, y_end) 
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

    def on_select_delete(self, eclick, erelease):
        """Callback for when the deletion rectangle is selected."""
        x_start, y_start = int(eclick.xdata), int(eclick.ydata)
        x_end, y_end = int(erelease.xdata), int(erelease.ydata)
        deletion_area = (x_start, x_end, y_start, y_end)
        self.deletion_areas.append(deletion_area) 

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

            # Generate calibrated CSV if calibration parameters exist
            if self.calibration_parameters:
                # Construct the calibrated filename in the same directory as the original file
                calibrated_filename = os.path.join(f"calibrated_{base_filename}")
                with open(calibrated_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Wave Index", "Frame Number", "Calibrated Center of Mass X Coord"])

                    for wave_idx, wave_line in enumerate(wave_lines):
                        for (y, x_center) in wave_line:
                            frame_number = y + self.y_offset  # Ensure consistency
                            calibrated_x = self.calibration_parameters['slope'] * (x_center - self.x_offset_start) + self.calibration_parameters['intercept']
                            writer.writerow([wave_idx + 1, frame_number, calibrated_x])

        except Exception as e:
            msg = "Error while saving. See console for details."
            error_popup(msg)
            print(f"Error saving wave centerlines to CSV: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SFA_FECO_UI(root)
    root.mainloop()