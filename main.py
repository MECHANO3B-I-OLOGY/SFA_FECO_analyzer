import csv
import cv2
import json
import matplotlib.backends._backend_tk as backend_tk
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
import shutil
import subprocess
import sys
import tkinter as tk
import _tkinter

from enums import CalibrationValues
from PIL import Image, ImageSequence, ImageTk
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.widgets import Cursor, RectangleSelector, Slider, SpanSelector
from screeninfo import get_monitors
from sys import platform
from tkinter import ttk, filedialog

import distance as dist
import tracking  
from exceptions import error_popup, warning_popup, checkbox_popup

# -------------------- Patch Tkinter configure --------------------
_original_configure = tk.Misc.configure

def configure_ignore_tclerror(self, cnf=None, **kw):
    try:
        return _original_configure(self, cnf, **kw)
    except _tkinter.TclError:
        # Ignore invalid command name errors (common during figure redraws)
        pass

tk.Misc.configure = configure_ignore_tclerror

# -------------------- Patch Matplotlib Toolbar update --------------------
_original_toolbar_update = backend_tk.NavigationToolbar2Tk.update

def toolbar_update_ignore_tclerror(self):
    try:
        _original_toolbar_update(self)
    except _tkinter.TclError:
        # Ignore errors caused by destroyed toolbar buttons
        pass

backend_tk.NavigationToolbar2Tk.update = toolbar_update_ignore_tclerror

def resource(path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, path)
    return path

class SFA_FECO_UI:
    """
        Main UI function for SFA FECO 
    """
    def __init__(self, root):
        self.root = root
        self.root.title("SFA FECO Analyzer")

        # Constants for window sizing and positioning
        self.DEFAULT_WIDTH_RATIO = 0.5
        self.DEFAULT_HEIGHT_RATIO = 0.75

        self.MAX_FILE_DISP_LENGTH = 20

        # Initialize file paths and parameters
        self.raw_video_file_path = None
        self.split_file_path = None
        self.data_file_path = None
        self.analyze_output_file_path = None
        self.motion_output_file_path = None

        self.roi_offset = 0
        self.analysis_x_offset = None
        self.analysis_y_offset = None
        self.wave_lines = None
        self.dispDistPairsIn = None
        self.dispDistPairsOut = None

        self.internal_flags = self.loadInternalJSON()

        self.calibration_values = self.load_calibration(self.internal_flags["auto_load_calibration"])

        self.wavelength_calibration_video_file_path = self.calibration_values["mercury_video_file"]
        self.thickness_input_file_path = self.calibration_values["thickness_video_file"]
        self.magnification_file = self.calibration_values["magnification_image_file"]
        self.radius_input_file_path = self.calibration_values["radius_video_file"]
        self.diameter_input_file_path = self.calibration_values["diameter_video_file"]

        self.mica_thickness = self.calibration_values["mica_thickness"]
        self.lambdaOdd = self.calibration_values["lambdas"]["odd"]
        self.lambdaEven = self.calibration_values["lambdas"]["even"]
        self.calibration_parameters = self.calibration_values["calibration_parameters"]
        self.scale = self.calibration_values["magnification_scale"]
        self.f = self.calibration_values["f_value"]
        self.radius = self.calibration_values["radius"]
        self.diameter = self.calibration_values["diameter"]
        self.split_frame_num = self.calibration_values["turnaround_frame"]

        self.javaExists = self.check_java()

        # Set protocol for window close to ensure full exit
        self.root.protocol("WM_DELETE_WINDOW", self.exit_application)

        self.root.geometry(self.internal_flags["geometry"])

        # Configure validation to accept only numbers
        vcmd = (root.register(self.validate_numeric_input), '%P')

        # === Add scrollable root frame ===
        self.canvas = tk.Canvas(self.root)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Scrollbars
        self.v_scroll = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.v_scroll.grid(row=0, column=1, sticky="ns")

        self.h_scroll = ttk.Scrollbar(self.root, orient="horizontal", command=self.canvas.xview)
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        # Make root window expandable
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Frame INSIDE the canvas that will hold your entire UI
        self.main_frame = ttk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.main_frame, anchor="nw")

        # Update scroll region whenever the frame changes size
        def update_scroll_region(event):
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

        self.main_frame.bind("<Configure>", update_scroll_region)

        # Setup styles
        self.setup_styles()

        # Configure grid layout for the root window
        for i in range(40):
            self.main_frame.grid_rowconfigure(i, weight=0)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=0)
        self.main_frame.grid_columnconfigure(2, weight=1)
        self.main_frame.grid_columnconfigure(3, weight=0)
        self.main_frame.grid_columnconfigure(4, weight=1)
        self.main_frame.grid_columnconfigure(5, weight=0)
        self.main_frame.grid_columnconfigure(6, weight=1)


        # region Subframe for Calibration
        # --- Calibration Load/Save Buttons and Checkbox ---
        # Frame to hold load/save buttons and checkbox together
        self.calibration_subframe = ttk.Frame(self.main_frame)
        self.calibration_subframe.grid(row=0, column=0, rowspan=2, sticky='ew', pady=(25,0))

        self.calibration_buttons_frame = ttk.Frame(self.calibration_subframe)
        self.calibration_buttons_frame.grid(row=0, column=0, sticky='w', padx=10, pady=(0, 0))

        # Buttons side by side
        self.load_calibration_button = ttk.Button(self.calibration_buttons_frame, text="Load Calibration", command=lambda: self.load_calibration(False, filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])))
        self.load_calibration_button.grid(row=0, column=0, padx=(0, 5), pady=0, sticky='w')

        self.save_calibration_button = ttk.Button(self.calibration_buttons_frame, text="Save Calibration", command=lambda: self.save_calibration(False))
        self.save_calibration_button.grid(row=0, column=1, padx=(5, 0), pady=0, sticky='w')

        # Checkbox under buttons, in the same frame
        self.load_previous_var = tk.BooleanVar()
        self.load_previous_var.set(self.internal_flags["auto_load_calibration"])
        self.load_previous_checkbox = ttk.Checkbutton(
            self.calibration_buttons_frame,
            text="Load previous calibration by default",
            variable=self.load_previous_var
        )
        self.load_previous_checkbox.grid(row=1, column=0, columnspan=2, sticky='w', pady=(2, 2))

        # Add a horizontal separator between rows
        calibrate_separator1 = ttk.Separator(self.calibration_subframe, orient="horizontal")
        calibrate_separator1.grid(row=2, column=0, sticky='ew', pady=10)

        # --- Step 1 label now moves down one row ---
        prep_label = ttk.Label(self.calibration_subframe, text="STEP 1: Dispersion Calibration", style='Step.TLabel', font=20)
        prep_label.grid(row=2+1, column=0, sticky='ew', padx=10)

        # Configure the column of the subframe to expand
        self.calibration_subframe.columnconfigure(0, weight=1)

        # Select wavelength calibration video file
        self.select_calibration_file_button = ttk.Button(self.calibration_subframe, text="Select Hg Lines (No Mica) Video", command=self.select_wavelength_calibration_file, style='Regular.TButton')
        self.select_calibration_file_button.grid(row=3+1, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the selected wavelength calibration video file's name
        self.wavelength_calibration_file_label = ttk.Label(self.calibration_subframe, text="No file selected", style='Regular.TLabel')
        self.wavelength_calibration_file_label.grid(row=4+1, column=0, sticky='ew', padx=10)

        # Checkbox frame (keeps them together neatly)
        self.checkbox_frame = ttk.Frame(self.calibration_subframe)
        self.checkbox_frame.grid(row=5+1, column=0, sticky='w', padx=10, pady=5)

        # Instruction label inside the frame
        self.checkbox_instruction = ttk.Label(
            self.checkbox_frame,
            text="If only using 2 Hg lines, select which you will use:",
            style='Regular.TLabel'
        )
        self.checkbox_instruction.grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 5))

        # Variables for checkboxes
        self.green_var = tk.BooleanVar()
        self.yellow1_var = tk.BooleanVar()
        self.yellow2_var = tk.BooleanVar()

        self.green_var.set(self.calibration_values["mercury_lines"]["green"])
        self.yellow1_var.set(self.calibration_values["mercury_lines"]["yellow_1"])
        self.yellow2_var.set(self.calibration_values["mercury_lines"]["yellow_2"])

        # Checkboxes
        self.green_check = ttk.Checkbutton(self.checkbox_frame, text="Green", variable=self.green_var)
        self.green_check.grid(row=1, column=0, sticky='w', padx=(0, 10))

        self.yellow1_check = ttk.Checkbutton(self.checkbox_frame, text="Yellow 1", variable=self.yellow1_var)
        self.yellow1_check.grid(row=1, column=1, sticky='w', padx=(0, 10))

        self.yellow2_check = ttk.Checkbutton(self.checkbox_frame, text="Yellow 2", variable=self.yellow2_var)
        self.yellow2_check.grid(row=1, column=2, sticky='w')

        # Calibrate Wavelengths button
        self.execute_wavelength_calibration = ttk.Button(self.calibration_subframe, text="Calibrate Wavelengths", command=self.run_wavelength_calibration, style='Regular.TButton')
        self.execute_wavelength_calibration.grid(row=6+1, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the wavelength calibration status
        self.calibration_completion_label = ttk.Label(self.calibration_subframe, text="Calibration not completed", style='Regular.TLabel')
        self.calibration_completion_label.grid(row=7+1, column=0, sticky='new', padx=10, pady=(0, 20))

         # Subframe for dispersion values
        self.dispersion_frame = ttk.Frame(self.calibration_subframe)
        self.dispersion_frame.grid(row=self.calibration_subframe.grid_size()[1], column=0, sticky='ew', padx=10, pady=(5, 20))

        # Dispersion 1
        self.dispersion1_label = ttk.Label(self.dispersion_frame, text="Dispersion 1:", style='Regular.TLabel')
        self.dispersion1_label.grid(row=0, column=0, sticky='w', pady=2, padx=(0, 5))
        self.dispersion1_entry = ttk.Entry(self.dispersion_frame, width=15)
        self.dispersion1_entry.grid(row=0, column=1, sticky='w', pady=2)

        # Dispersion 2
        self.dispersion2_label = ttk.Label(self.dispersion_frame, text="Dispersion 2:", style='Regular.TLabel')
        self.dispersion2_label.grid(row=1, column=0, sticky='w', pady=2, padx=(0, 5))

        self.dispersion2_entry = ttk.Entry(self.dispersion_frame, width=15)
        self.dispersion2_entry.grid(row=1, column=1, sticky='w', pady=2)

        # Dispersion 3
        self.dispersion3_label = ttk.Label(self.dispersion_frame, text="Dispersion 3:", style='Regular.TLabel')
        self.dispersion3_label.grid(row=2, column=0, sticky='w', pady=2, padx=(0, 5))

        self.dispersion3_entry = ttk.Entry(self.dispersion_frame, width=15)
        self.dispersion3_entry.grid(row=2, column=1, sticky='w', pady=2)

        self.dispersionAvg_label = ttk.Label(self.dispersion_frame, text="Average Dispersion:", style='Regular.TLabel')
        self.dispersionAvg_label.grid(row=3, column=0, sticky='w', pady=2, padx=(0, 5))

        self.dispersionAvg_entry = ttk.Entry(self.dispersion_frame, width=15)
        self.dispersionAvg_entry.grid(row=3, column=1, sticky='w', pady=2)

        self.dispersionStd_label = ttk.Label(self.dispersion_frame, text="Standard Deviation:", style='Regular.TLabel')
        self.dispersionStd_label.grid(row=4, column=0, sticky='w', pady=2, padx=(0, 5))

        self.dispersionStd_entry = ttk.Entry(self.dispersion_frame, width=15)
        self.dispersionStd_entry.grid(row=4, column=1, sticky='w', pady=2)

        
        
        self.dispersion1_entry.config(state="readonly")
        self.dispersion2_entry.config(state="readonly")
        self.dispersion3_entry.config(state="readonly")
        self.dispersionAvg_entry.config(state="readonly")
        self.dispersionStd_entry.config(state="readonly")
        
        # Add a horizontal separator between rows
        calibrate_separator1 = ttk.Separator(self.calibration_subframe, orient="horizontal")
        calibrate_separator1.grid(row=9+1, column=0, sticky='ew', pady=10)

        # Select Thickness File button
        self.select_thickness_file_button = ttk.Button(self.calibration_subframe, text="Select Mica Mica Contact (No Hg) Video", command=self.select_thickness_file, style='Regular.TButton')
        self.select_thickness_file_button.grid(row=10+1, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the selected thickness file's name
        self.thickness_file_label = ttk.Label(self.calibration_subframe, text="No file selected", style='Regular.TLabel')
        self.thickness_file_label.grid(row=11+1, column=0, sticky='new', padx=10)

        # Calibrate Thickness button
        self.execute_thickness_calibration = ttk.Button(self.calibration_subframe, text="Calculate Mica Thickness", command=self.run_thickness_calibration, style='Regular.TButton')
        self.execute_thickness_calibration.grid(row=12+1, column=0, sticky='ew', padx=10, pady=5)

        # Frame to hold mica thickness label and entry side by side
        self.thickness_frame = ttk.Frame(self.calibration_subframe)
        self.thickness_frame.grid(row=13+1, column=0, sticky='w', padx=10, pady=(5, 5))

        self.calibration_thickness_label = ttk.Label(self.thickness_frame, text="Mica thickness (μm):", style='Regular.TLabel')
        self.calibration_thickness_label.grid(row=0, column=0, sticky='w', padx=(0, 5))

        self.thickness_display = tk.Entry(self.thickness_frame, width=10)
        self.thickness_display.grid(row=0, column=1, sticky='w')

        # Insert and disable
        self.thickness_display.insert(0, str(self.mica_thickness))

        self.fringe_frame = ttk.Frame(self.calibration_subframe)
        self.fringe_frame.grid(row=15+1, column=0, sticky='ew', padx=10, pady=(5, 10))

        self.fringe_label = ttk.Label(self.fringe_frame, text="Fringe number (n):", style='Regular.TLabel')
        self.fringe_label.grid(row=0, column=0, sticky='w', padx=(0, 5))

        self.fringe_entry = ttk.Entry(self.fringe_frame, width=15)
        self.fringe_entry.grid(row=0, column=1, sticky='w')

        self.fringe_entry.insert(0, str(self.calibration_values["fringe_number"]))

        # Add a horizontal separator between rows


        # Add a vertical separator between columns
        vertical_separator = ttk.Separator(self.main_frame, orient="vertical")
        vertical_separator.grid(row=0, column=1, rowspan=7, sticky='ns', padx=10)

        # region Step 1: Prep
        
        # Subframe for Raw Video selection
        self.radius_subframe = ttk.Frame(self.main_frame)
        self.radius_subframe.grid(row=0, column=2, sticky='new', pady=(25,0))

        prep_label = ttk.Label(self.radius_subframe, text="STEP 2: Radius of Curvature Calibration", style='Step.TLabel', font=20)
        prep_label.grid(row=0, column=0, sticky='ew', padx=10)

        setupSelectorFrame = ttk.Frame(self.radius_subframe)
        setupSelectorFrame.grid(row=1, column=0, sticky='ew', padx=10, pady=(5, 10))

        self.setupLabel = ttk.Label(setupSelectorFrame, text="Setup: ", style='Regular.TLabel')
        self.setupLabel.pack(side='left', padx=(0, 5))

        self.setupEntryVar = tk.StringVar(value=str(self.calibration_values["setup"])) 
        self.setupEntry = ttk.Entry(setupSelectorFrame, textvariable=self.setupEntryVar, width=25, style='Regular.TEntry')
        self.setupEntry.pack(side='left', padx=(0, 5))
        self.setupEntry.config(state="readonly")

        self.setupButton = ttk.Button(
            setupSelectorFrame,
            text="Select Setup",
            command=self.selectSetupWindow,
            style='Regular.TButton',
            width=15
        )
        self.setupButton.pack(fill='x', pady=5)
        
        # --- Magnification Calibration Section ---
        self.select_magnification_image_button = ttk.Button(
            self.radius_subframe,
            text="Select Magnification Image",
            command=self.select_magnification_file,  # define this method later
            style='Regular.TButton'
        )
        self.select_magnification_image_button.grid(row=2, column=0, sticky='ew', padx=10, pady=5)

        # Label to display selected magnification image file
        self.magnification_image_label = ttk.Label(
            self.radius_subframe,
            text="No file selected",
            style='Regular.TLabel'
        )
        self.magnification_image_label.grid(row=3, column=0, sticky='new', padx=10)

        # Frame for f value label and entry side by side
        self.scale_frame = ttk.Frame(self.radius_subframe)
        self.scale_frame.grid(row=4, column=0, sticky='w', padx=10, pady=(5, 5))

        self.scale_label = ttk.Label(self.scale_frame, text=r"scale (μm/tick):", style='Regular.TLabel')
        self.scale_label.grid(row=0, column=0, sticky='w', padx=(0, 5))

        self.scale_display = ttk.Entry(self.scale_frame, width=15, validate='key')
        self.scale_display.grid(row=0, column=1, sticky='w')

        self.scale_display.insert(0, str(self.scale))

        # Button to calculate magnification factor
        self.calculate_magnification_button = ttk.Button(
            self.radius_subframe,
            text="Calculate Magnification Factor",
            command=self.calculate_magnification_factor,  # define this method later
            style='Regular.TButton'
        )
        self.calculate_magnification_button.grid(row=5, column=0, sticky='ew', padx=10, pady=5)

        # Frame for f value label and entry side by side
        self.f_frame = ttk.Frame(self.radius_subframe)
        self.f_frame.grid(row=6, column=0, sticky='w', padx=10, pady=(5, 5))

        self.calibration_f_label = ttk.Label(self.f_frame, text=r"f value (px/μm):", style='Regular.TLabel')
        self.calibration_f_label.grid(row=0, column=0, sticky='w', padx=(0, 5))

        self.f_display = ttk.Entry(self.f_frame, width=15, validate='key')
        self.f_display.grid(row=0, column=1, sticky='w')

        self.f_display.insert(0, str(self.f))

        # Select radius File button
        self.select_radius_file_button = ttk.Button(self.radius_subframe, text="Select Radius of Curvature Calibration Video", command=self.select_radius_file, style='Regular.TButton')
        self.select_radius_file_button.grid(row=7, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the selected radius file's name
        self.radius_file_label = ttk.Label(self.radius_subframe, text="No file selected", style='Regular.TLabel')
        self.radius_file_label.grid(row=8, column=0, sticky='new', padx=10)


        # Calibrate radius button
        self.execute_radius_calibration = ttk.Button(self.radius_subframe, text="Find Radius", command=self.run_radius_calibration, style='Regular.TButton')
        self.execute_radius_calibration.grid(row=9, column=0, sticky='ew', padx=10, pady=5)

        # Frame for radius label and entry side by side
        self.radius_frame = ttk.Frame(self.radius_subframe)
        self.radius_frame.grid(row=10, column=0, sticky='w', padx=10, pady=(5, 5))

        self.calibration_radius_label = ttk.Label(self.radius_frame, text=r"Radius of Curvature (m):", style='Regular.TLabel')
        self.calibration_radius_label.grid(row=0, column=0, sticky='w', padx=(0, 5))

        self.radius_display = ttk.Entry(self.radius_frame, width=15, textvariable=tk.StringVar(value=str(self.radius)), validate='key')
        self.radius_display.grid(row=0, column=1, sticky='w')

        self.radius_display.insert(0, str(self.radius))

        # Add a horizontal separator between rows
        calibrate_separator1 = ttk.Separator(self.radius_subframe, orient="horizontal")
        calibrate_separator1.grid(row=11, column=0, sticky='ew', pady=10)

        prep_label = ttk.Label(self.radius_subframe, text="Optional: Diameter Calculation", style='Step.TLabel', font=20)
        prep_label.grid(row=12, column=0, sticky='ew', padx=10)

        # Select radius File button
        self.select_diameter_file_button = ttk.Button(self.radius_subframe, text="Select Fringe Diameter Calculation Video", command=self.select_diameter_file, style='Regular.TButton')
        self.select_diameter_file_button.grid(row=13, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the selected radius file's name
        self.diameter_file_label = ttk.Label(self.radius_subframe, text="No file selected", style='Regular.TLabel')
        self.diameter_file_label.grid(row=14, column=0, sticky='new', padx=10)

        # Calibrate radius button
        self.execute_diameter_calibration = ttk.Button(self.radius_subframe, text="Find Diameter", command=self.run_diameter_calibration, style='Regular.TButton')
        self.execute_diameter_calibration.grid(row=15, column=0, sticky='ew', padx=10, pady=5)

        # Frame for radius label and entry side by side
        self.diameter_frame = ttk.Frame(self.radius_subframe)
        self.diameter_frame.grid(row=16, column=0, sticky='w', padx=10, pady=(5, 5))

        self.calibration_diameter_label = ttk.Label(self.diameter_frame, text=r"Fridge Diameter (μm):", style='Regular.TLabel')
        self.calibration_diameter_label.grid(row=0, column=0, sticky='w', padx=(0, 5))

        self.diameter_display = ttk.Entry(self.diameter_frame, width=15, textvariable=str(self.diameter), validate='key')
        self.diameter_display.grid(row=0, column=1, sticky='w')

        self.diameter_display.insert(0, str(self.diameter))
        
        # endregion
        
        # Add a vertical separator between columns
        vertical_separator = ttk.Separator(self.main_frame, orient="vertical")
        vertical_separator.grid(row=0, column=3, rowspan=7, sticky='ns', padx=10)

        # region Step 1: Prep
        
        # Subframe for Raw Video selection
        self.raw_video_subframe = ttk.Frame(self.main_frame)
        self.raw_video_subframe.grid(row=0, column=4, sticky='new', pady=(25,0))

        prep_label = ttk.Label(self.raw_video_subframe, text="STEP 3: Prep", style='Step.TLabel', font=20)
        prep_label.grid(row=0, column=0, sticky='ew', padx=10)

        self.raw_video_subframe.columnconfigure(0, weight=1)

        # Raw video data select button
        self.select_raw_button = ttk.Button(self.raw_video_subframe, text="Select Video for Distance Calculation", command=self.select_raw_video, style='Regular.TButton')
        self.select_raw_button.grid(row=1, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the selected file's name
        self.raw_file_label = ttk.Label(self.raw_video_subframe, text="No file selected", style='Regular.TLabel')
        self.raw_file_label.grid(row=2, column=0, sticky='ew', padx=10)

        # Crop/Preprocess button
        self.crop_button = ttk.Button(self.raw_video_subframe, text="Crop", command=self.open_crop_preprocess_window, style='Regular.TButton')
        self.crop_button.grid(row=3, column=0, sticky='ew', padx=10, pady=5)

        calibrate_separator1 = ttk.Separator(self.raw_video_subframe, orient="horizontal")
        calibrate_separator1.grid(row=4, column=0, sticky='ew', pady=10)

        # Subframe for Generate Motion Profile
        self.motion_profile_subframe = ttk.Frame(self.raw_video_subframe)
        self.motion_profile_subframe.grid(row=5, column=0, sticky='ew')

        self.motion_profile_subframe.columnconfigure(0, weight=1)        

        # Subframe for motion output file selection
        self.motion_output_subframe = ttk.Frame(self.motion_profile_subframe)
        self.motion_output_subframe.grid(row=0, column=0, sticky='ew', padx=10, pady=5)

        # Label for the motion output file
        self.motion_output_label = ttk.Label(self.motion_output_subframe, text="Output File:")
        self.motion_output_label.grid(row=0, column=0, sticky='w')

        # Textbox for file name entry
        self.motion_output_file_var = tk.StringVar()
        self.motion_output_file_var.set("motion_profile_output")  # Default file name
        self.motion_output_entry = ttk.Entry(
            self.motion_output_subframe, 
            textvariable=self.motion_output_file_var, 
            width=30,  # Adjust width as needed
            style='Regular.TEntry'
        )
        self.motion_output_entry.grid(row=0, column=1, sticky='ew') 

        # Generate motion profile button
        self.generate_motion_button = ttk.Button(self.motion_profile_subframe, text="Generate Motion Profile", command=self.generate_motion_profile, style='Regular.TButton')
        self.generate_motion_button.grid(row=2, column=0, sticky='ew', padx=10, pady=5)

        # Frame for radio buttons
        self.mode_var = tk.StringVar(value="singlet")  # default selection
        self.mode_frame = ttk.Frame(self.motion_profile_subframe)
        self.mode_frame.grid(row=1, column=0, sticky='w', padx=10, pady=5)

        self.singlet_radio = ttk.Radiobutton(
            self.mode_frame, text="Singlet",
            variable=self.mode_var, value="singlet"
        )
        self.singlet_radio.grid(row=0, column=0, sticky='w')

        self.doublet_radio = ttk.Radiobutton(
            self.mode_frame, text="Doublet",
            variable=self.mode_var, value="doublet"
        )
        self.doublet_radio.grid(row=0, column=1, sticky='w')
        # endregion

        calibrate_separator1 = ttk.Separator(self.raw_video_subframe, orient="horizontal")
        calibrate_separator1.grid(row=6, column=0, sticky='ew', pady=10)

        # region Step 2: Analyze
        step3_label = ttk.Label(self.raw_video_subframe, text="STEP 4: Analyze", style='Step.TLabel', font=20)
        step3_label.grid(row=7, column=0, sticky='ew', padx=10)

        # region Subframe for Data File Selection
        self.motion_profile_file_subframe = ttk.Frame(self.raw_video_subframe)
        self.motion_profile_file_subframe.grid(row=8, column=0, sticky='ew')

        self.motion_profile_file_subframe.columnconfigure(0, weight=1)

        # Button to choose an existing data file
        self.choose_motion_profile_file_button = ttk.Button(self.motion_profile_file_subframe, text="Choose Motion Profile File", command=self.select_analysis_input_image_file, style='Regular.TButton')
        self.choose_motion_profile_file_button.grid(row=0, column=0, sticky='ew', padx=10, pady=5)

        # File field for the data file
        self.motion_profile_file_label = ttk.Label(self.motion_profile_file_subframe, text="No file selected", style='Regular.TLabel')
        self.motion_profile_file_label.grid(row=1, column=0, sticky='enw', padx=10, pady=(0, 10))
        # endregion

        # region Subframe for Analyze
        self.analyze_subframe = ttk.Frame(self.raw_video_subframe)
        self.analyze_subframe.grid(row=10, column=0, sticky='ew')

        self.analyze_subframe.columnconfigure(0, weight=1)

        # Output file selection textbox for Analyzes
        self.analyze_output_file_frame = ttk.Frame(self.analyze_subframe)
        self.analyze_output_file_label = ttk.Label(self.analyze_output_file_frame, text="Output File Name:", style='Regular.TLabel')
        self.analyze_output_file_label.pack(side="left", padx=(0, 5))

        self.analyze_output_file_var = tk.StringVar(value="analysis_output")
        self.analyze_output_file_textbox = ttk.Entry(self.analyze_output_file_frame, textvariable=self.analyze_output_file_var, style='Regular.TEntry')
        self.analyze_output_file_textbox.pack(side="left", fill="x", expand=True)

        self.analyze_output_file_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        # Analyze button
        self.analyze_button = ttk.Button(self.analyze_subframe, text="Analyze", command=self.analyze, style='Regular.TButton')
        self.analyze_button.grid(row=2, column=0, sticky='ew', padx=10, pady=(0,20))
        '''
        # Estimate Turnaround button
        self.estimate_turnaround_button = ttk.Button(self.analyze_subframe, text="Estimate Turnaround of Output", command=self.estimate_turnaround, style='Regular.TButton')
        self.estimate_turnaround_button.grid(row=3, column=0, sticky='ew', padx=10, pady=5)
        '''
        # endregion

        # region Subframe for Split functionality
        self.split_subframe = ttk.Frame(self.raw_video_subframe)
        self.split_subframe.grid(row=11, column=0, sticky='ew')

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

        # Add a vertical separator between columns
        vertical_separator = ttk.Separator(self.main_frame, orient="vertical")
        vertical_separator.grid(row=0, column=5, rowspan=7, sticky='ns', padx=10)

        # Subframe for visualization controls
        self.visualize_subframe = ttk.Frame(self.main_frame)
        self.visualize_subframe.grid(row=0, column=6, sticky='new', padx=10, pady=25)

        # region Step 3: Visualize
        step4_label = ttk.Label(self.visualize_subframe, text="STEP 5: Visualize", style='Step.TLabel', font=20)
        step4_label.pack(anchor='w', pady=(0, 5))

        # --- Camera FPS row ---
        fps_frame = ttk.Frame(self.visualize_subframe)
        fps_frame.pack(anchor='w', pady=(0, 5))

        self.camera_fps_label = ttk.Label(fps_frame, text="Camera FPS:", style='Regular.TLabel')
        self.camera_fps_label.pack(side='left', padx=(0, 5))

        self.camera_fps_var = tk.StringVar(value=str(self.calibration_values["fps"]))
        self.camera_fps_entry = ttk.Entry(fps_frame, textvariable=self.camera_fps_var, width=10, style='Regular.TEntry')
        self.camera_fps_entry.pack(side='left')

        # --- Radio buttons row ---
        radio_frame = ttk.Frame(self.visualize_subframe)
        radio_frame.pack(anchor='w', pady=(0, 8))

        self.visualize_mode = tk.StringVar(value="in")  # default mode

        self.radio_in = ttk.Radiobutton(
            radio_frame, text="In Run", variable=self.visualize_mode, value="in", style='Regular.TRadiobutton'
        )
        self.radio_in.pack(side='left', padx=(0, 10))

        self.radio_out = ttk.Radiobutton(
            radio_frame, text="Out Run", variable=self.visualize_mode, value="out", style='Regular.TRadiobutton'
        )
        self.radio_out.pack(side='left', padx=(0, 10))

        # --- Single visualize button ---
        self.visualize_button = ttk.Button(
            self.visualize_subframe,
            text="Visualize Distance Over Time",
            command=self.visualize_distance_over_time,
            style='Regular.TButton'
        )
        self.visualize_button.pack(fill='x', pady=5)

         # --- Buffer line ---
        ttk.Separator(self.visualize_subframe, orient='horizontal').pack(fill='x', pady=5)

        # --- Input k row ---
        k_frame = ttk.Frame(self.visualize_subframe)
        k_frame.pack(anchor='w', pady=(5, 5))

        self.k_label = ttk.Label(k_frame, text="Input k (mN/m):", style='Regular.TLabel')
        self.k_label.pack(side='left', padx=(0, 5))

        self.k_var = tk.StringVar(value=str(self.calibration_values["spring_constant"]))
        self.k_entry = ttk.Entry(k_frame, textvariable=self.k_var, width=10, style='Regular.TEntry')
        self.k_entry.pack(side='left')

        # --- Radio buttons for force visualization ---
        force_radio_frame = ttk.Frame(self.visualize_subframe)
        force_radio_frame.pack(anchor='w', pady=(0, 8))

        self.force_mode = tk.StringVar(value="in")  # separate variable for force visualization

        self.force_radio_in = ttk.Radiobutton(
            force_radio_frame, text="In Run", variable=self.force_mode, value="in", style='Regular.TRadiobutton'
        )
        self.force_radio_in.pack(side='left', padx=(0, 10))

        self.force_radio_out = ttk.Radiobutton(
            force_radio_frame, text="Out Run", variable=self.force_mode, value="out", style='Regular.TRadiobutton'
        )
        self.force_radio_out.pack(side='left', padx=(0, 10))

        self.force_radio_full = ttk.Radiobutton(
            force_radio_frame, text="Full Run", variable=self.force_mode, value="full", style='Regular.TRadiobutton'
        )
        self.force_radio_full.pack(side='left', padx=(0, 10))

        # --- Radio buttons for log/linear y ---
        linear_radio_frame = ttk.Frame(self.visualize_subframe)
        linear_radio_frame.pack(anchor='w', pady=(0, 8))

        self.scale_mode = tk.StringVar(value="linear")  # separate variable for force visualization

        self.force_radio_linear = ttk.Radiobutton(
            linear_radio_frame, text="Linear Plot", variable=self.scale_mode, value="linear", style='Regular.TRadiobutton'
        )
        self.force_radio_linear.pack(side='left', padx=(0, 10))

        self.force_radio_log = ttk.Radiobutton(
            linear_radio_frame, text="Semilog Plot", variable=self.scale_mode, value="log", style='Regular.TRadiobutton'
        )
        self.force_radio_log.pack(side='left', padx=(0, 10))

        # --- Visualize Force button ---
        self.visualize_force_button = ttk.Button(
            self.visualize_subframe,
            text="Visualize Force Over Distance",
            command=self.visualize_force_over_distance,  # attach your function here
            style='Regular.TButton'
        )
        self.visualize_force_button.pack(fill='x', pady=5)
        
        # endregion

        # Ensure all 4 main regions (columns 0, 2, 4, 6) have equal horizontal weight
        for col in [0, 2, 4]:
            self.main_frame.grid_columnconfigure(col, weight=1, uniform="region")

        # Optionally, make separators take minimal space
        for sep_col in [1, 3]:
            self.main_frame.grid_columnconfigure(sep_col, weight=0)

        self.javaPrompt()

        self.updateScreenValues()

    def exit_application(self):
        """Cleanly exit the application."""

        geometry = self.root.geometry()
        self.internal_flags.update({"geometry": geometry})

        loadCali = self.load_previous_var.get()
        self.internal_flags.update({"auto_load_calibration": loadCali})

        cache_dir = os.path.join(os.getcwd(), "cache")
        flag_path = os.path.join(cache_dir, "internal.json")
        with open(flag_path, "w") as f:
            json.dump(self.internal_flags, f)

        self.save_calibration(True)

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
        self.btn_style.configure('TCheckbutton', padding=(0, 0, 0, 0))
    
    def select_wavelength_calibration_file(self):
        """
            Function for selecting wavelength calibration input file. Checks for validity and updates label. 
        """
        # Open a file dialog to select a TIFF file

        file_path = self.getFile()

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

    

    def setDispersionEntries(self, d1,d2,d3,avg,std):
        def exponentFormatting(num):
            s = f"{num:.10g}"  
            if 'e' in s or 'E' in s:
                mantissa, exp = s.split('e')
                mantissa = mantissa[:6]
                return f"{mantissa}e{exp}"
            else:
                return s[:6]

        self.dispersion1_entry.config(state="normal")
        self.dispersion2_entry.config(state="normal")
        self.dispersion3_entry.config(state="normal")
        self.dispersionAvg_entry.config(state="normal")
        self.dispersionStd_entry.config(state="normal")

        self.dispersion1_entry.delete(0, "end")
        self.dispersion1_entry.insert(0, exponentFormatting(d1))

        self.dispersion2_entry.delete(0, "end")
        self.dispersion2_entry.insert(0, exponentFormatting(d2))

        self.dispersion3_entry.delete(0, "end")
        self.dispersion3_entry.insert(0, exponentFormatting(d3))

        self.dispersionAvg_entry.delete(0, "end")
        self.dispersionAvg_entry.insert(0, exponentFormatting(avg))

        self.dispersionStd_entry.delete(0, "end")
        self.dispersionStd_entry.insert(0, exponentFormatting(std))

        self.dispersion1_entry.config(state="readonly")
        self.dispersion2_entry.config(state="readonly")
        self.dispersion3_entry.config(state="readonly")
        self.dispersionAvg_entry.config(state="readonly")
        self.dispersionStd_entry.config(state="readonly")

    def select_thickness_file(self):
        """Select input file for Calibrate Thickness. Updates label."""
        self.thickness_input_file_path = self.getFile()
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
        self.thickness_display.delete(0, "end")
        self.thickness_display.insert(0, str(abs(thickness))[:4])
        
        # Disable the widget again to make it read-only
        self.thickness_display.config(state="readonly")

    def select_magnification_file(self):
        """Select input file for Calibrate Thickness. Updates label."""
        self.magnification_file = self.getFile()
        if self.magnification_file:

            if len(self.magnification_file) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.magnification_file[len(self.magnification_file) - self.MAX_FILE_DISP_LENGTH:]
                self.magnification_image_label.config(text=data_file_text)
            else:
                self.magnification_image_label.config(text=self.magnification_file) 
    
    def calculate_magnification_factor(self):
        if(not self.magnification_file): 
            msg = "Please select an input file"
            error_popup(msg)
            return
        Magnification_Calculation_Window(self.magnification_file, float(self.scale_display.get()))

    def select_radius_file(self):
        """Select input file for radius. Updates label."""
        self.radius_input_file_path = self.getFile()
        if self.radius_input_file_path:

            if len(self.radius_input_file_path) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.radius_input_file_path[len(self.radius_input_file_path) - self.MAX_FILE_DISP_LENGTH:]
                self.radius_file_label.config(text=data_file_text)
            else:
                self.radius_file_label.config(text=self.radius_input_file_path) 

    def select_diameter_file(self):
        """Select input file for radius. Updates label."""
        self.diameter_input_file_path = self.getFile()
        if self.diameter_input_file_path:

            if len(self.diameter_input_file_path) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.diameter_input_file_path[len(self.diameter_input_file_path) - self.MAX_FILE_DISP_LENGTH:]
                self.diameter_file_label.config(text=data_file_text)
            else:
                self.diameter_file_label.config(text=self.diameter_input_file_path) 

    def run_radius_calibration(self): 
        with open("setups.json", "r") as f:
            temp = json.load(f)
        RadiusMeasurementWindow(self.radius_input_file_path, self.f, self.calibration_parameters, [self.lambdaOdd, self.lambdaEven], int(self.fringe_entry.get()), temp[str(self.setupEntryVar.get())][2], self.callback_radius)

    def run_diameter_calibration(self): 
        Fringe_Diameter_Window(self.diameter_input_file_path, f=self.f)

    def callback_radius(self, r):
        self.radius = r
        self.radius_display.delete(0, "end")
        self.radius_display.insert(0, str(self.radius)[:5])

    def select_raw_video(self):
        """
            Function for the user to select a file for the input. Updates the label and checks for validity.
        """
        # Open a file dialog to select a TIFF file
        file_path = self.getFile()
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

    def generate_motion_profile(self):
        """
        Function for calling tracking.generate_motion_profile. Checks for valid input and output files.
        """
        # Ensure an input file is selected
        if self.raw_video_file_path:
            # Ensure the output file path is valid
            if self.motion_output_file_var.get():  # Get the filename from the textbox
                output_folder = os.path.join(os.getcwd(), "Output")
                os.makedirs(output_folder, exist_ok=True)  # Ensure the Output folder exists

                # Ensure the extension is added
                file_name = self.motion_output_file_var.get()
                if not file_name.endswith(".tif"):
                    file_name = f"{file_name}.tif"
                filename = os.path.join(output_folder, file_name)
                self.motion_output_file_path = filename  # Update the output file path

                # Validate that crop information exists
                if hasattr(self, 'y_start') and hasattr(self, 'y_end'):
                    # Call the fine approximation function with the Y crop info
                    tracking.generate_motion_profile(self.raw_video_file_path, self.y_start, self.y_end, filename)

                    # Display the saved file path
                    display_text = f"Data saved: {filename}"
                    if len(filename) > self.MAX_FILE_DISP_LENGTH:
                        display_text = '...' + filename[len(filename) - self.MAX_FILE_DISP_LENGTH:]
                    self.motion_profile_file_label.config(text=display_text)
                else:
                    error_popup("Please select a region of interest in the crop/preprocess window.")
            else:
                error_popup("Please enter a valid output file name.")
        else:
            error_popup("Please select an input file.")
        return

    # STEP 2

    def select_analysis_input_image_file(self):
        """
            Function for the user to select a file for analysis. Updates label accordingly. 
        """
        # Allow the user to choose an existing data file
        check_file = self.getFile()
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
 
    def analyze(self):
        """
        Function for opening the analysis window. Checks for input and output file validity.
        """
        # Ensure input file is selected
        if not self.motion_output_file_path:
            error_popup("Please select an input file.")
            return

        # Ensure output file name is valid
        if self.analyze_output_file_var.get():
            output_folder = os.path.join(os.getcwd(), "Output")
            os.makedirs(output_folder, exist_ok=True)  # Ensure the Output folder exists

            # Ensure the extension is added
            file_name = self.analyze_output_file_var.get()
            if not file_name.endswith(".csv"):
                file_name = f"{file_name}.csv"
            self.analyze_output_file_path = os.path.join(output_folder, file_name)

            # Open the analysis window
            Motion_Analysis_Window(
                self.motion_output_file_path,
                self.calibration_parameters,
                self.analyze_output_file_path,
                self.get_mode(),
                self.callback_handle_crop_offset
            )
        else:
            error_popup("Please enter a valid output file name.")

    def get_mode(self):
        return self.mode_var.get()

    def get_HG_Lines(self):
        return self.green_var, self.yellow1_var, self.yellow2_var

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

        self.split_frame_num = int(self.split_var.get())

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
            
    def validate_numeric_input(self, new_value):
        """Allow only numbers (integers and floats) and an optional leading minus sign or decimal point."""
        if new_value == "" or new_value == "-"  or new_value == "." or new_value == "-.":  # Allow empty field and leading '-'
            return True
        try:
            float(new_value)  # Try converting to a float
            return True
        except ValueError:
            error_popup("Invalid input, must be numeric")
            return False  # Reject input if it’s not a number

    def selectSetupWindow(self):
        with open("setups.json", "r") as f:
            temp = json.load(f)
        ImageSelectorWindow(root, temp, default_key=str(self.setupEntryVar.get()))

    #mode, wave_lines, split_frame_num, fps

    def visualize_distance_over_time(self):
        with open("setups.json", "r") as f:
            temp = json.load(f)
        TimeVsDistanceWindow(self.visualize_mode.get(), self.wave_lines, int(self.split_var.get()), int(self.camera_fps_var.get()), [self.lambdaOdd, self.lambdaEven], self.calibration_parameters, int(self.fringe_entry.get()), temp[str(self.setupEntryVar.get())][2])

    #class ForceVsDistanceWindow:
        #def __init__(self, mode, wave_lines, split_frame_num, springConstant):

    def visualize_force_over_distance(self):
        if self.force_mode.get() == "in":
            ForceVsDistanceWindow(self.force_mode.get(), self.dispDistPairsIn, int(self.k_var.get()), float(self.radius_display.get()), self.scale_mode.get())
        elif self.force_mode.get() == "out":
            ForceVsDistanceWindow(self.force_mode.get(), self.dispDistPairsOut, int(self.k_var.get()), float(self.radius_display.get()), self.scale_mode.get())
        else:
            ForceVsDistanceWindow("full", (self.dispDistPairsIn, self.dispDistPairsOut), int(self.k_var.get()), float(self.radius_display.get()), self.scale_mode.get())


    def getFile(self):
        if self.javaExists:
            files = [("TIFF Files", "*.tif *.tiff"), ("CXD Files", "*.cxd"), ("All Files", "*")]
        else:
            files = [("TIFF Files", "*.tif *.tiff"), ("All Files", "*")]

        file_path = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd()),
            title='Browse for image file',
            filetypes=files
        )

        if file_path:
            file_path = self.cxdToTiff(file_path)

        return file_path

    def setN(self, n):
        self.fringe_entry.delete(0, tk.END)
        self.fringe_entry.insert(0, str(n))

    def setLambdas(self, lambdas):
        self.lambdaOdd = lambdas[0]
        self.lambdaEven = lambdas[1]

    def setF(self, f):
        self.f = f
        self.f_display.delete(0, "end")
        self.f_display.insert(0, str(self.f)[:5])

    def setWaveLines(self, wave_lines):
        self.wave_lines = wave_lines

    def setDispDistPairs(self, pairs, mode):
        if mode == "in":
            self.dispDistPairsIn = pairs
        else:
            self.dispDistPairsOut = pairs

    def setSetup(self, setup):
        self.setupEntryVar.set(setup)

    def setDiameter(self, diameter):
        self.diameter_display.delete(0, "end")
        self.diameter_display.insert(0, str(diameter)[:6])

    def loadInternalJSON(self):
        def is_geometry_on_screen(geometry):

            def parse_geometry(geometry):
                size, x_y = geometry.split('+', 1)
                w, h = size.split('x')
                x, y = x_y.split('+')
                return int(w), int(h), int(x), int(y)

            try:
                w, h, x, y = parse_geometry(geometry)
                for m in get_monitors():
                    if x < m.x + m.width and x + w > m.x and \
                    y < m.y + m.height and y + h > m.y:
                        return True
                return False
            except Exception:
                return False
        cache_dir = os.path.join(os.getcwd(), "cache")
        flag_path = os.path.join(cache_dir, "internal.json")

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Define window size
        window_width = int(screen_width * self.DEFAULT_WIDTH_RATIO)
        window_height = int(screen_height * self.DEFAULT_HEIGHT_RATIO)

        # Load existing flags if file exists
        
        if os.path.exists(resource(flag_path)):
            try:
                with open(resource(flag_path), "r") as f:
                    temp = json.load(f)
                    if not is_geometry_on_screen(temp["geometry"]):
                        temp.update({"geometry": f"{window_width}x{window_height}+300+100"})

                    return temp
                    
            except Exception:
                pass
        else:
            # internal.json does not exist yet
            defaultFlags = {
                "skip_java_warning": False,
                "geometry": f"{window_width}x{window_height}+300+100",
                "auto_load_calibration": True,
            }

            return defaultFlags

    def load_calibration(self, loadPrev, fileName = None):
        cache_dir = os.path.join(os.getcwd(), "cache")
        flag_path = os.path.join(cache_dir, "previousCalibration.json")

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        prevExists = os.path.exists(flag_path)

        if loadPrev and prevExists:
            with open(flag_path, "r") as f:
                return json.load(f)
        elif fileName != None:
            with open(fileName, "r") as f:
                self.calibration_values = json.load(f)
                self.updateScreenValues()
                return 
        else:
            temp = {
                "mercury_video_file": "",
                "mercury_lines": {
                    "green": False,
                    "yellow_1": False,
                    "yellow_2": False
                },
                "dispersion_values": {
                    "1": np.NaN,
                    "2": np.NaN,
                    "3": np.NaN,
                    "average": np.NaN,
                    "standard_deviation": np.NaN
                },
                "calibration_parameters": {
                    "slope": np.NaN,
                    "intercept": np.NaN,
                    "offset": np.NaN
                },
                "thickness_video_file": "",
                "mica_thickness": np.NaN, 
                "fringe_number": 0,
                "lambdas": {
                    "odd": np.NaN,
                    "even": np.NaN
                },
                "magnification_image_file": "",
                "magnification_scale": 0,
                "radius_video_file": "",
                "f_value": np.NaN,
                "radius": np.NaN, 
                "diameter_video_file": "",
                "diameter": np.NaN,
                "turnaround_frame": 0,
                "setup": "Mica Mica Symmetric",
                "fps": 2,
                "spring_constant": 0
            }

            return temp

    def updateScreenValues(self):
        self.wavelength_calibration_video_file_path = self.calibration_values["mercury_video_file"]
        self.thickness_input_file_path = self.calibration_values["thickness_video_file"]
        self.radius_input_file_path = self.calibration_values["radius_video_file"]
        self.diameter_input_file_path = self.calibration_values["diameter_video_file"]

        self.mica_thickness = self.calibration_values["mica_thickness"]
        self.lambdaOdd = self.calibration_values["lambdas"]["odd"]
        self.lambdaEven = self.calibration_values["lambdas"]["even"]
        self.calibration_parameters = self.calibration_values["calibration_parameters"]
        self.f = self.calibration_values["f_value"]
        self.radius = self.calibration_values["radius"]
        self.diameter = self.calibration_values["radius"]
        self.split_frame_num = self.calibration_values["turnaround_frame"]

        if self.wavelength_calibration_video_file_path != "":
            if len(self.wavelength_calibration_video_file_path) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.wavelength_calibration_video_file_path[len(self.wavelength_calibration_video_file_path) - self.MAX_FILE_DISP_LENGTH:]
                self.wavelength_calibration_file_label.config(text=data_file_text)
            else: 
                self.wavelength_calibration_file_label.config(text=self.wavelength_calibration_video_file_path)

        self.green_var.set(self.calibration_values["mercury_lines"]["green"])
        self.yellow1_var.set(self.calibration_values["mercury_lines"]["yellow_1"])
        self.yellow2_var.set(self.calibration_values["mercury_lines"]["yellow_2"])

        self.setDispersionEntries(self.calibration_values["dispersion_values"]["1"],self.calibration_values["dispersion_values"]["2"],self.calibration_values["dispersion_values"]["3"],self.calibration_values["dispersion_values"]["average"],self.calibration_values["dispersion_values"]["standard_deviation"])

        if self.thickness_input_file_path != "":
            if len(self.thickness_input_file_path) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.thickness_input_file_path[len(self.thickness_input_file_path) - self.MAX_FILE_DISP_LENGTH:]
                self.thickness_file_label.config(text=data_file_text)
            else:
                self.thickness_file_label.config(text=self.thickness_input_file_path) 

        if self.magnification_file:

            if len(self.magnification_file) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.magnification_file[len(self.magnification_file) - self.MAX_FILE_DISP_LENGTH:]
                self.magnification_image_label.config(text=data_file_text)
            else:
                self.magnification_image_label.config(text=self.magnification_file) 

        if self.radius_input_file_path != "":
            if len(self.radius_input_file_path) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.radius_input_file_path[len(self.radius_input_file_path) - self.MAX_FILE_DISP_LENGTH:]
                self.radius_file_label.config(text=data_file_text)
            else:
                self.radius_file_label.config(text=self.radius_input_file_path) 

        if self.diameter_input_file_path != "":
            if len(self.diameter_input_file_path) > self.MAX_FILE_DISP_LENGTH:
                data_file_text = '...' + self.diameter_input_file_path[len(self.diameter_input_file_path) - self.MAX_FILE_DISP_LENGTH:]
                self.diameter_file_label.config(text=data_file_text)
            else:
                self.diameter_file_label.config(text=self.diameter_input_file_path) 

        self.thickness_display.delete(0, "end")
        self.thickness_display.insert(0, str(self.mica_thickness))

        self.fringe_entry.delete(0, "end")
        self.fringe_entry.insert(0, str(self.calibration_values["fringe_number"]))

        self.scale_display.delete(0, "end")
        self.scale_display.insert(0, str(self.scale))

        self.f_display.delete(0, "end")
        self.f_display.insert(0, str(self.f))

        self.radius_display.delete(0, "end")
        self.radius_display.insert(0, str(self.radius)[:5])

        self.split_var.set(str(self.split_frame_num))

        self.setupEntryVar.set(str(self.calibration_values["setup"])) 

        self.camera_fps_var.set(str(self.calibration_values["fps"]))

        self.k_var.set(str(self.calibration_values["spring_constant"]))


    def save_calibration(self, onClose):
        # implement saving calibration logic
        temp = {
                "mercury_video_file": self.wavelength_calibration_video_file_path,
                "mercury_lines": {
                    "green": self.green_var.get(),
                    "yellow_1": self.yellow1_var.get(),
                    "yellow_2": self.yellow2_var.get()
                },
                "dispersion_values": {
                    "1": float(self.dispersion1_entry.get()),
                    "2": float(self.dispersion2_entry.get()),
                    "3": float(self.dispersion3_entry.get()),
                    "average": float(self.dispersionAvg_entry.get()),
                    "standard_deviation": float(self.dispersionStd_entry.get())
                },
                "calibration_parameters": self.calibration_parameters,
                "thickness_video_file": self.thickness_input_file_path,
                "mica_thickness": float(str(self.mica_thickness)[:4]), 
                "fringe_number": int(self.fringe_entry.get()),
                "lambdas": {
                    "odd": self.lambdaOdd,
                    "even": self.lambdaEven
                },
                "magnification_scale": float(self.scale_display.get()),
                "magnification_image_file": self.magnification_file,
                "radius_video_file": self.radius_input_file_path,
                "f_value": float(self.f_display.get()),
                "radius": float(self.radius_display.get()), 
                "diameter_video_file": self.diameter_input_file_path,
                "diameter": float(self.diameter_display.get()),
                "turnaround_frame": int(self.split_var.get()),
                "setup": str(self.setupEntryVar.get()),
                "fps": int(self.camera_fps_var.get()),
                "spring_constant":  int(self.k_var.get())
            }
        if onClose:

            cache_dir = os.path.join(os.getcwd(), "cache")
            flag_path = os.path.join(cache_dir, "previousCalibration.json")
            with open(flag_path, "w") as f:
                json.dump(temp, f)
        else:
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("JSON files", "*.json")],
                title="Save calibration file as..."
            )
            if filename:
                with open(filename, "w") as f:
                    json.dump(temp, f)
            else:
                exceptions.warning_popup("No file selected, aborting")

    def check_java(self):
        # Quick check if "java" is in PATH
        java_path = shutil.which("java")
        if java_path is None:
            return False

        try:
            # Run "java -version" (stderr contains version info)
            result = subprocess.run(
                ["java", "-version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            output = result.stdout.strip() or result.stderr.strip()
            if result.returncode == 0 and ("version" in output.lower() or "openjdk" in output.lower()):
                return True
            else:
                return False

        except Exception as e:
            return False

    def javaPrompt(self):
        # Access specific flag
        skip_java_warning = self.internal_flags.get("skip_java_warning", False)

        # Behavior logic
        if not self.javaExists and not skip_java_warning:
            if checkbox_popup(self.root, "Java Warning", "Java is not installed, or not properly set up in PATH. Disabling CXD to TIFF conversion"):
                self.internal_flags["skip_java_warning"] = True
                try:
                    with open(flag_path, "w") as f:
                        json.dump(self.internal_flags, f, indent=4)
                except Exception:
                    pass

    def cxdToTiff(self, file):
        if(file.lower().endswith(".cxd")):
            if platform == "win32":
                command = ['.\\bfconverter\\bfconvert.bat', file, f"{file[:-4]}.tiff"]
            else:
                command = ['./bfconverter/bfconvert', file, f"{file[:-4]}.tiff"]
            subprocess.run(command)
            file = file[:-4] + ".tiff"
        return file
    
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

        self.ax.set_xlabel("Pixels")
        self.ax.set_ylabel("Pixels")

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
            self.ax.set_xlabel("Pixels")
            self.ax.set_ylabel("Pixels")
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
        self.num_waves = 2 if sum(int(x.get()) for x in app.get_HG_Lines()) == 2 else 3

        #  Output variables

        self.dispersion1 = np.NaN
        self.dispersion2 = np.NaN
        self.dispersion3 = np.NaN
        self.dispersionAvg = np.NaN
        self.dispersionStd = np.NaN

        # Set up the Matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.2, top=.85)  # Leave space for the slider

        # Add instruction text above the plot
        self.instruction_text = self.fig.text(
            0.5, 0.95,  # Centered horizontally, near the top of the figure
            "Step 1: Select the region to crop by clicking and dragging. Press Enter to confirm.",
            ha="center", va="center", fontsize=10, wrap=True
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

        self.ax.set_xlabel("Pixels")
        self.ax.set_ylabel("Pixels")

        # Show the plot
        plt.show()

    def handle_key_press(self, event):
        """Handles key press events for crop confirmation or cancellation."""
        if event.key == "enter":
            if self.stage == 1:
                self.confirm_crop()
            elif self.stage == 2:
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
        self.instruction_text.set_wrap(True)
        self.fig.canvas.draw_idle()

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
        self.ax.set_xlabel("Pixels")
        self.ax.set_ylabel("Pixels")
        self.update_instructions("Step 1: Select the region with all Hg bars by clicking and dragging. Press Enter to confirm.")
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
            self.ax.set_xlabel("Pixels")
            self.ax.set_ylabel("Pixels")
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
            # CHANGE WORDING (CONFIRM IDENTIFIED LINES)
            self.update_instructions(f"Select {self.num_waves} lines for calibration. Red lines are selected and green are not. Press enter when finished, or escape to restart.")
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

            if self.waves is not None and len(self.waves) == self.num_waves:
                self.selected_waves = [int(np.mean([point[1] for point in wave])) for wave in self.waves]
                self.update_overlay()
                #self.calculate_transformation()
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
        self.waves = tracking.new_analyze_and_append_waves(np.array(image), wave_threshold=110,modality=app.mode_var.get(), smooth=False)
        self.display_waves()

    def display_waves(self):
        """Displays the waves over the cropped image using Matplotlib."""
        if self.waves and self.cropped_image is not None:
            self.ax.clear()
            self.ax.set_xlabel("Pixels")
            self.ax.set_ylabel("Pixels")
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
        self.ax.set_xlabel("Pixels")
        self.ax.set_ylabel("Pixels")
        self.fig.canvas.draw()

    def calculate_transformation(self):
        """Calculates the calibration equation using the selected waves."""

        x_values = sorted(self.selected_waves)
        y_values = [
            CalibrationValues.HG_GREEN.value,
            CalibrationValues.HG_YELLOW_1.value,
            CalibrationValues.HG_YELLOW_2.value
        ]

        if len(self.selected_waves) == 2:
            g, y1, y2 = app.get_HG_Lines()

            if (int(g.get()) + int(y1.get()) + int(y2.get()) != 2):
                return

            if (not g.get()):
                self.dispersion1 = (y_values[2] - y_values[1])/(x_values[1] - x_values[0])
                self.dispersionAvg = self.dispersion1
                y_values = [y_values[1]] + [y_values[2]]
            elif (not y1.get()):
                self.dispersion3 = (y_values[2] - y_values[0])/(x_values[1] - x_values[0])
                self.dispersionAvg = self.dispersion3
                y_values = [y_values[0]] + [y_values[2]]
            else:
                self.dispersion2 = (y_values[1] - y_values[0])/(x_values[1] - x_values[0])
                self.dispersionAvg = self.dispersion2
                y_values = [y_values[0]] + [y_values[1]]

        elif len(self.selected_waves) == 3:
            self.dispersion1 = (y_values[2] - y_values[1])/(x_values[2] - x_values[1])
            self.dispersion2 = (y_values[1] - y_values[0])/(x_values[1] - x_values[0])
            self.dispersion3 = (y_values[2] - y_values[0])/(x_values[2] - x_values[0])
            self.dispersionAvg = np.average([self.dispersion1,self.dispersion2,self.dispersion3])
            self.dispersionStd = np.std([self.dispersion1,self.dispersion2,self.dispersion3])

        else:
            return


        app.setDispersionEntries(self.dispersion1,self.dispersion2,self.dispersion3,self.dispersionAvg,self.dispersionStd)
        
        calibration_equation = {"slope": self.dispersionAvg, "intercept": CalibrationValues.HG_GREEN.value, "offset": x_values[0]}
        #print(calibration_equation)
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

        self.secax = self.ax.secondary_xaxis('top', functions=(self.pixToWave, self.waveToPix))
        self.secax.set_xlabel(r"Wavelength, $\it{\lambda}$ (nm)")

        # Add instruction text above the plot
        self.instruction_text = self.fig.text(
            0.5, 0.95,  # Centered horizontally, near the top of the figure
            "Step 1: Select a region including the peaks of the FECO by clicking and dragging. Press Enter to confirm.",
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

        self.fig.canvas.mpl_connect('draw_event', self.update_secondary_axis)

        # Show the plot
        plt.show()

    def update_secondary_axis(self, event):
        if hasattr(self, "secax"):
            self.secax.set_xlim(self.ax.get_xlim())

    def pixToWave(self, x):
        return self.calibration_parameters["slope"]*(x-self.calibration_parameters["offset"]) + self.calibration_parameters["intercept"]

    def waveToPix(self, lam):
        return (lam - self.calibration_parameters["intercept"]) / self.calibration_parameters["slope"] + self.calibration_parameters["offset"]

    def refresh_secondary_axis(self):
        """Ensure the top wavelength axis exists and matches the current x-limits."""
        # Remove any previous secondary axis safely
        if hasattr(self, "secax") and self.secax:
            try:
                self.secax.remove()
            except Exception:
                pass

        # Recreate the axis based on the current transform functions
        self.secax = self.ax.secondary_xaxis('top', functions=(self.pixToWave, self.waveToPix))
        self.secax.set_xlabel(r"Wavelength, $\it{\lambda}$ (nm)")

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
        self.instruction_text.set_wrap(True)
        self.fig.canvas.draw_idle()

    def scale_image(self, image):
        """Scales the image by the specified factor."""
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        width, height = image.size
        return image.resize((int(width * self.scale_factor), int(height * self.scale_factor)), Image.LANCZOS)

    def display_image(self, image):
        """Displays the current frame in the Matplotlib axes."""
        self.ax.clear()
        self.ax.imshow(np.array(image), cmap="gray", aspect="auto")

        self.ax.set_xlabel("Pixels")
        self.ax.set_ylabel("Pixels")

        self.refresh_secondary_axis()  

        self.update_instructions("Step 1: Select the region to crop by clicking and dragging. Press Enter to confirm.")
        self.fig.canvas.draw_idle()

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
        if self.stage == 1 and self.crop_start_y is not None and event.inaxes ==self.ax:
            event.inaxes = self.ax

            if event.ydata is None:
                return
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
        self.waves = tracking.new_analyze_and_append_waves(
            normalized_image,
            wave_threshold=40,
            min_points_per_wave=10,
            min_wave_gap=10,
            modality=app.mode_var.get(),
            smooth = False
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
        self.ax.clear()
        self.ax.imshow(np.array(self.cropped_frame), cmap='gray')

        self.ax.set_xlabel("Pixels")
        self.ax.set_ylabel("Pixels")

        for avg_x in self.filtered_averages:
            self.ax.axvline(x=avg_x, color='lime', linestyle='-')

        self.refresh_secondary_axis()  
        
        self.fig.canvas.draw_idle()

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
        #self.ax.axis('off')

        self.ax.set_xlabel("Pixels")
        self.ax.set_ylabel("Pixels")

        # Draw all wave lines, red for selected and green for unselected
        for avg_x in self.filtered_averages:
            color = 'lime'  # Green for unselected
            if avg_x in self.selected_waves:
                color = 'red'  # Red for selected
            self.ax.axvline(x=avg_x, color=color, linestyle='-')

        self.refresh_secondary_axis()

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
        self.selected_wavelengths = [slope * (x - self.calibration_parameters['offset']) + intercept for x in self.selected_waves] 

        if(self.selected_wavelengths[0] < self.selected_wavelengths[1]):
            lambdas = self.selected_wavelengths
        else:
            lambdas = [self.selected_wavelengths[1]] + [self.selected_wavelengths[0]]

        app.setLambdas(lambdas)

        #print(lambdas)

        thickness = self.calculate_thickness()
        if thickness:
            self.callback(thickness)

        n = int(np.round(((lambdas[1]/(lambdas[1]-lambdas[0]))-1)/1.024)) 

        app.setN(n)

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

class Magnification_Calculation_Window:
    """
    A tool for selecting a horizontal region from a TIFF image and saving its pixel values.

    Features:
        1. Displays the image (converted to grayscale).
        2. Allows user to select a horizontal region using a click-and-drag motion.
        3. Extracts and stores the pixel values within the selected region.
        4. Invokes an optional callback function with the extracted pixel array.

    Parameters:
        file_path (str): Path to the TIFF image.
        callback (function): Optional function to call with the extracted pixel data (numpy array).
    """

    def __init__(self, file_path, scale):
        self.file_path = file_path
        self.scale = scale
        self.image = self.load_as_grayscale(self.file_path)
          # Ensure grayscale
        self.image_array = np.array(self.image)

        self.selected_region = None
        self.selected_data = None

        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.imshow(self.image_array, cmap="gray", aspect="auto")
        self.ax.set_title("Magnification Scale Image")
        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")

        # Add instruction text
        self.instruction_text = self.fig.text(
            0.5, 0.95,
            "Drag to select a horizontal region including all tick lines. Press Enter to confirm or Esc to cancel.",
            ha="center", va="center", fontsize=10
        )

        # Span selector for horizontal selection
        self.span = SpanSelector(
            self.ax,
            onselect=self.on_select,
            direction="horizontal",
            useblit=True,
            interactive=True,
            props=dict(alpha=0.3, facecolor="red"),  
        )

        # Connect keypress events
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        plt.show()

    def load_as_grayscale(self, file_path):
        # Open image with PIL
        im = Image.open(file_path)
        
        # Convert to NumPy array to preserve bit depth & channels
        arr = np.array(im)
        
        # If image has multiple channels (RGB or RGBA)
        if arr.ndim == 3:
            # Average across color channels to get grayscale
            arr = arr.mean(axis=2)
        
        # Convert to float for normalization (handles 16-bit, etc.)
        arr = arr.astype(np.float32)
        
        # Normalize safely to 0–255
        arr = 255 * (arr - arr.min()) / (arr.max() - arr.min())
        arr = arr.astype(np.uint8)
        
        # Return as Pillow image in "L" mode
        return Image.fromarray(arr, mode="L")

    def on_select(self, x_min, x_max):
        """Handles selection of the horizontal region."""
        self.selected_region = (int(x_min), int(x_max))
        self.update_overlay()

    def update_overlay(self):
        """Draws vertical lines to show the selected region."""
        self.ax.clear()
        self.ax.imshow(self.image_array, cmap="gray", aspect="auto")
        if self.selected_region:
            x_min, x_max = self.selected_region
            self.ax.axvline(x_min, color="red", linestyle="--", linewidth=1.5)
            self.ax.axvline(x_max, color="red", linestyle="--", linewidth=1.5)
        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")
        self.ax.set_title("Selected Region Preview")
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        """Handles Enter/Esc key presses."""
        if event.key == "enter" and self.selected_region is not None:
            self.confirm_selection()
        elif event.key == "escape":
            self.cancel_selection()

    def confirm_selection(self):
        """Extracts average brightness per row, finds peaks, and overlays horizontal lines at peaks."""
        x_min, x_max = sorted(self.selected_region)
        # Clip within bounds
        x_min = max(0, x_min)
        x_max = min(self.image_array.shape[1], x_max)

        # Compute average brightness per row
        filtered = cv2.medianBlur(self.image_array, ksize=3)  # ksize must be odd (3, 5, 7, ...)

        # Normalize to 0–255
        normalized = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)

        y_values = np.arange(normalized.shape[0])
        avg_brightness = np.mean(self.image_array[:, x_min:x_max], axis=1)
        self.selected_data = list(zip(y_values, avg_brightness))

        # Find peaks using your existing tracking function
        peaks = tracking.arbitrary_gaussian_fits(self.selected_data, plot=True, max_gaussians = 15, prominence=15)["means"]

        self.calcF(peaks)

        self.ax.clear()
        self.ax.imshow(self.image_array, cmap="gray", aspect="auto")

        # Draw horizontal lines at peaks across the entire width
        image_width = self.image_array.shape[1]
        for peak_y in peaks:
            self.ax.hlines(y=peak_y, xmin=1, xmax=image_width -1, color="lime", linestyle="--", linewidth=1.5)

        # Draw vertical lines for selected region
        self.ax.axvline(x=x_min, color="red", linestyle="--", linewidth=1.5)
        self.ax.axvline(x=x_max, color="red", linestyle="--", linewidth=1.5)

        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")
        self.ax.set_title("Magnification Scale Image")

        self.instruction_text.set_text("Tick lines found, close to confirm or reselect region")
        self.fig.canvas.draw_idle()

    def calcF(self, peaks):
        sumDist = 0
        for i in range(len(peaks) - 1):
            sumDist += peaks[i+1] - peaks[i]
        avg = sumDist / (len(peaks) - 1)
        app.setF(avg/self.scale)

    def cancel_selection(self):
        """Clears selection and redraws image."""
        self.selected_region = None
        self.update_overlay()
        self.instruction_text.set_text("Selection cleared. Drag again to select a region.")


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
    PEAK_SELECT_MODE = 'peak_select'
    FIGURE_SIZE = (12, 4)

    def __init__(self, motion_profile_file_path, calibration_parameters, output_file_path, modality, offset_callback=None) -> None:
        self.y_offset = 0
        self.x_offset_start = 0
        self.x_offset_end = 0
        self.calibration_parameters = calibration_parameters
        self.modality = modality
        self.offset_callback = offset_callback

        self.output_filename = output_file_path
        self.timelapse_image = np.array(Image.open(motion_profile_file_path).convert('L'))
        self.file_path = motion_profile_file_path

        self.mode = Motion_Analysis_Window.CROPPING_MODE
        self.cropping_complete = False
        self.crop_area = None
        self.deletion_areas = []
        self.peak_points = []

        h, w = self.timelapse_image.shape
        aspect_ratio = w / h  # width / height

        self.fig, self.ax = plt.subplots(figsize=Motion_Analysis_Window.FIGURE_SIZE)
        self.ax.imshow(self.timelapse_image, cmap='gray', origin='upper')

        self.secax = self.ax.secondary_xaxis('top', functions=(self.pixToWave, self.waveToPix))
        self.secax.set_xlabel(r"Wavelength, $\it{\lambda}$ (nm)")

        # --- Adjust visual aspect ratio only ---
        # "aspect" = (height of data unit) / (width of data unit)
        # smaller -> stretches vertically, larger -> compresses vertically

        if aspect_ratio < 1:  
            # Too tall: stretch it (make it appear less tall → wider)
            # We want display ratio to reach 2:3 → target_aspect = h/w / (3/2)
            target_ratio = 1
            scale_factor = (aspect_ratio / target_ratio)
            self.ax.set_aspect(scale_factor)  # stretch vertically
        elif aspect_ratio > 6:
            # Too wide: compress it vertically (towards 1:1)
            target_ratio = 6
            scale_factor = (aspect_ratio / target_ratio)
            self.ax.set_aspect(scale_factor)  # compress vertically
        else:
            # Between 2:3 and 1:1 — leave normal
            self.ax.set_aspect('auto')

        self.ax.set_title(
            "Click and drag to crop the image, then press any key to confirm. "
            "Press Escape to cancel selection."
        )

        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.offsetX))

        self.rect_selector = RectangleSelector(
            self.ax, self.on_select_crop, useblit=True, interactive=True
        )
        self.fig.canvas.mpl_connect('key_press_event', self.handle_key_press)

        #self._draw_cid = self.fig.canvas.mpl_connect('draw_event', self.update_secondary_axis)

        plt.show(block=True)

        #if self.cropping_complete:
            #self.run_analysis()

    def offsetX(self,x , pos):
        return f"{int(x + self.x_offset_start)}"

    def update_secondary_axis(self, event):
        """Safely sync the top wavelength axis limits with the main axis."""
        try:
            # quick sanity checks
            if not hasattr(self, "secax") or self.secax is None:
                return
            if not hasattr(self, "ax") or self.ax is None:
                return
            # ensure figure still exists
            if not plt.fignum_exists(getattr(self, "fig").number):
                return

            # If there's no canvas/manager/toolbar, bail out
            canvas = getattr(self, "fig", None).canvas
            manager = getattr(canvas, "manager", None)
            toolbar = getattr(manager, "toolbar", None)
            # set_xlim is cheap; do it only if everything looks OK
            if toolbar is None or manager is None or canvas is None:
                # still safe to set limits (no toolbar update) — but check ax exists
                self.secax.set_xlim(self.ax.get_xlim())
                return

            # normal case: set limits
            self.secax.set_xlim(self.ax.get_xlim())

        except _tkinter.TclError:
            # occurs when Tk widgets were destroyed mid-update; ignore
            return
        except Exception:
            # swallow any other race-condition exceptions silently
            return

    def pixToWave(self, x):
        return self.calibration_parameters["slope"]*(x + self.x_offset_start - self.calibration_parameters["offset"]) + self.calibration_parameters["intercept"]

    def waveToPix(self, lam):
        return (lam - self.calibration_parameters["intercept"]) / self.calibration_parameters["slope"] + self.calibration_parameters["offset"] - self.x_offset_start

    def refresh_secondary_axis(self):
        """Ensure the top wavelength axis exists and is visible after crop/redraw."""
        if not hasattr(self, "fig") or not plt.fignum_exists(self.fig.number):
            return

        # Remove old secax if it exists to avoid conflicts
        if getattr(self, "secax", None) is not None:
            try:
                self.secax.remove()
            except Exception:
                pass

        # Create a new secondary x-axis on top
        self.secax = self.ax.secondary_xaxis('top', functions=(self.pixToWave, self.waveToPix))
        self.secax.set_xlabel(r"Wavelength, $\it{\lambda}$ (nm)")

    def handle_key_press(self, event):
        """Centralized key press handler based on mode."""
        if self.mode == self.CROPPING_MODE:
            self.handle_crop_keypress(event)
        elif self.mode == self.DELETION_MODE:
            self.handle_delete_keypress(event)
            #canvas.draw()

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

        # Deactivate cropping
        self.rect_selector.set_active(False)

        self.mode = Motion_Analysis_Window.PEAK_SELECT_MODE
        try:
            plt.close(self.fig)
        except Exception:
            pass

        self.show_peak_selection_window()

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

    # === NEW ===
    def show_peak_selection_window(self):
        """Show cropped image and let user click peaks."""
        self.fig, self.ax = plt.subplots(figsize=(10,4))
        self.ax.imshow(self.cropped_image, cmap='gray')
        self.ax.set_title("Click peaks to mark them. Press Enter to continue.")

        # Connect click + key
        self.fig.canvas.mpl_connect("button_press_event", self.on_peak_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_peak_keypress)

        plt.show(block=True)

    # === NEW ===
    def on_peak_click(self, event):
        """Save peak locations as the user clicks the cropped image."""
        if event.inaxes != self.ax:
            return
        x, y = int(event.xdata), int(event.ydata)
        self.peak_points.append((x, y))

        # Visual feedback
        self.ax.plot(x, y, 'ro')
        self.fig.canvas.draw()

    # === NEW ===
    def on_peak_keypress(self, event):
        """Finish peak selection when user hits Enter."""
        if event.key == 'enter':
            plt.close(self.fig)
            self.after_peak_selection()

        # === NEW ===
    def after_peak_selection(self):
        """Called once user finishes selecting peaks."""
        print("User selected peaks:", self.peak_points)

        # TODO later: use these peaks to improve wavefinding
        # For now run your original wavefinding:
        self.run_analysis()

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
        #self.wave_lines = tracking.gaussianWaveDetection(self.cropped_image, self.peak_points)
        self.wave_lines = tracking.new_analyze_and_append_waves(self.cropped_image,modality=app.mode_var.get())

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
        self.ax.imshow(self.cropped_image, cmap='gray', origin='upper')

        self.ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.offsetX))

        self.refresh_secondary_axis()

        # --- Apply visual aspect ratio scaling for display convenience ---
        h, w = self.cropped_image.shape
        aspect_ratio = w / h  # width / height

        if aspect_ratio < 1:
            # Too tall → stretch vertically toward 1:1
            target_ratio = 1
            scale_factor = aspect_ratio / target_ratio
            self.ax.set_aspect(scale_factor)
        elif aspect_ratio > 6:
            # Too wide → compress vertically toward 4:1
            target_ratio = 6
            scale_factor = aspect_ratio / target_ratio
            self.ax.set_aspect(scale_factor)
        else:
            # Within [1, 4] range → leave natural aspect
            self.ax.set_aspect('auto')

        # Replot the wave lines
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.wave_lines)))

        for idx, wave_line in enumerate(self.wave_lines):
            if wave_line:  # Ensure the wave_line is not empty
                y_coords = [point[0] for point in wave_line]
                x_coords = [point[1] for point in wave_line]
                self.ax.plot(x_coords, y_coords, color=colors[idx], label=f"Wave {idx + 1}")

        # Reapply title, labels, and legend
        self.ax.set_title("Highlight data to delete it. Enter to accept, Esc to cancel, close window to save.")
        self.ax.set_xlabel("Pixels")
        self.ax.set_ylabel("Frame Number")
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Redraw the updated plot
        plt.draw()

        # Adjust the layout to include all elements
        self.fig.tight_layout()

    def on_close(self, event):
        """Save the modified wave lines and the figure when the window is closed."""
        # Save the wave centerlines to CSV
        
        outputWaveLines = [[(x, y + self.x_offset_start) for (x, y) in wave] for wave in self.wave_lines]

        app.setWaveLines(outputWaveLines)
        self.save_wave_centerlines_to_csv(self.wave_lines, self.output_filename)
        
        # Update the plot title to a proper name
        self.ax.set_title("Final Wave Centerlines")
        
        if hasattr(self, "rect_selector"):
            self.rect_selector.set_active(False)
            self.rect_selector.disconnect_events()
            self.rect_selector.background = None 

        # Force redraw before switching backend
        self.fig.canvas.draw()
        
        # Save the figure as a PDF
        pdf_filename = os.path.join("Output", "last_centerline_visualization.pdf")
        try:
            self.fig.savefig(pdf_filename, format='pdf', bbox_inches='tight')
        except Exception as e:
            print(f"Error saving figure as PDF: {e}")

        try:
            plt.close(self.fig)
        except Exception:
            # Ignore any errors caused by Tkinter callbacks firing during close
            pass

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
                calibrated_filename = os.path.join(output_dir, f"calibrated_{base_filename}")
                with open(calibrated_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Wave Index", "Frame Number", "Calibrated Center of Mass X Coord"])

                    for wave_idx, wave_line in enumerate(wave_lines):
                        for (y, x_center) in wave_line:
                            frame_number = y + self.y_offset  # Ensure consistency
                            calibrated_x = self.calibration_parameters['slope'] * (x_center + self.x_offset_start - self.calibration_parameters['offset']) + self.calibration_parameters['intercept']
                            writer.writerow([wave_idx + 1, frame_number, calibrated_x])

        except Exception as e:
            msg = "Error while saving. See console for details."
            error_popup(msg)
            print(f"Error saving wave centerlines to CSV: {e}")

class RadiusMeasurementWindow:
    def __init__(self, image_path, magnification_factor, calibration_parameters, lambdas, n, equation, callback):
        """
        Opens a window for the user to select three points on a TIFF image to calculate the radius of curvature.

        Args:
            image_path (str): Path to the TIFF file.
            magnification_factor (float): The magnification factor 'f'.
            callback (function): Function to return the computed radius value.
        """
        self.image_path = image_path
        self.f = magnification_factor
        self.callback = callback
        self.calibration_parameters = calibration_parameters
        self.lambdas = lambdas
        self.n = n
        self.equation = equation
        self.points = []  # Store selected points

        # Load the image
        self.image = Image.open(image_path)
        self.image = np.array(self.image)

        # Initialize the figure and axes
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.image, cmap='gray')

        self.ax.set_xlabel("Pixels")
        self.ax.set_ylabel("Pixels")
        self.secax = self.ax.secondary_xaxis('top', functions=(self.pixToWave, self.waveToPix))
        self.secax.set_xlabel(r"Wavelength, $\it{\lambda}$ (nm)")
        self.update_title()

        # Enable interactive zoom and pan
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # Add a cursor for better accuracy
        self.cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)

        plt.show()

    def update_secondary_axis(self, event):
        """Safely sync the top wavelength axis limits with the main axis."""
        try:
            # quick sanity checks
            if not hasattr(self, "secax") or self.secax is None:
                return
            if not hasattr(self, "ax") or self.ax is None:
                return
            # ensure figure still exists
            if not plt.fignum_exists(getattr(self, "fig").number):
                return

            # If there's no canvas/manager/toolbar, bail out
            canvas = getattr(self, "fig", None).canvas
            manager = getattr(canvas, "manager", None)
            toolbar = getattr(manager, "toolbar", None)
            # set_xlim is cheap; do it only if everything looks OK
            if toolbar is None or manager is None or canvas is None:
                # still safe to set limits (no toolbar update) — but check ax exists
                self.secax.set_xlim(self.ax.get_xlim())
                return

            # normal case: set limits
            self.secax.set_xlim(self.ax.get_xlim())

        except _tkinter.TclError:
            # occurs when Tk widgets were destroyed mid-update; ignore
            return
        except Exception:
            # swallow any other race-condition exceptions silently
            return

    def pixToWave(self, x):
        return self.calibration_parameters["slope"]*(x  - self.calibration_parameters["offset"]) + self.calibration_parameters["intercept"]

    def waveToPix(self, lam):
        return (lam - self.calibration_parameters["intercept"]) / self.calibration_parameters["slope"] + self.calibration_parameters["offset"] 

    def refresh_secondary_axis(self):
        """Ensure the top wavelength axis exists and is visible after crop/redraw."""
        if not hasattr(self, "fig") or not plt.fignum_exists(self.fig.number):
            return

        # Remove old secax if it exists to avoid conflicts
        if getattr(self, "secax", None) is not None:
            try:
                self.secax.remove()
            except Exception:
                pass

        # Create a new secondary x-axis on top
        self.secax = self.ax.secondary_xaxis('top', functions=(self.pixToWave, self.waveToPix))
        self.secax.set_xlabel(r"Wavelength, $\it{\lambda}$ (nm)")


    def update_title(self):
        """Updates the title to reflect instructions and current state."""
        title = f"Select 3 points (D1, Xtop, Xbottom) - Left Click to Add, Right Click to Undo, Enter to Confirm"
        self.ax.set_title(title)
        self.fig.canvas.draw()

    def on_click(self, event):
        """Handles user clicks to select and delete points."""
        if event.xdata is None or event.ydata is None:
            return  # Ignore clicks outside the image

        if event.button == 1:  # Left-click to add a point
            if len(self.points) < 3:
                self.points.append((event.xdata, event.ydata))
                self.ax.plot(event.xdata, event.ydata, 'ro')  # Mark the point
                self.fig.canvas.draw()
        elif event.button == 3:  # Right-click to remove the last placed point
            if self.points:
                self.points.pop()
                self.redraw_points()

    def redraw_points(self):
        """Redraws points after deletion."""
        self.ax.clear()
        self.ax.imshow(self.image, cmap='gray')
        self.update_title()
        for x, y in self.points:
            self.ax.plot(x, y, 'ro')
        self.fig.canvas.draw()

    def on_key_press(self, event):
        """Handles key press events."""
        if event.key == 'escape':  # Reset selection
            self.points.clear()
            self.redraw_points()
            print("Selection reset.")
        elif event.key == 'enter' and len(self.points) == 3:  # Confirm and process
            self.process_points()

    def process_points(self):
        """Determines D1, Xtop, and Xbottom based on point locations and calculates the radius."""
        x_sorted = sorted(self.points, key=lambda p: p[0])  # Sort by x-coordinates

        min_dist_idx = []
        min_dist = np.inf

        for i in range(3):
            for j in range(i+1,3):
                if abs(x_sorted[i][0] - x_sorted[j][0]) < min_dist:
                    min_dist = abs(x_sorted[i][0] - x_sorted[j][0])
                    min_dist_idx = [i,j]

        # Assign Xtop and Xbottom based on their y-values
        Xbottom, Xtop = sorted([self.points[min_dist_idx[0]], self.points[min_dist_idx[1]]], key=lambda p: p[1])

        # Assign D1 as the remaining point
        D1 = [p for p in self.points if p not in (Xtop, Xbottom)][0]

        # Extract coordinates
        x_top, y_top = Xtop
        x_bottom, y_bottom = Xbottom
        x_d1, y_d1 = D1

        # Calculate X distance
        X = abs(y_top - y_bottom)

        x_d1 = (self.calibration_parameters["slope"] * (x_d1 - self.calibration_parameters["offset"]) + self.calibration_parameters["intercept"]) 
        x_bottom = (self.calibration_parameters["slope"] * (x_bottom - self.calibration_parameters["offset"]) + self.calibration_parameters["intercept"]) 

        self.dist = dist.distance(self.lambdas[0], self.lambdas[1], n=self.n)

        x_d1, x_bottom = np.array(self.dist.arrayDistance([x_d1, x_bottom], self.equation))

        # Compute radius using the formula
        D_diff = abs(x_d1 - x_bottom)  # Only the difference matters

        radius = (((X / (self.f/10e-6))) ** 2 ) / (8 * (D_diff*10e-9))

        self.callback(radius)  # Return value via callback
        plt.close(self.fig)  # Close the window after calculation

class Fringe_Diameter_Window:
    """
    A tool for selecting a horizontal region in an image or video frame
    and computing the real-space distance of that region using a px/um scaling.

    Parameters:
        input_file_path (str): Path to the multi-frame image / video file.
        callback (function): Called with the computed distance (in microns).
        f (float): Conversion factor in px/um.
                   Real_distance(um) = pixel_height / f
    """

    def __init__(self, input_file_path, f):
        self.input_file_path = input_file_path
        self.f = f                      # px per micrometer

        # Load image or video as PIL
        self.image = Image.open(self.input_file_path)

        # State variables
        self.crop_start_y = None
        self.crop_end_y = None
        self.temp_crop_rectangle = None
        self.stage = 1
        self.scale_factor = 1
        self.cropped_image = None

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        plt.subplots_adjust(bottom=0.2, top=.85)

        # Instruction text
        self.instruction_text = self.fig.text(
            0.5, 0.95,
            "Step 1: Select a horizontal region near the peak of a fringe to calculate the fringe diameter.",
            ha="center", va="center", fontsize=10, wrap=True
        )

        # Frame slider if needed
        self.current_frame_index = 0
        self.update_frame(0)

        if hasattr(self.image, "n_frames"):
            slider_ax = plt.axes([0.2, 0.05, 0.6, 0.03])
            self.slider = Slider(slider_ax, "Frame", 0,
                                 self.image.n_frames - 1,
                                 valinit=0, valstep=1)
            self.slider.on_changed(self.update_frame)
        else:
            self.slider = None

        # Event bindings
        self.fig.canvas.mpl_connect("button_press_event", self.handle_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self.drag_crop)
        self.fig.canvas.mpl_connect("button_release_event", self.end_crop)
        self.fig.canvas.mpl_connect("key_press_event", self.handle_key_press)

        self.ax.set_xlabel("Pixels")
        self.ax.set_ylabel("Pixels")
        plt.show()

    # ------------------------------------------
    # Events and UI
    # ------------------------------------------

    def update_instructions(self, text):
        self.instruction_text.set_text(text)
        self.instruction_text.set_wrap(True)
        self.fig.canvas.draw_idle()

    def update_frame(self, value):
        self.current_frame_index = int(value)
        self.image.seek(self.current_frame_index)
        scaled = self.scale_image(self.image)
        self.display_image(scaled)

    def scale_image(self, img):
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        w, h = img.size
        return img.resize((int(w * self.scale_factor),
                           int(h * self.scale_factor)),
                           Image.LANCZOS)

    def display_image(self, image):
        self.ax.clear()
        self.ax.imshow(np.array(image), cmap="gray")
        self.ax.set_title(f"Frame {self.current_frame_index}")
        self.update_instructions(
            "Step 1: Select a horizontal region near the peak of a fringe. Press Enter to confirm."
        )
        self.fig.canvas.draw()

    def handle_click(self, event):
        if self.stage == 1 and event.inaxes == self.ax:
            self.start_crop(event)

    def start_crop(self, event):
        self.cancel_crop()  # Clear previous selection
        self.crop_start_y = event.ydata
        self.temp_crop_rectangle = None
        self.fig.canvas.draw()

    def drag_crop(self, event):
        if (
            self.stage == 1
            and self.crop_start_y is not None
            and self.crop_end_y is None
            and event.inaxes == self.ax
        ):
            if self.temp_crop_rectangle:
                self.temp_crop_rectangle.remove()

            y2 = event.ydata
            self.temp_crop_rectangle = self.ax.add_patch(
                plt.Rectangle(
                    (0, min(self.crop_start_y, y2)),
                    self.image.width,
                    abs(y2 - self.crop_start_y),
                    edgecolor="red",
                    facecolor="none",
                    linestyle="-",
                    linewidth=1.5
                )
            )
            self.fig.canvas.draw()

    def end_crop(self, event):
        if self.stage == 1 and self.crop_start_y is not None and event.inaxes == self.ax:
            self.crop_end_y = event.ydata
            self.update_instructions("Press Enter to confirm or Esc to reset.")
            self.ax.axhline(y=self.crop_start_y, color="red")
            self.ax.axhline(y=self.crop_end_y, color="red")
            self.fig.canvas.draw()

    def handle_key_press(self, event):
        if event.key == "enter":
            if self.stage == 1:
                self.confirm_crop()
        elif event.key == "escape":
            if self.stage == 1:
                self.cancel_crop()

    # ------------------------------------------
    # Crop confirmation
    # ------------------------------------------

    def confirm_crop(self):
        if self.crop_start_y is None or self.crop_end_y is None:
            self.update_instructions("No region selected. Try again.")
            return

        y1, y2 = sorted((int(self.crop_start_y), int(self.crop_end_y)))
        self.image.seek(self.current_frame_index)

        pixel_height = abs(y2 - y1)
        real_distance_um = pixel_height / self.f  # µm

        # Call user callback
        app.setDiameter(real_distance_um)

        # Close window
        self.close_window()

    def cancel_crop(self):
        self.crop_start_y = None
        self.crop_end_y = None
        if self.temp_crop_rectangle:
            try:
                self.temp_crop_rectangle.remove()
            except Exception:
                pass
            self.temp_crop_rectangle = None
        self.update_frame(self.current_frame_index)

    def close_window(self):
        plt.close(self.fig)

class ImageSelectorWindow(tk.Toplevel):
    def __init__(self, parent, items_with_images, default_key=None):
        """
        Creates a window with a scrolling list on the left, and two stacked images on the right.

        Args:
            parent: parent Tk window (e.g., root)
            items_with_images: dict of the form
                {
                    "Label 1": ["path/to/image1a.png", "path/to/image1b.png", "function_name"],
                    "Label 2": ["path/to/image2a.png", "path/to/image2b.png", "function_name"],
                    ...
                }
            default_key: (optional) which key to select and show by default
        """
        super().__init__(parent)
        self.title("Image Selector")
        self.items_with_images = items_with_images
        self.default_key = default_key

        self.geometry("900x600")      # starting window size (wider & taller)
        self.minsize(700, 500)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Track images for resizing
        self.original_images = [None, None]
        self.displayed_images = [None, None]

        # --- Main Layout ---
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)

        self.main_frame.columnconfigure(0, weight=0)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        # --- Left: Scrollable listbox ---
        self.list_frame = ttk.Frame(self.main_frame)
        self.list_frame.grid(row=0, column=0, sticky="ns", padx=5, pady=5)

        self.scrollbar = ttk.Scrollbar(self.list_frame, orient="vertical")
        self.listbox = tk.Listbox(self.list_frame, yscrollcommand=self.scrollbar.set, height=20)
        self.scrollbar.config(command=self.listbox.yview)
        self.scrollbar.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)

        for label in self.items_with_images.keys():
            self.listbox.insert(tk.END, label)

        # --- Right: Image Display Area ---
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.image_frame.rowconfigure(0, weight=1)
        self.image_frame.rowconfigure(1, weight=1)
        self.image_frame.columnconfigure(0, weight=1)

        self.image_label_top = ttk.Label(self.image_frame, anchor="center")
        self.image_label_top.grid(row=0, column=0, sticky="nsew")

        self.image_label_bottom = ttk.Label(self.image_frame, anchor="center")
        self.image_label_bottom.grid(row=1, column=0, sticky="nsew")

        # Bind events
        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.image_frame.bind("<Configure>", self.on_resize)

        # --- If default_key provided, preselect it ---
        if self.default_key and self.default_key in self.items_with_images:
            idx = list(self.items_with_images.keys()).index(self.default_key)
            self.listbox.selection_set(idx)
            self.listbox.activate(idx)
            self.listbox.see(idx)
            self.show_item(self.default_key)

    def on_select(self, event):
        """Triggered when user clicks a list item."""
        selection = self.listbox.curselection()
        if not selection:
            return
        selected_label = self.listbox.get(selection[0])
        self.show_item(selected_label)

    def show_item(self, label):
        """Display images for a given key."""
        img_info = self.items_with_images.get(label)
        if not img_info:
            return

        # Only use first two entries (ignore function name or extra info)
        img_paths = img_info[:2]
        top_path, bottom_path = (img_paths + [None, None])[:2]

        self.original_images[0] = self.load_image(top_path)
        self.original_images[1] = self.load_image(bottom_path)
        self.update_displayed_images()

    def load_image(self, path):
        if not path:
            return None
        try:
            return Image.open(path)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None

    def update_displayed_images(self):
        top_h = self.image_label_top.winfo_height()
        bottom_h = self.image_label_bottom.winfo_height()
        width = self.image_frame.winfo_width()

        # Guard: skip if dimensions not ready yet
        if top_h <= 1 or bottom_h <= 1 or width <= 1:
            # Try again after 50ms
            self.after(50, self.update_displayed_images)
            return

        for i, (img, label, height) in enumerate(
            zip(self.original_images,
                [self.image_label_top, self.image_label_bottom],
                [top_h, bottom_h])
        ):
            if img is None:
                label.config(image="", text="No image available")
                self.displayed_images[i] = None
                continue

            aspect_ratio = img.width / img.height
            new_height = max(1, height - 10)
            new_width = max(1, min(int(new_height * aspect_ratio), width - 10))
            resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            tk_img = ImageTk.PhotoImage(resized)
            label.config(image=tk_img, text="")
            self.displayed_images[i] = tk_img

    def on_resize(self, event):
        if any(self.original_images):
            self.update_displayed_images()

    def on_close(self):
        selection = self.listbox.curselection()
        selected_label = self.listbox.get(selection[0])

        app.setSetup(selected_label)

        self.destroy()

class TimeVsDistanceWindow:

    def __init__(self, mode, wave_lines, split_frame_num, fps, lambdas, parameters, n, equation):
        self.mode = mode
        self.split_frame_num = split_frame_num
        self.slope = None
        self.intercept = None
        self.x_line = None

        if self.mode == "in":
            self.wave_lines = wave_lines[0][:split_frame_num]
        elif self.mode == "out":
            self.wave_lines = wave_lines[0][split_frame_num+1:]
        else:
            self.wave_lines = wave_lines[0]

        self.y_vals = np.array([p[0]/fps for p in self.wave_lines])
        x_vals = np.array([p[1] for p in self.wave_lines])

        self.dist = dist.distance(lambdas[0], lambdas[1], n=n)

        x_vals = parameters["slope"]*(x_vals-parameters["offset"]) + parameters["intercept"]
        self.x_vals = np.array(self.dist.arrayDistance(x_vals, equation))

        # Create figure and scatter plot
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.sc = self.ax.scatter(self.y_vals, self.x_vals, color='blue', s=20)

        x_min = self.y_vals.min()
        x_max = self.y_vals.max()
        padding = 0.05 * (x_max - x_min)  # optional 5% padding

        self.ax.set_xlim(x_min - padding, x_max + padding)

        self.ax.set_xlabel(r"Time, $\mathit{t}$ (s)")
        self.ax.set_ylabel(r"Distance, $\mathit{D}$ (nm)")
        self.ax.set_title(f"Time vs Distance Scatter Plot {mode}")
        self.ax.grid(True)

        self.instruction_text = self.fig.text(
            0.5, 0.95,  # x, y in figure coordinates
            "Click and drag to select linear region for velocity calculation.",
            ha='center', va='center', fontsize=10
        )

        # Add horizontal span selector
        self.span = SpanSelector(self.ax, self.on_select, direction='horizontal', useblit=True,
                                 props=dict(alpha=0.3, facecolor='red'))

        # Reference to current regression line (so we can remove it)
        self.current_line = None

        plt.show()

    def on_select(self, y_min, y_max):
        """
        Callback when the user selects a horizontal span.
        Computes linear regression for the points inside the selected region using NumPy.
        The regression line is extended across the full horizontal axis.
        Old regression line is removed before plotting a new one.
        """
        # Filter points within the selected y-range
        mask = (self.y_vals >= y_min) & (self.y_vals <= y_max)
        selected_x = self.y_vals[mask]
        selected_y = self.x_vals[mask]

        if len(selected_x) < 2:
            print("Select a larger region with at least 2 points.")
            return

        # NumPy linear regression
        self.slope, self.intercept = np.polyfit(selected_x, selected_y, 1)

        # Compute predicted y-values and R²
        y_pred = self.slope * selected_x + self.intercept
        residuals = selected_y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((selected_y - np.mean(selected_y))**2)
        r2 = 1 - (ss_res / ss_tot)

        # Store current axes limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # Plot regression line only over selected region
        self.x_line = self.slope * self.y_vals + self.intercept

        # Remove old line if exists
        if self.current_line is not None:
            self.current_line.remove()

        # Plot new regression line with legend label showing m, b, R²
        label = fr"m = {self.slope:.3f} $\mathit{{nm/s}}$, b = {self.intercept:.3f} $\mathit{{nm}}$, R² = {r2:.3f}"
        self.current_line, = self.ax.plot(self.y_vals, self.x_line, color='green', linewidth=2, label=label)

        # Restore original axes limits to prevent autoscaling
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        # Add/update legend
        self.ax.legend(loc="best")

        # Redraw canvas
        self.fig.canvas.draw()

        self.calcDisplacementDistancePairs()

    def calcDisplacementDistancePairs(self):
        pairs = [(self.x_vals[i], self.x_vals[i] - self.x_line[i]) for i in range(len(self.x_vals))]
        app.setDispDistPairs(pairs, self.mode)


class ForceVsDistanceWindow:
    def __init__(self, mode, pairs, springConstant, radius, y_scale='linear'):
        self.mode = mode
        self.springConstant = springConstant
        self.radius = radius

        plt.figure(figsize=(10, 6))

        # --- Mode: "in" or "out" → pairs is one list ---
        if mode in ("in", "out"):
            x_vals = [p[0] for p in pairs]
            y_vals = [p[1] * springConstant * 1e-7/radius for p in pairs]

            color = "blue" if mode == "in" else "red"
            label = f"{mode} data"

            plt.scatter(x_vals, y_vals, color=color, s=20, label=label)

        # --- Mode: "both" → plot both datasets ---
        else:
            in_pairs, out_pairs = pairs

            # IN data
            x_in = [p[0] for p in in_pairs]
            y_in = [p[1] * springConstant * 1e-7/radius for p in in_pairs]
            plt.scatter(x_in, y_in, color="blue", s=20, label="in")

            # OUT data
            x_out = [p[0] for p in out_pairs]
            y_out = [p[1] * springConstant * 1e-7/radius for p in out_pairs]
            plt.scatter(x_out, y_out, color="red", s=20, label="out")

        plt.xlabel(r"Distance, $\mathit{D}$ (nm)")
        plt.ylabel(r"Force/Radius, $\mathit{F/R}$ (mN/m)")
        plt.title("Force vs Distance Scatter Plot")
        plt.grid(True)
        plt.yscale(y_scale)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = SFA_FECO_UI(root)
    root.mainloop()