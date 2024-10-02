import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import csv
from PIL import Image, ImageTk, ImageSequence

import tracking  # Assuming this is a custom module for edge analysis
from exceptions import error_popup, warning_popup


class SFA_FECO_UI:
    """
        Main UI function for SFA FECO 
    """
    def __init__(self, root):
        self.root = root
        self.root.title("SFA FECO Analyzer")
        self.file_path = None
        self.motion_profile_file_path = None  # To store the chosen or generated data file path
        self.is_sidebar_open = False
        self.root.geometry("400x400+300+100") # Force the parent window to start at a set position

        # Get screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Define scaling factors
        window_width = int(screen_width * 0.35)
        window_height = int(screen_height * 0.7)
        
        self.root.geometry(f"{window_width}x{window_height}")

        # Setup styles
        self.setup_styles()

         # Configure grid layout for the root window
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=1)
        root.grid_rowconfigure(3, weight=1)
        root.grid_rowconfigure(4, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=0)
        root.grid_columnconfigure(2, weight=1)

        # Prep Section: Add a column for the raw video select, crop/preprocess, and generate motion profile buttons
        prep_label = ttk.Label(root, text="STEP 1: Prep", style='Step.TLabel')
        prep_label.grid(row=0, column=0, sticky='ew', padx=10)

        # Raw video data select button
        self.select_raw_button = ttk.Button(root, text="Select Raw Video File", command=self.select_raw_video, style='Regular.TButton')
        self.select_raw_button.grid(row=1, column=0, sticky='ew', padx=10, pady=5)

        # Label to display the selected file's name
        self.file_label = ttk.Label(root, text="No file selected", style='Regular.TLabel')
        self.file_label.grid(row=2, column=0, sticky='new', padx=10)

        # Crop/Preprocess button
        self.crop_button = ttk.Button(root, text="Crop/Preprocess", command=self.open_crop_preprocess_window, style='Regular.TButton')
        self.crop_button.grid(row=3, column=0, sticky='ew', padx=10, pady=5)

        # Generate motion profile button
        self.generate_motion_button = ttk.Button(root, text="Generate Motion Profile", command=self.generate_motion_profile, style='Regular.TButton')
        self.generate_motion_button.grid(row=4, column=0, sticky='ew', padx=10, pady=5)

        # Step 3: Analyze
        step3_label = ttk.Label(root, text="STEP 2: Analyze", style='Step.TLabel')
        step3_label.grid(row=0, column=2, sticky='ew', padx=10)

        # Button to choose an existing data file
        self.choose_data_button = ttk.Button(root, text="Choose Data File", command=self.choose_data_file, style='Regular.TButton')
        self.choose_data_button.grid(row=1, column=2, sticky='ew', padx=10)

        # File field for the output data
        self.data_file_label = ttk.Label(root, text="No file selected", style='Regular.TLabel')
        self.data_file_label.grid(row=2, column=2, columnspan=2, sticky='enw', padx=10)

        self.analyze_button = ttk.Button(root, text="Analyze", command=self.analyze, style='Regular.TButton')
        self.analyze_button.grid(row=3, column=2, sticky='ew')

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
        file_path = "FR1-P1-bis.tif"
        if file_path:
            # Save the selected file path
            self.file_path = file_path
            
            # Update the label to display the file name
            self.file_label.config(text=f"Selected File: {os.path.basename(file_path)}")

            # Check if the file exists
            if not os.path.isfile(self.file_path):
                msg = "Invalid file"
                error_popup(msg)

    def open_crop_preprocess_window(self):
        Frame_Prep_Window(self.file_path, self.handle_roi_selection)
        
    def handle_roi_selection(self, roi_data):
        """
        Handle the ROI data returned from the Frame_Prep_Window.
        :param roi_data: Tuple containing (y_start, y_end, offset, frame).
        """
        self.y_start, self.y_end, self.offset, cropped_frame = roi_data
        # print(f"ROI Selected: Y-Start: {y_start}, Y-End: {y_end}, Offset: {offset}")

    def close_popup(self):
        self.root.destroy()
    
    def generate_motion_profile(self):
        # Ensure a file is selected before analyzing
        if self.file_path:
            # Ask the user for a filename to save the data
            # filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            filename = "test3.tiff"
            if filename:
                if hasattr(self, 'y_start') and hasattr(self, 'y_end'):
                    # Call the fine approximation function with the Y crop info
                    tracking.generate_motion_timelapse(self.file_path, self.y_start, self.y_end, filename,)
                    self.motion_profile_file_path = filename
                    self.data_file_label.config(text=f"Data saved: {self.motion_profile_file_path}")
                else:
                    print("Y crop coordinates are not set. Please set up the analysis area.")


    def choose_data_file(self):
        max_length = 20;
        # Allow the user to choose an existing data file
        self.motion_profile_file_path = filedialog.askopenfilename(filetypes=[("Tiff files", "*.tiff")])
        if len(self.motion_profile_file_path) > max_length:
            data_file_text = '...' + self.motion_profile_file_path[len(self.motion_profile_file_path) - max_length:]
        if self.motion_profile_file_path:
            self.data_file_label.config(text=f"Using file: {data_file_text}")

    def analyze(self):
        Motion_Analysis_Window(self.motion_profile_file_path)

class Frame_Prep_Window:
    def __init__(self, file_path, roi_callback=None):
        self.file_path = file_path
        self.roi_callback = roi_callback

        # Initialize variables
        self.current_frame = 0
        self.original_frame = None
        self.cropped_frame = None
        self.roi_y_start = None
        self.roi_y_end = None
        self.draw_rectangle = True
        self.current_offset = 0
        self.frames = []

        # Load the TIFF file using PIL
        try:
            self.tiff_image = Image.open(self.file_path)
            self.frames = [frame.copy() for frame in ImageSequence.Iterator(self.tiff_image)]
        except Exception as e:
            error_popup(f"Failed to load TIFF file: {e}")
            return

        # Create the window
        self.window = tk.Toplevel()
        self.window.title("Prep TIFF File")

        # Create the canvas
        self.canvas = tk.Canvas(self.window, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=3, sticky="nsew")

        # Define checkbox variable for edge analysis
        self.edge_var = tk.BooleanVar()
        self.edge_checkbutton = tk.Checkbutton(
            self.window, text="Enable Edge Analysis", variable=self.edge_var, command=self.display_frame
        )
        self.edge_checkbutton.grid(row=1, column=2, sticky="ew", padx=10)

        # Read the first frame
        self.original_frame = self.frames[0]
        self.display_frame()

        # Add a slider if there are frames
        if len(self.frames) > 1:
            self.slider = tk.Scale(
                self.window, from_=0, to=len(self.frames) - 1,
                orient=tk.HORIZONTAL, command=self.on_slider_change
            )
            self.slider.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        # Bind key and mouse events
        self.window.bind("<Escape>", self.reset_image)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)

        # Set initial window size
        self.update_window_size()

    def update_window_size(self):
        """Update window size based on the current frame dimensions."""
        if self.original_frame is not None:
            frame_width, frame_height = self.original_frame.size
            self.canvas.config(width=frame_width, height=frame_height)
            self.window.geometry(f"{frame_width}x{frame_height + 50}")

    def display_frame(self):
        """Display the current frame on the canvas."""
        # 
        frame = self.cropped_frame.copy() if self.cropped_frame is not None else self.original_frame.copy()

        if self.edge_var.get():  # Apply edge analysis if enabled
            pil_image = tracking.display_edges(np.array(frame))
            # Convert the PIL Image to PhotoImage
            tk_image = ImageTk.PhotoImage(pil_image)
        else:
            # Convert the numpy array or PIL image directly to PhotoImage
            if isinstance(frame, np.ndarray):
                pil_image = Image.fromarray(frame)
            else:
                pil_image = frame  # Assume it's already a PIL Image

            tk_image = ImageTk.PhotoImage(pil_image)

        # Clear the canvas before displaying the new frame
        self.canvas.delete("all")

        # Display the image on the canvas
        self.canvas.create_image(0, self.current_offset, anchor=tk.NW, image=tk_image)
        self.canvas.image = tk_image  # Keep a reference to avoid garbage collection

        # Draw the ROI rectangle if needed
        if self.draw_rectangle and self.roi_y_start is not None and self.roi_y_end is not None:
            self.draw_roi_rectangle()

        # Update the window size based on the frame
        self.update_window_size()

    def reset_image(self, event=None):
        """Reset the image to its original state without cropping or ROI."""
        self.cropped_frame = None
        self.roi_y_start = None
        self.roi_y_end = None
        self.current_offset = 0
        self.draw_rectangle = True
        self.display_frame()

    def apply_roi(self):
        """Apply the selected ROI to the frame."""
        if self.roi_y_start is not None and self.roi_y_end is not None:
            y_start = min(self.roi_y_start, self.roi_y_end)
            y_end = max(self.roi_y_start, self.roi_y_end)
            frame = self.original_frame.copy()  # Ensure this is a PIL image

            # Use the new crop_frame method
            self.cropped_frame = self.crop_frame()

            if self.cropped_frame:
                self.current_offset = y_start
                self.draw_rectangle = False
                self.display_frame()

                # Call the callback if provided
                if self.roi_callback:
                    self.roi_callback((y_start, y_end, self.current_offset, self.cropped_frame))
                else:
                    print("No callback provided. ROI selection will not be returned.")

    def crop_frame(self):
        """
        Crop the frame based on the provided Y coordinates.
        """
        frame_width, frame_height = self.original_frame.size

        self.current_offset = min(self.roi_y_start, self.roi_y_end)

        # Ensure y_start and y_end are within the frame height and valid
        if self.roi_y_start < frame_height and self.roi_y_end <= frame_height and self.roi_y_start < self.roi_y_end:
            cropped_frame = self.original_frame.crop((0, self.roi_y_start, frame_width, self.roi_y_end))
            return cropped_frame
        else:
            print("Invalid cropping coordinates.")
            return None    

    def draw_roi_rectangle(self):
        """Draw the ROI rectangle on the canvas."""
        if self.roi_y_start is not None and self.roi_y_end is not None:
            y_start = min(self.roi_y_start, self.roi_y_end)
            y_end = max(self.roi_y_start, self.roi_y_end)
            self.canvas.create_rectangle(0, y_start, self.canvas.winfo_width(), y_end, outline="red", width=2)

    def on_mouse_down(self, event):
        """Handle mouse button press event."""
        self.draw_rectangle = True
        self.roi_y_start = event.y
        self.roi_y_end = event.y
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def on_mouse_drag(self, event):
        """Handle mouse drag event."""
        if self.roi_y_start is not None:
            self.roi_y_end = event.y
            self.display_frame()

    def on_mouse_up(self, event):
        """Handle mouse button release event."""
        self.roi_y_end = event.y
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.apply_roi()

    def on_slider_change(self, value):
        """Handle slider changes to update the current frame."""
        self.current_frame = int(value)
        self.original_frame = self.frames[self.current_frame]

        # Apply cropping using existing ROI if defined
        if self.roi_y_start is not None and self.roi_y_end is not None:
            self.cropped_frame = self.crop_frame()

            if self.cropped_frame:
                self.draw_rectangle = False
                self.display_frame()
            else:
                print("Cropping failed due to invalid coordinates.")
        else:
            # Display the full frame if no ROI is defined
            self.display_frame()

class Motion_Analysis_Window:
    def __init__(self, file_path) -> None:
        # Step 1: Request output file name
        self.output_filename = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not self.output_filename:
            print("No output file selected. Aborting.")
            return

        # Step 2: Load the timelapse image (convert it to a NumPy array)
        self.timelapse_image = np.array(Image.open(file_path).convert('L'))  # Grayscale conversion

        # Step 3: Prompt the user to crop the image first
        self.cropping_complete = False
        self.crop_area = None  # Store the crop area

        # Increase the figure size for larger display
        self.fig, self.ax = plt.subplots(figsize=(12, 8))  # Set larger figure size
        self.ax.imshow(self.timelapse_image, cmap='gray')
        self.ax.set_title("Click and drag to crop the image, then press any key to confirm. Press escape to cancel selection.")

        # Create a RectangleSelector for cropping the image
        self.rect_selector = RectangleSelector(self.ax, self.on_select_crop, useblit=True, interactive=True)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press_crop)

        plt.show(block=True)

        # Step 4: Run the analysis after cropping
        if self.cropping_complete:
            self.run_analysis()

    def on_select_crop(self, eclick, erelease):
        """Callback for selecting crop area."""
        self.crop_area = (int(eclick.xdata), int(erelease.xdata), int(eclick.ydata), int(erelease.ydata))

    def on_key_press_crop(self, event):
        """Handle key press events for confirming or retrying the crop."""
        if event.key == 'escape':
            # Reset the selection and allow the user to try again
            print("Retrying crop selection...")
            self.rect_selector.set_active(True)
        else:
            # Confirm the crop selection and proceed
            if self.crop_area:
                print(f"Crop confirmed: {self.crop_area}")
                self.cropping_complete = True
                plt.close(self.fig)  # Close the figure to continue

    def run_analysis(self):
        """Run the analysis on the cropped image."""
        x_start, x_end, y_start, y_end = self.crop_area
        self.cropped_image = self.timelapse_image[y_start:y_end, x_start:x_end]

        # Step 5: Analyze the cropped image
        self.wave_lines = tracking.analyze_and_append_waves(self.cropped_image)

        # Step 6: Visualize and enable data deletion
        self.visualize_wave_centerlines(self.cropped_image, self.wave_lines, enable_deletion=True)

        # Step 7: Save the results as a CSV file
        self.save_wave_centerlines_to_csv(self.wave_lines, self.output_filename)

    def visualize_wave_centerlines(self, image, wave_lines, enable_deletion=False):
        """Visualize and enable deletion on the results."""
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(image, cmap='gray')
        colors = plt.cm.rainbow(np.linspace(0, 1, len(wave_lines)))

        for idx, wave_line in enumerate(wave_lines):
            y_coords = [point[0] for point in wave_line]
            x_coords = [point[1] for point in wave_line]
            self.ax.plot(x_coords, y_coords, color=colors[idx], label=f"Wave {idx + 1}")

        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("Detected Wave Centerlines")
        plt.xlabel("X (columns)")
        plt.ylabel("Y (rows)")
        plt.tight_layout()

        if enable_deletion:
            # Enable selection of areas to delete data
            self.rect_selector = RectangleSelector(self.ax, self.on_select_delete, useblit=True)
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press_delete)
            self.deletion_areas = []  # Store deletion areas
            self.fig.canvas.mpl_connect('close_event', self.on_close)

        plt.show()

    def on_select_delete(self, eclick, erelease):
        """Callback to record the area selected for deletion."""
        x_start, x_end = sorted([int(eclick.xdata), int(erelease.xdata)])
        y_start, y_end = sorted([int(eclick.ydata), int(erelease.ydata)])
        self.deletion_areas.append((x_start, x_end, y_start, y_end))
        print(f"Area selected for deletion: {x_start}-{x_end}, {y_start}-{y_end}")

    def on_key_press_delete(self, event):
        """Handle key press events for confirming deletion or confirming crop."""
        if event.key == 'enter':
            # If Enter is pressed, apply deletions
            print("Confirming deletion...")
            self.apply_deletions()
        else:
            # For any other key, confirm the crop and proceed
            if self.crop_area:
                print(f"Crop confirmed: {self.crop_area}")
                self.cropping_complete = True
                plt.close(self.fig)  # Close the figure to continue

    def apply_deletions(self):
        """Apply deletions to the selected data."""
        for area in self.deletion_areas:
            x_start, x_end, y_start, y_end = area
            for wave_line in self.wave_lines:
                wave_line[:] = [(y, x) for (y, x) in wave_line if not (x_start <= x <= x_end and y_start <= y <= y_end)]

        # Redraw the figure with the updated data
        self.ax.clear()
        
        # Replot the image
        self.ax.imshow(self.cropped_image, cmap='gray')

        # Replot the wave lines
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.wave_lines)))

        for idx, wave_line in enumerate(self.wave_lines):
            y_coords = [point[0] for point in wave_line]
            x_coords = [point[1] for point in wave_line]
            self.ax.plot(x_coords, y_coords, color=colors[idx], label=f"Wave {idx + 1}")

        # Reapply title, labels, and legend
        self.ax.set_title("Highlight data to delete it. Enter to accept, esc to cancel, exit window to save")
        self.ax.set_xlabel("X (columns)")
        self.ax.set_ylabel("Y (rows)")
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Redraw the updated plot
        plt.draw()

    def on_close(self, event):
        """Save the modified wave lines when the window is closed."""
        self.save_wave_centerlines_to_csv(self.wave_lines, self.output_filename)

    def save_wave_centerlines_to_csv(self, wave_lines, output_filename):
        """Save the wave centerlines to a CSV file."""
        try:
            with open(output_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Wave Index", "Y-Coordinate (row)", "X-Center (column)"])

                for wave_idx, wave_line in enumerate(wave_lines):
                    for (y, x_center) in wave_line:
                        writer.writerow([wave_idx + 1, y, x_center])

            print(f"Wave centerlines successfully saved to {output_filename}")
        except Exception as e:
            print(f"Error saving wave centerlines to CSV: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SFA_FECO_UI(root)
    root.mainloop()
