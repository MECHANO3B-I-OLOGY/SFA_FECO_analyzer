import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
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
        self.data_file_path = None  # To store the chosen or generated data file path
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

        self.analyze_button = ttk.Button(root, text="Analyze", command=self.analyze, style='Regular.TButton')
        self.analyze_button.grid(row=1, column=2, sticky='ew')

        # File field for the output data
        self.data_file_label = ttk.Label(root, text="No file selected", style='Regular.TLabel')
        self.data_file_label.grid(row=2, column=2, columnspan=2, sticky='ew', padx=10)

        # Button to choose an existing data file
        self.choose_data_button = ttk.Button(root, text="Choose Data File", command=self.choose_data_file, style='Regular.TButton')
        self.choose_data_button.grid(row=3, column=2, sticky='ew', padx=10)

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
                    tracking.process_timelapse(self.file_path, self.y_start, self.y_end, filename,)
                    self.data_file_path = filename
                    self.data_file_label.config(text=f"Data saved: {self.data_file_path}")
                else:
                    print("Y crop coordinates are not set. Please set up the analysis area.")


    def choose_data_file(self):
        # Allow the user to choose an existing data file
        self.data_file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.data_file_path:
            self.data_file_label.config(text=f"Using file: {self.data_file_path}")

    def analyze(self):
        return

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

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    app = SFA_FECO_UI(root)
    root.mainloop()
