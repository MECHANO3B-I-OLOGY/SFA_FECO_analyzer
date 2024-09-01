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

        # Setup styles
        self.setup_styles()

        # Configure grid layout for the root window
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=1)
        root.grid_rowconfigure(3, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        # Get screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Define scaling factors 
        window_width = int(screen_width * 0.5)
        window_height = int(screen_height * 0.7)

        # Set the window size and position
        self.root.geometry(f"{window_width}x{window_height}")

        # File select button
        self.select_button = ttk.Button(root, text="Select File", command=self.select_file, style='Regular.TButton')
        self.select_button.grid(row=0, column=0)

        # Label to display the selected file's name
        self.file_label = ttk.Label(root, text="")
        self.file_label.grid(row=1, column=0)

        # Label to display the setup button
        self.file_label = ttk.Label(text="STEP 1: Set analysis area")
        self.file_label.grid(row=1, column=0)

        # Analysis setup button
        self.setup_button = ttk.Button(root, text="Set up", command=self.open_frame_inspect_window, style='Regular.TButton')
        self.setup_button.grid(row=2, column=0)

        # Button to close the application
        self.close_button = ttk.Button(root, text="Close", command=self.close_popup, style='Regular.TButton')
        self.close_button.grid(row=3, column=0, pady=(0, 10))

    def setup_styles(self):
        self.btn_style = ttk.Style()
        self.btn_style.configure(
            "Regular.TButton",
            padding=(10, 5),
            relief="raised",
            width=20
        )

    def select_file(self):
        # Open a file dialog to select a TIFF file
        """file_path = filedialog.askopenfilename(
            initialdir=os.path.join(os.getcwd()),
            title='Browse for TIFF file',
            filetypes=[("TIFF Files", "*.tif *.tiff")]
        )"""
        file_path = "P1 HW.tif"
        if file_path:
            # Save the selected file path
            self.file_path = file_path
            
            # Update the label to display the file name
            self.file_label.config(text=f"Selected File: {os.path.basename(file_path)}")

            # Check if the file exists
            if not os.path.isfile(self.file_path):
                msg = "Invalid file"
                error_popup(msg)

    def open_frame_inspect_window(self):
        Frame_Inspect_Window(self.file_path, self.handle_roi_selection)
        
    def handle_roi_selection(self, roi_data):
        """
        Handle the ROI data returned from the Frame_Inspect_Window.
        :param roi_data: Tuple containing (y_start, y_end, offset, frame).
        """
        y_start, y_end, offset, cropped_frame = roi_data
        # print(f"ROI Selected: Y-Start: {y_start}, Y-End: {y_end}, Offset: {offset}")

    def close_popup(self):
        self.root.destroy()

class Frame_Inspect_Window:
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
        self.window.title("Inspect TIFF File")

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
        # Ensure self.cropped_frame and self.original_frame are PIL images if needed
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
            frame = self.original_frame.copy()  # Make sure this is a PIL image

            frame_width, frame_height = frame.size
            print(f"Frame Size: {frame_width}x{frame_height}")
            print(f"Cropping Coordinates: {0}, {y_start}, {frame_width}, {y_end}")

            if y_start < frame_height and y_end <= frame_height and y_start < y_end:
                self.cropped_frame = frame.crop((0, y_start, frame_width, y_end))
                self.current_offset = y_start
            else:
                print("Invalid cropping coordinates.")

            self.draw_rectangle = False
            self.display_frame()

            # Check if the callback is provided and callable
            if self.roi_callback:
                self.roi_callback((y_start, y_end, self.current_offset, self.cropped_frame))
            else:
                print("No callback provided. ROI selection will not be returned to the main window.")

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
        self.display_frame()

# Example usage
if __name__ == "__main__":
    root = tk.Tk()
    app = SFA_FECO_UI(root)
    root.mainloop()
