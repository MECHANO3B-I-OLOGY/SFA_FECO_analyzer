# import cv2
import os
import tracking
import numpy as np
import pandas as pd
import tkinter as tk
import cv2
from PIL import Image, ImageTk 
from tkinter import ttk
from tkinter import filedialog

from exceptions import error_popup, warning_popup

# import tracking

class SFA_FECO_UI:
    def __init__(self, root):
        self.root = root
        self.root.title("SFA FECO Analyzer")
        self.file_path = None

        # setup styles
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

        # Define scaling factors (e.g., 50% of screen width and 70% of screen height)
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

        # File inspect button
        self.select_button = ttk.Button(root, text="Inspect File", command=lambda: Frame_Inspect_Window(self.file_path), style='Regular.TButton')
        self.select_button.grid(row=2, column=0)

        # # Button that will close the popup
        self.label = ttk.Label(root, text="Close")
        self.label.grid(row=3, column=0)

        
        self.button = ttk.Button(root, text="OK", command=self.close_popup, style='Regular.TButton')
        self.button.grid(row=4, column=0)

    def setup_styles(self):
        self.btn_style = ttk.Style()
        self.btn_style.configure(
            "Regular.TButton",
            padding=(10, 5),
            relief="raised",
            width=20
        )

    def select_file(self):
        # Open a file dialog to select a file
        # file_path = filedialog.askopenfilename(
        #     initialdir=os.path.join(os.getcwd()),
        #     title='Browse for AVI file',
        #     filetypes=[("AVI Files", "*.avi")]
        # )
        file_path = "P1 HW_AVI.avi"
        if file_path:
            # Save the selected file path
            self.file_path = file_path
            
            # Update the label to display the file name
            self.file_label.config(text=f"Selected File: {os.path.basename(file_path)}")

            # Check if the file exists
            if not os.path.isfile(self.file_path):
                msg = "Invalid file"
                error_popup(msg)

    def close_popup(self):
        self.root.destroy()

class Frame_Inspect_Window:
    def __init__(self, file_path):
        self.file_path = file_path
        
        # Initialize the frame number and ROI variables
        self.current_frame = 0
        self.original_frame = None
        self.cropped_frame = None
        self.roi_y_start = None
        self.roi_y_end = None
        self.draw_rectangle = True  # Flag to control rectangle drawing
        self.current_offset = 0  # Track offset for Y coordinates
        
        # Create a new window
        self.window = tk.Toplevel()
        self.window.title("Inspect AVI File")
        
        # Create a canvas for image display
        self.canvas = tk.Canvas(self.window, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=3, sticky="nsew")

        # Define checkbox variable for edge analysis
        self.edge_var = tk.BooleanVar()
        self.edge_checkbutton = tk.Checkbutton(self.window, text="Enable Edge Analysis", variable=self.edge_var, command=self.display_frame)
        self.edge_checkbutton.grid(row=1, column=2, sticky="ew", padx=10)

        # Load the AVI file
        self.capture = cv2.VideoCapture(self.file_path)
        if not self.capture.isOpened():
            error_popup("Failed to load video.")
            return
        
        # Read the first frame
        self.original_frame = self.read_frame()
        self.display_frame()
        
        if int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 0:
            # Add a slider to navigate through the frames
            self.slider = tk.Scale(self.window, from_=0, to=int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, orient=tk.HORIZONTAL, command=self.on_slider_change)
            self.slider.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        
            # Update the slider max value
            self.slider.config(to=int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
        
        # Bind key events
        self.window.bind("<Escape>", self.reset_image)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)

        # Get the size of the first frame
        frame_height, frame_width = self.original_frame.shape[:2]
        self.window.geometry(f"{frame_width}x{frame_height + 50}")  # Adjust for the slider height
        self.canvas.config(width=frame_width, height=frame_height)

    def read_frame(self):
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.capture.read()
        if not ret:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return frame

    def display_frame(self):
        frame = self.original_frame.copy()
        
        if self.cropped_frame is not None:
            frame = self.cropped_frame

        if self.edge_var.get():  # Apply edge analysis if enabled
            frame = tracking.analyze_edges(frame)
            print("on")
        else:
            print("off")
            
            frame = self.original_frame.copy()
            if self.cropped_frame is not None:
                frame = self.cropped_frame

        # Convert the image from BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the image to PIL format and then to ImageTk format
        pil_image = Image.fromarray(frame)
        tk_image = ImageTk.PhotoImage(pil_image)
        
        # Clear the canvas before displaying the new frame
        self.canvas.delete("all")

        # Display the image on the canvas
        self.canvas.create_image(0, self.current_offset, anchor=tk.NW, image=tk_image)
        self.canvas.image = tk_image  # Keep a reference to avoid garbage collection

        # Draw the ROI rectangle if needed
        if self.draw_rectangle and self.roi_y_start is not None and self.roi_y_end is not None:
            self.draw_roi_rectangle()

    def draw_roi_rectangle(self):
        if self.roi_y_start is not None and self.roi_y_end is not None:
            y_start = min(self.roi_y_start, self.roi_y_end)
            y_end = max(self.roi_y_start, self.roi_y_end)
            self.canvas.create_rectangle(0, y_start, self.canvas.winfo_width(), y_end, outline="red", width=2)

    def on_slider_change(self, value):
        self.current_frame = int(value)
        self.original_frame = self.read_frame()
        if self.roi_y_start is not None and self.roi_y_end is not None:
            self.apply_roi()  # Apply ROI to the new frame
        self.display_frame()

    def reset_image(self, event=None):
        self.cropped_frame = None
        self.roi_y_start = None
        self.roi_y_end = None
        self.current_offset = 0
        self.draw_rectangle = True  # Enable rectangle drawing
        self.display_frame()

    def on_mouse_down(self, event):
        self.draw_rectangle = True
        self.roi_y_start = event.y
        self.roi_y_end = event.y
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    def on_mouse_drag(self, event):
        self.roi_y_end = event.y
        self.display_frame()

    def on_mouse_up(self, event):
        self.roi_y_end = event.y
        self.canvas.unbind("<ButtonRelease-1>")
        self.apply_roi()

    def apply_roi(self):
        if self.roi_y_start is not None and self.roi_y_end is not None:
            y_start = min(self.roi_y_start, self.roi_y_end)
            y_end = max(self.roi_y_start, self.roi_y_end)
            frame = self.read_frame()
            
            if y_start < frame.shape[0] and y_end <= frame.shape[0]:
                self.cropped_frame = frame[y_start:y_end, :]
                self.current_offset = y_start  # Track the offset for further processing

            self.draw_rectangle = False  # Disable rectangle drawing after selection
            self.display_frame()

if __name__ == '__main__':
    root = tk.Tk()
    # Set the window size to 800x600 pixels
    root.geometry("800x600")
    window = SFA_FECO_UI(root)
    root.mainloop()
