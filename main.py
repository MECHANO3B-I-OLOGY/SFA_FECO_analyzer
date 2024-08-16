# import cv2
import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from exceptions import error_popup, warning_popup

# import tracking

class SFA_FECO_UI:
    def __init__(self, root):
        self.root = root
        self.root.title("SFA FECO Analyzer")
        self.file_path = None

        # Configure grid layout for the root window
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=1)
        root.grid_rowconfigure(3, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Get screen width and height
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Define scaling factors (e.g., 50% of screen width and 70% of screen height)
        window_width = int(screen_width * 0.5)
        window_height = int(screen_height * 0.7)

        # Set the window size and position
        self.root.geometry(f"{window_width}x{window_height}")

        # File select button
        self.select_button = tk.Button(root, text="Select File", command=self.select_file, style='Regular.TButton')
        self.select_button.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

        # Label to display the selected file's name
        self.file_label = tk.Label(root, text="")
        self.file_label.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

        # Label with the message
        self.label = tk.Label(root, text="Close")
        self.label.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

        # Button that will close the popup
        self.button = tk.Button(root, text="OK", command=self.close_popup)
        self.button.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

    def select_file(self):
        # Open a file dialog to select a file
        file_path = filedialog.askopenfilename(initialdir=os.path.join(os.getcwd()),
                                        title='Browse for video file',)

        if file_path:
            # Save the selected file path
            self.selected_file_path = file_path
            
            # Update the label to display the file name
            self.file_label.config(text=f"Selected File: {file_path.split('/')[-1]}")

    def close_popup(self):
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    # Set the window size to 800x600 pixels
    root.geometry("800x600")
    window = SFA_FECO_UI(root)
    root.mainloop()
