# SFA FECO Wave Motion Analyzer

### A Python-based tool for analyzing wave motion in grayscale TIFF images, providing interactive features such as image cropping, wave centerline analysis, and post-analysis data cleaning. The tool allows users to crop the region of interest from the input image, analyze the wave motion across frames, and interactively delete unwanted or broken data points from the results.

## Features

Interactive Cropping: Allows users to click and drag on the image to select a region of interest for analysis.
Wave Centerline Analysis: Detects and visualizes wave centerlines across frames.
Post-Analysis Data Cleaning: Users can select areas to delete broken or unnecessary data from the analysis results.
Results Saving: Saves the wave centerlines as a CSV file and visualizes the results as an image (TIFF format).

## Installation

### Prerequisites

- contourpy==1.2.1
- cycler==0.12.1
- fonttools==4.53.1
- kiwisolver==1.4.5
- matplotlib==3.8.2
- numpy==1.26.3
- opencv-contrib-python==4.10.0.82
- packaging==24.1
- pandas==2.1.1
- pillow==10.2.0
- pyparsing==3.1.2
- python-dateutil==2.9.0.post0
- pytz==2024.1
- scipy==1.14.1
- screeninfo==0.8.1
- six==1.16.0
- tk==0.1.0
- tzdata==2024.1

## Setup Instructions

### 1. Clone the repository:

git clone https://github.com/yourusername/wave-motion-analyzer.git
cd wave-motion-analyzer

## 2. Install dependencies:

pip install -r requirements.txt

# Usage

## Steps to Run the Program:

### Run the main script:

python main.py

## In-Program:

### Select the Input Image:

The tool prompts you to select a grayscale TIFF image for analysis.

### Interactive Cropping:

After loading the image, you can click and drag to select a region of interest. Press any key to confirm the crop or press Esc to retry.

### Wave Centerline Analysis:

Once the crop is confirmed, the tool runs the wave centerline analysis on the cropped image.

### Post-Analysis Data Cleaning:

After the analysis, you can highlight areas of the result to delete broken or unnecessary data points. Press Enter to confirm deletion.

### Results:

The wave centerlines are saved as a CSV file, and the visualization is saved as a TIFF image in the output folder.

## Example Workflow:

Cropping: Select the cropping area of the image to focus the analysis on a specific region.
Wave Analysis: View the detected wave centerlines on the cropped image.
Data Cleaning: Interactively delete unwanted data points.
Save Results: Automatically saves the cleaned data to a CSV and visualization to a TIFF image.
Packages and Dependencies
NumPy: Array manipulation and numerical calculations.
Matplotlib: For visualizing the image and wave centerlines.
Pillow: For loading and handling image files (specifically TIFF format).
Tkinter: Used for file dialog interactions.
You can install all required dependencies using the requirements.txt file.

pip install -r requirements.txt

### File Structure

```
.
├── main.py # Entry point of the program
├── tracking.py # Script for wave analysis functions
├── requirements.txt # List of Python dependencies
├── README.md # Project README
```

### Contributing

Feel free to open an issue or submit a pull request if you want to contribute to the project.

### License

This project is licensed under the MIT License. See the LICENSE file for details.
