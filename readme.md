# SFA FECO Motion Analyzer

### A Python-based tool for analyzing motion in grayscale TIFF images, providing interactive features such as image cropping, wave centerline analysis, and post-analysis data cleaning. The tool allows users to crop the region of interest from the input image, analyze the wave motion across frames, and interactively delete unwanted or broken data points from the results.

## Features

- Note: input multipage .tiff or .cxd (linux only) file
- Interactive Cropping: Allows users to click and drag on the - image to select a region of interest for analysis.
- Wave Centerline Analysis: Detects and visualizes wave - centerlines across frames. Saves as TIFF imaage.
- Post-Analysis Data Cleaning: Users can select areas to delete - broken or unnecessary data from the analysis results.
- Results Saving: Saves the wave centerlines as a CSV file and - visualizes the results as an image (TIFF format).

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

### Setup Instructions

#### 1. Clone the repository:

git clone https://github.com/yourusername/wave-motion-analyzer.git
cd wave-motion-analyzer

#### 2. Install dependencies and set up virtual environment:

python -m venv venv
.\env\Scripts\activate
pip install -r requirements.txt

# Usage

## Steps to Run the Program:

### Run the main script:

python main.py

## In-Program:

Function buttons which require input/output files will have buttons for selecting these.

### Step 0: Calibrate (optional)

Here the user has the opprotunity calibrate two parts of the physical experiment; the conversion from digital length to physical wavelength, and the thickness of the mica sheets being used.

#### Wavelength Calibration

Once the user has selected a tiff video, they will be able to select a frame and an area they would like to use to detect the calibration markers. This area should contain the markers and as little else as possible, and should be fairly narrow. Once a crop area is selected the program will run a line detection function to find all of the lines in the image. The user will need to select 3 to serve as markers. The x location of these will be linearly regressed with mercury wavelength markers:
HG_GREEN = 546.075
HG_YELLOW_1 = 576.959
HG_YELLOW_2 = 579.065
To find the conversion equation from pixels to nanometers.

#### Thickness Calibration

Once the user has selected a tiff video, they will be able to select a frame and an area they would like to use to detect the calibration markers. This area should contain the markers and as little else as possible, and should be fairly narrow. Once a crop area is selected the program will run a line detection function to find all of the lines in the image. The user will need to select 2 to serve as markers.

I don't know how this works so I'm leaving it like this for now.

### Step 1: Prep

This stage allows you to prepare the input TIFF file for further analysis by cropping it and viewing the frames.

#### Select Raw Video File

The tool prompts you to select a TIFF video for analysis.

#### Crop/Preprocess

Tool for selecting an area of the video to run future analysis on. Select an area that is quite narrow and as vertically uniform as possible. The more "tilted" a wave is within the area, the poorer the quality of the results.

#### Generate Motion Profile:

This button will take the cropped video and run a series of operations on it to create a single image, the "motion profile." This is done by computing the video's vertical "center of intensity" via analysis of pixel brightness. This center is becomes the center axis of a "region of interest" for each frame of the video. The brightness of the pixel rows in this region are summed and normalized, then filtered for low-intensity noise. The now one-dimensional frame is "stacked" on top of all other frames' one-dimensional summary. This creates a single frame which shows the profile of motion over the duration of the video. The result is automatically selected for the next step.

### Step 2: Analyze

#### Choose Input and Output files.

The program will request that the user select a motion profile (as from the last step) be chosen for analysis. If a motion profile has been generated in the current instance of the application it will automatically be selected for this step.

#### Wave Centerline Analysis:

First, the tool will ask the user for the desired name of the output csv. It will then ask the user to crop the motion profile to the desired area. Once the crop is confirmed and the window exited, the tool runs the wave centerline analysis on the cropped image. This finds the center of mass of each bright area for each row, and connect the results to form lines along the center of each wave's motion analysis.

#### Post-Analysis Data Cleaning:

After the analysis, you can highlight areas of the result to delete broken or unnecessary data points.

#### Estimate Turnaround:

Using the output from the previous section, this button will attempt to estimate the wave turnaround point by taking the moving portions of each wave, which will appear as a linear diagonal line segments across the image, and extrapolating the intersection point of the two line segments if they were to continue without changing direction.

The wave centerlines are saved as a CSV file, and the visualization is saved as a PDF image in the output folder.

## Example Workflow:

1. Select raw tiff to process
2. Crop out noise
3. Calibrate Wavelength
4. Generate motion profile
5. Analyze
   1. Select output file
   2. Crop out messy data
   3. Delete broken data
6. Estimate Turnaround

## Packages and Dependencies

NumPy: Array manipulation and numerical calculations.
Matplotlib: For visualizing the image and wave centerlines.
Pillow: For loading and handling image files (specifically TIFF format).
Tkinter: Used for UI

## File Structure

```
.
├── main.py # Entry point of the program
├── tracking.py # Script for wave analysis functions
├── requirements.txt # List of Python dependencies
├── README.md # Project README
```

### Contributing

Feel free to open an issue or submit a pull request if you want to contribute to the project.

### Acknowledgements  

[Bioformats](https://github.com/ome/bioformats?tab=readme-ov-file), a java library for reading and writing life sciences image formats, is used and packaged under the GPL 2.0 to convert .cxd files to .tiff files for ease of use.

### License

This project is licensed under the MIT License. See the LICENSE file for details.
