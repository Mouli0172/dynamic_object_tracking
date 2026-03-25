# Dynamic Object Tracking

## 1. What is this Project?
This project is a Computer Vision tracking script built with Python and OpenCV. It is designed to isolate and track a rolling tire in a highly dynamic video environment. 

The core tracking engine utilizes the **CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) Tracker**, which provides high accuracy for objects undergoing significant scaling and rotation. To handle complex edge cases—like heavy shadow blending, occlusion, and rapid camera panning—the script features a custom-built morphological fallback scanner that automatically re-acquires the target if the primary tracker fails.

## 2. Environment Setup

**Create the virtual environment:**
```bash
# For Windows
python -m venv venv

# For macOS/Linux
python3 -m venv venv
```
# Activate the virtual environment:

```bash
# For Windows
venv\Scripts\activate

# For macOS/Linux
source venv/bin/activate
```
## 3. Installing Dependencies
Once your virtual environment is active, install the necessary packages using the provided requirements file.
```bash
pip install -r requirements.txt
```
## 4. How to Run the Code
1. Ensure the source video is located in the root directory alongside auto_object_tracker.py.

2. Run the tracker script from your terminal:

```bash
python auto_object_tracker.py
```
3. The video window will open and begin playing. Wait for the exact moment the tire is fully released from the subject's hands, then press 'p' on your keyboard to pause the video.

4. Click and drag your mouse to draw a tight bounding box around the tire.

5. Press SPACE or ENTER to initialize the tracking sequence.

6.The script will process the video dynamically and save the final result as tracked_output.mp4 in your project folder.
