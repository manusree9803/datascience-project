This project uses YOLOv3 and OpenCV to perform real-time object detection through your webcam.

Features
Detects multiple objects in real-time

Highlights objects with bounding boxes and labels

Uses Non-Maximum Suppression (NMS) to reduce overlapping boxes

Requirements
Python 3.x

OpenCV (opencv-python)

NumPy

Setup Instructions
Install the dependencies:

bash
Copy
Edit
pip install opencv-python numpy
Download YOLOv3 files:

yolov3.weights

yolov3.cfg

coco.names

Place all three files in the same directory as your Python script.

Run the code:

bash
Copy
Edit
python your_script_name.py
Press q to exit the webcam window.

Folder Structure
bash
Copy
Edit
/project-folder
  ├── yolov3.cfg
  ├── yolov3.weights
  ├── coco.names
  ├── detect.py  # (your Python code)
  └── README.md
Output
Real-time webcam feed with detected objects boxed and labeled.

Detection confidence shown next to labels.

