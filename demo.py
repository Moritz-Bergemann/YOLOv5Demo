# Get the libraries we need to run the program
import numpy as np  # numpy
import cv2 as cv    # OpenCV
import torch        # PyTorch

# Get input from device 0 (the webcam)
CAMERA_DEVICE_NUM = 0

# Download YOLOv5 machine learning model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Get the webcam video capture stream
capture = cv.VideoCapture(CAMERA_DEVICE_NUM)

# For every frame in the image...
while capture.isOpened():
    ret, frame = capture.read()

    # End the program if we didn't get a new frame
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
        
    # Use the model to run a prediction on the frame
    results = model(frame)

    # Get the prediction results and format them
    result_img = np.array(results.render())
    result_img = np.squeeze(result_img)

    # Show the results on screen
    cv.imshow('YOLOv5 computer vision demo', result_img)
    
    # Allow the program to be quit using the 'q' key
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
