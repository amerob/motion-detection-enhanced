import cv2
import datetime
import time
import winsound
import os 

# Get the current working directory (where the code is located)
script_directory = os.path.dirname(os.path.abspath(__file__))

# Initialize the webcam
webcam = cv2.VideoCapture(0)

# Constraints
MIN_CONTOUR_AREA = 2500
MOTION_TIMEOUT = 10
VIDEO_FPS = 24

# Variables
motion_detected = False
video_writer = None
last_motion_time = time.time()

while webcam.isOpened():
    ret, frame1 = webcam.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Convert frame to grayscale and apply Gaussian Blur
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold the blurred frame
    _, thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Iterate through contours to detect significant motion
    for contour in contours:
        if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
            continue
        
        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        if not motion_detected:
            motion_detected = True
            last_motion_time = time.time()
            
            # Record the current timestamp for the video file name
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_filename = os.path.join(script_directory, f"motion_{current_time}.avi")
            
            # Initialize the video writer
            video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), VIDEO_FPS, (frame1.shape[1], frame1.shape[0]))
        
        # Write the frame to the video file
        video_writer.write(frame1)
    
    # If no motion detected for the specified timeout, stop recording
    if motion_detected and len(contours) == 0:
        elapsed_time = time.time() - last_motion_time
        if elapsed_time >= MOTION_TIMEOUT:
            motion_detected = False
            if video_writer:
                video_writer.release()
                video_writer = None
    
    # Display the current frame
    cv2.imshow("Webcam Feed", frame1)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release resources
webcam.release()
if video_writer:
    video_writer.release()

cv2.destroyAllWindows()
