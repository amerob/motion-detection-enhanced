import cv2
import datetime
import time
import winsound
import os 

# Get the current working directory (where the code is located)
script_directory = os.path.dirname(os.path.abspath(__file__))

#innit the webcamq
webcam = cv2.VideoCapture(0)

#constraints
MIN_CONTOUR_AREA =2500
MOTION_TIMEOUT = 10
Video_FPS=24
#variables
motion_detected = False
video_writer = None
last_motion_time = time.time()

while webcam.isOpened():
    ret, frame1 = webcam.read()
    ret, frame2 = webcam.read()
    diff = cv2.absdiff(frame1, frame2)
    gray=cv2.cvtColor(diff,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    noise,thresh=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated=cv2.dilate(thresh,None,3)
    border, noise2 = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(frame1,border,-1,(0,255,0),2)
    #ittirating thru the contours to detect significan motion
    for c in border:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA :
            continue
        x,y,w,h=cv2.boundingRect(c) #making the border rectangular
        # winsound.Beep(500,200) #freq, time in ms
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0),2)
        if not motion_detected:
            motion_detected = True
            last_motion_time = time.time()
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # Create the full path for the video file
            video_filename = os.path.join(
                script_directory, f"motion_{current_time}.avi")
            video_writer = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(
                'M', 'J', 'P', 'G'), Video_FPS, (frame1.shape[1], frame1.shape[0]))

        # Record the frame to the video file
        video_writer.write(frame1)

    # If motion is no longer detected, stop recording and release the video writer
    if motion_detected and len(border) == 0:
        elapsed_time = time.time() - last_motion_time
        if elapsed_time >= MOTION_TIMEOUT:
            motion_detected = False
            video_writer.release()
            video_writer = None
    if cv2.waitKey(10)==ord('q'):
        break
    cv2.imshow("my cam",frame1)