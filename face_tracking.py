from djitellopy import Tello
import cv2
import numpy as np
import dlib
from imutils import face_utils

# Initialize the face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Constants
width = 640
height = 480
deadZone = 50  # Adjust this dead zone as needed
guidelines_enabled = True  # Toggle guidelines display with 'g' key

# Connect to Tello
me = Tello()
me.connect()

# Start video stream
me.streamoff()
me.streamon()

# Take off
me.takeoff()

# Main loop
while True:
    # Get the frame from Tello
    frame_read = me.get_frame_read()
    frame = frame_read.frame
    img = cv2.resize(frame, (width, height))
    imgContour = img.copy()

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    if not faces:
        # If no faces detected, hover in place
        me.left_right_velocity = 0
        me.for_back_velocity = 0
        me.up_down_velocity = 0
        me.yaw_velocity = 0
    else:
        # Get the first detected face (you can modify this to track multiple faces)
        face = faces[0]

        # Get the facial landmarks
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Calculate the center of the face (you can use different points for tracking)
        cx = int((shape[0][0] + shape[16][0]) / 2)

        # Calculate the direction to move the drone to keep the face in the center
        if cx < int(width / 2) - deadZone:
            cv2.putText(imgContour, " GO LEFT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            me.left_right_velocity = -30
        elif cx > int(width / 2) + deadZone:
            cv2.putText(imgContour, " GO RIGHT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
            me.left_right_velocity = 30
        else:
            # Face is in the desired position, hover in place
            me.left_right_velocity = 0

    # Continue flying up until reaching 100cm
    if me.get_height() < 100:
        me.up_down_velocity = 20  # Adjust the velocity as needed
    else:
        me.up_down_velocity = 0

    # Display guidelines if enabled
    if guidelines_enabled:
        cv2.line(imgContour, (int(width / 2) - deadZone, 0), (int(width / 2) - deadZone, height), (255, 255, 0), 3)
        cv2.line(imgContour, (int(width / 2) + deadZone, 0), (int(width / 2) + deadZone, height), (255, 255, 0), 3)
        cv2.circle(imgContour, (int(width / 2), int(height / 2)), 5, (0, 0, 255), 5)
        cv2.line(imgContour, (0, int(height / 2) - deadZone), (width, int(height / 2) - deadZone), (255, 255, 0), 3)
        cv2.line(imgContour, (0, int(height / 2) + deadZone), (width, int(height / 2) + deadZone), (255, 255, 0), 3)

    # Display the image with tracking information
    cv2.imshow('Face Tracking', imgContour)

    # Send velocity commands to the Tello
    me.send_rc_control(me.left_right_velocity, me.for_back_velocity, me.up_down_velocity, me.yaw_velocity)

    # Handle key presses
    # Break the loop when 'keys pressed' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        me.land()
        break

    if cv2.waitKey(0) & 0xFF == ord('w'):
        me.move_forward(30)

    if cv2.waitKey(0) & 0xFF == ord('s'):
        me.move_back(30)

    if cv2.waitKey(0) & 0xFF == ord('a'):
        me.move_left(30)

    if cv2.waitKey(0) & 0xFF == ord('d'):
        me.move_right(30)

    if cv2.waitKey(0) & 0xFF == ord('g'):
        guidelines_enabled = not guidelines_enabled


# Clean up
cv2.destroyAllWindows()

# Land the drone and release the video stream
me.land()
me.streamoff()

# Ensure the drone is landed before exiting
while not me.land():
    pass

# Release the drone resources and safely exit the program
me.release()
