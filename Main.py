import cv2
import time
from djitellopy import Tello

tello = Tello()

tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

tello.takeoff()
tello.move_up(200)
time.sleep(10) # Wait 20 seconds

cv2.imwrite("GangGang.png", frame_read.frame)

tello.land()
