import cv2
from djitellopy import Tello

tello = Tello()

tello.connect()
tello.streamon()
frame_read = tello.get_frame_read()

tello.takeoff()
tello.move_up(160)
cv2.imwrite("GangGang.png", frame_read.frame)

tello.land()
