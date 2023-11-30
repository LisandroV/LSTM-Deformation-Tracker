"""
This Script reproduces the video showing the position of the finger according to the finger_position.txt file
"""
import sys
import os
import numpy as np
import cv2

sys.path.append("./")
from src.read_data.finger_position_reader import read_finger_positions_file


DATA_DIR: str = "data/sponge_shortside"

finger_positions_file = os.path.join(DATA_DIR, "finger_position.txt")
video_file = os.path.join(DATA_DIR, "video.mp4")

circle_color = (255, 255, 0)
circle_radio = 30
video_speed = 40  # the smaller the faster
positions: np.ndarray = read_finger_positions_file(finger_positions_file)

video_capture = cv2.VideoCapture(video_file)
pause = True
for point in positions:
    any_problem, frame = video_capture.read()
    if not any_problem:
        raise Exception("There was a problem while reading the video")
    cv2.circle(frame, (int(point[0]), int(point[1])), circle_radio, circle_color)
    print(point)
    cv2.imshow("Finger position", frame)

    # Exit if ESC pressed
    k = (
        cv2.waitKey(video_speed) & 0xFF
    )  # will get stucked here in the end if ESC is not pressed
    if k == 27:
        pause = False

if pause:
    cv2.waitKey(-1)
video_capture.release()
cv2.destroyAllWindows()
