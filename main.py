# from ultralytics import YOLO
import time

import cv2
import yolov8_person_opencv as yp

cap = cv2.VideoCapture(0)

pre_timeframe = 0
new_timeframe = 0

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("Frame não encontrado")

    new_timeframe = time.time()
    fps = 1/(new_timeframe-pre_timeframe)
    pre_timeframe = new_timeframe
    fps = int(fps)
    yp.ProcessaFrame(frame, fps)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
