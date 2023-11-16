# from ultralytics import YOLO
import cv2
import yolov8_person_opencv as yp

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("Frame não encontrado")

    yp.ProcessaFrame(frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
