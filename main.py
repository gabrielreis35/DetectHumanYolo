import time
from ultralytics import YOLO
import cv2
import random

my_file = open("utils/coco.txt", "r")

# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

# Vals to resize video frames | small frame optimise the run
frame_wid = 640
frame_hyt = 480

pre_timeframe = 0
new_timeframe = 0

pre_time = 0
new_time = 0
time_process = 0


def ProcessaFrame(frame, fps):
    pre_time = time.time()
    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.80, save=False)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    # print(DP)

    cv2.putText(frame, str(fps), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            # print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  # returns one box
            clsID = box.cls.numpy()[0]
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]
            if clsID == 0:
                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )
                new_time = time.time()
                time_process = new_time - pre_time
                print(time_process)
            else:
                continue

    cv2.imshow("ObjectDetection", frame)

cap = cv2.VideoCapture(0)

while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("Frame não encontrado")

    new_timeframe = time.time()
    fps = 1/(new_timeframe-pre_timeframe)
    pre_timeframe = new_timeframe
    fps = int(fps)
    ProcessaFrame(frame, fps)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
