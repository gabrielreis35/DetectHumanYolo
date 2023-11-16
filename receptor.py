# import cv2
from vidstream import StreamingServer
import threading

receiving = StreamingServer('192.168.15.2', 8000)
# receiving.start_server()
th = threading.Thread(target=receiving.start_server)
th.start()

# cv2.imshow("frame", receiving)

while input("") != "stop":
    continue

receiving.stop_server()