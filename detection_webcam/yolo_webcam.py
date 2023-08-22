#
from ultralytics import YOLO

import cv2
import cvzone

# to read video feed from web cam
cap = cv2.VideoCapture(0)

# to set width and height dimension
cap.set(3, 640)
cap.set(4, 480)

model = YOLO('yolov8n.pt')

while True:
    # read next video frame and store the value in img?
    success, img = cap.read()

    results = model(img, stream=True)
    # get the bounding box of each result
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, x2, y2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

    # display the image in the window
    cv2.imshow("Image", img)
    # do nothing until a key is pressed
    cv2.waitKey(1)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
