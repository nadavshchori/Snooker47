from ultralytics import YOLO
import PIL
from PIL import ImageEnhance
import cv2
import numpy as np


model = YOLO("yoloe-26n-seg.pt")
model.set_classes(["ball on green background"])


img = cv2.imread('pool.jpg')

results = model(img)
result = results[0]

# Each detection carries a class, a confidence score, and a box in pixels
for box in result.boxes:
    name = result.names[int(box.cls)]
    corners = box.xyxy[0].int().tolist()
    print()
    x1 = corners[0]
    y1 = corners[1]
    x2 = corners[2]
    y2 = corners[3]
    image = cv2.imread('pool.jpg')
    roi = image[y1:y2, x1:x2]
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)
    print(f"{name} {box.conf.item():.2f} {corners}")

#
# boxes = result.boxes  # Boxes object for bounding box outputs
# masks = result.masks  # Masks object for segmentation masks outputs
# keypoints = result.keypoints  # Keypoints object for pose outputs
# probs = result.probs  # Probs object for classification outputs
# obb = result.obb  # Oriented boxes object for OBB outputs
result.show()  # display to screen
# save to disk


