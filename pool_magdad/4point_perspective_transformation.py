from ultralytics import YOLO
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import os

TABLE_WIDTH = 255  # (cm)
TABLE_HEIGHT = 129  # (cm)
SCREEN_WIDTH = TABLE_WIDTH * 4
SCREEN_HEIGHT = TABLE_HEIGHT * 4
model = YOLO("yoloe-26n-seg.pt")
model.set_classes(["pool table from a 45 degree downward view"])


def find_center_box(box):
    corners = box.xyxy[0].int().tolist()
    x1 = corners[0]
    y1 = corners[1]
    x2 = corners[2]
    y2 = corners[3]
    center_x = (x1+x2)/2
    center_y = (y1+y2)/2
    return (center_x, center_y)

def YOLO_triengulation(img):
        
    model.set_classes(["pool table from a 45 degree downward view"])
    results = model(img)
    result = results[0]
    result.show()
    assert img is not None, "file could not be read, check with os.path.exists()"
    rows, cols, ch = img.shape

    corners = result.boxes[0].xyxy[0].int().tolist()
    x1 = corners[0]
    y1 = corners[1]
    x2 = corners[2]
    y2 = corners[3]
    top_left, top_right, bottom_right, bottom_left = (x1, y1), (x2, y1), (x2, y2), (x1, y2)
    print(f"Top left: {top_left}, Top right: {top_right}, Bottom right: {bottom_right}, Bottom left: {bottom_left}")

    pts1 = np.float32([bottom_left, bottom_right, top_right, top_left])
    pts2 = np.float32([[0, 0], [SCREEN_WIDTH, 0], [SCREEN_WIDTH, SCREEN_HEIGHT], [0, SCREEN_HEIGHT]])

    M = cv.getPerspectiveTransform(pts1, pts2)

    dst = cv.warpPerspective(img, M, (SCREEN_WIDTH, SCREEN_HEIGHT))
    plt.subplot(121), plt.imshow(img), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()

    return dst

start = time.time() 
img = cv.imread('pool_magdad\side_view.jpg')
results = model(img)
result = results[0]
result.show()
corners = result.boxes[0].xyxy[0].int().tolist()
x1 = corners[0]
y1 = corners[1]
x2 = corners[2]
y2 = corners[3]
sep_ball_img = img[y1:y2, x1:x2]     
cv.imshow("sep_ball_img", sep_ball_img)
cv.waitKey(0)
cv.destroyAllWindows()

end = time.time()
print(f"Time taken: {end - start} seconds")


def FAST():
    root = os.getcwd()
    # TODO : https://www.youtube.com/watch?v=zJUiZUNKhCI
