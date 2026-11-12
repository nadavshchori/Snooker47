from ultralytics import YOLO
import PIL
from PIL import ImageEnhance
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def separate_balls(result):
    for box in result.boxes:
        name = result.names[int(box.cls)]
        corners = box.xyxy[0].int().tolist()
        print()
        x1 = corners[0]
        y1 = corners[1]
        x2 = corners[2]
        y2 = corners[3]
        sep_ball = img[y1:y2, x1:x2]
        cv2.imshow("Separated ball", sep_ball)
        cv2.waitKey(0)
        edge_det(sep_ball)

def show_separated_balls(img):
    pass


def edge_det(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

    gradient_magnitude = cv2.magnitude(sobelx, sobely)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    cv2.imshow("Sobel Edge Detection", gradient_magnitude)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Canny_detector(img, weak_th=None, strong_th=None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (5, 5), 1.4)

    gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
    gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    height, width = img.shape

    mag_max = np.max(mag)
    if weak_th is None:
        weak_th = mag_max * 0.1
    if strong_th is None:
        strong_th = mag_max * 0.5

    nms = np.zeros_like(mag)

    for i_x in range(1, width - 1):
        for i_y in range(1, height - 1):

            grad_ang = ang[i_y, i_x]
            grad_ang = grad_ang % 180

            if (0 <= grad_ang < 22.5) or (157.5 <= grad_ang <= 180):
                before = mag[i_y, i_x - 1]
                after = mag[i_y, i_x + 1]

            elif (22.5 <= grad_ang < 67.5):
                before = mag[i_y - 1, i_x + 1]
                after = mag[i_y + 1, i_x - 1]

            elif (67.5 <= grad_ang < 112.5):
                before = mag[i_y - 1, i_x]
                after = mag[i_y + 1, i_x]

            else:  # 112.5 - 157.5
                before = mag[i_y - 1, i_x - 1]
                after = mag[i_y + 1, i_x + 1]

            if mag[i_y, i_x] >= before and mag[i_y, i_x] >= after:
                nms[i_y, i_x] = mag[i_y, i_x]
            else:
                nms[i_y, i_x] = 0

    result = np.zeros_like(nms)

    strong = 255
    weak = 75

    for i_x in range(width):
        for i_y in range(height):
            val = nms[i_y, i_x]

            if val >= strong_th:
                result[i_y, i_x] = strong
            elif val >= weak_th:
                result[i_y, i_x] = weak
            else:
                result[i_y, i_x] = 0

    return result

def show_canny(img):
    canny_img = Canny_detector(img)
    plt.subplot(1, 2, 1)
    plt.title('Input Image')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title('Canny Edges')
    plt.imshow(canny_img, cmap='gray')
    plt.axis('off')
    plt.show()


def binary_img(img):
    pass

def find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(edged,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.imshow('Canny Edges After Contouring', edged)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
    cv2.imshow('Contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def yolo_find_balls(img):
    results = model(img)
    result = results[0]
    return result




# -------------------------------------------------------------------- #
img = cv2.imread('pool.jpg')
model = YOLO("yoloe-26n-seg.pt")
model.set_classes(["ball on green background"])


result = yolo_find_balls(img)
# result.show()
separate_balls(result)
# edge_det(img)
# find_contours(img)



