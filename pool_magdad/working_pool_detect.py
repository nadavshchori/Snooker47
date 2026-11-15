from ultralytics import YOLO
import PIL
from PIL import ImageEnhance
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


def edge_detection_sobel(img):
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
    return result

def find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 200, 300)
    contours, hierarchy = cv2.findContours(edged,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.imshow('Canny Edges After Contouring', edged)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imshow('Contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_circles(img):
    output = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    gray = cv2.medianBlur(gray, 5)

    # Detect circles
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,
        param1=100,
        param2=40,
        minRadius=30,
        maxRadius=60
    )

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]
    cv2.circle(output, (x, y), r, (255, 0, 0), 2)  # Circle outline
    cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Center point

    # Show result
    cv2.imshow('Detected Circle', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def colors_sum_no_green(img):
    rows, cols = img.shape[:2]
    sum_RGB = 0

    for i in range(rows):
        for j in range(cols):
            sum_RGB += img[i, j][2] + img[i, j][0]
            img[i, j][1] = 0

    cv2.imshow('Image (no green)', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(sum_RGB)




def yolo_find_balls(img):
    results = model(img)
    result = results[0]
    result.names[0] = ""
    print()
    print(result.keypoints)
    result.show()
    return result

def separate_balls(result):
    lst_images = []
    for box in result.boxes:
        corners = box.xyxy[0].int().tolist()
        x1 = corners[0]
        y1 = corners[1]
        x2 = corners[2]
        y2 = corners[3]
        sep_ball_img = img[y1:y2, x1:x2]
        lst_images.append(sep_ball_img)
    return lst_images

def convert_to_binary_img(img):
    rows, cols = img.shape[:2]
    img_RGB = img.copy()
    img = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2GRAY)

    for i in range(rows):
        for j in range(cols):
            sum_pixel= int(img_RGB[i, j][0]) + int(img_RGB[i, j][1]) + int(img_RGB[i, j][2])
            if sum_pixel > 600:
                img[i, j] = 255
            else:
                img[i, j] = 0


    return img

def stripes_or_full(results,lst_ball_images):
    ball_type = ""
    for ball_img in lst_ball_images:
        bi_img = convert_to_binary_img(ball_img)  #only black and white
        sum_white = 0
        rows, cols = bi_img.shape[:2]
        for i in range(rows):
            for j in range(cols):
                sum_white += int(bi_img[i, j])
        if  sum_white >= 100000:
            ball_type = "white"

        elif 40000 <= sum_white <= 100000:
            ball_type = "stripped"

        else:
            ball_type = "full"
        print(sum_white,ball_type)
        cv2.imshow('binary Image (Weighted)', bi_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



# -------------------------------------------------------------------- #
img = cv2.imread('pool.jpg')

model = YOLO("yoloe-26n-seg.pt")
model.set_classes(["billiard round ball on green background maybe near other balls"])


result = yolo_find_balls(img)
lst_ball_images = separate_balls(result)
stripes_or_full(result,lst_ball_images)

