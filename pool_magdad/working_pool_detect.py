from os import name
from sympy import false
from ultralytics import YOLO
import PIL
from PIL import ImageEnhance
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt


# def edge_detection_sobel(img):
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
#     sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

#     gradient_magnitude = cv2.magnitude(sobelx, sobely)
#     gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

#     cv2.imshow("Sobel Edge Detection", gradient_magnitude)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def Canny_detector(img, weak_th=None, strong_th=None):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     img = cv2.GaussianBlur(img, (5, 5), 1.4)

#     gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
#     gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

#     mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

#     height, width = img.shape

#     mag_max = np.max(mag)
#     if weak_th is None:
#         weak_th = mag_max * 0.1
#     if strong_th is None:
#         strong_th = mag_max * 0.5

#     nms = np.zeros_like(mag)

#     for i_x in range(1, width - 1):
#         for i_y in range(1, height - 1):

#             grad_ang = ang[i_y, i_x]
#             grad_ang = grad_ang % 180

#             if (0 <= grad_ang < 22.5) or (157.5 <= grad_ang <= 180):
#                 before = mag[i_y, i_x - 1]
#                 after = mag[i_y, i_x + 1]

#             elif (22.5 <= grad_ang < 67.5):
#                 before = mag[i_y - 1, i_x + 1]
#                 after = mag[i_y + 1, i_x - 1]

#             elif (67.5 <= grad_ang < 112.5):
#                 before = mag[i_y - 1, i_x]
#                 after = mag[i_y + 1, i_x]

#             else:  # 112.5 - 157.5
#                 before = mag[i_y - 1, i_x - 1]
#                 after = mag[i_y + 1, i_x + 1]

#             if mag[i_y, i_x] >= before and mag[i_y, i_x] >= after:
#                 nms[i_y, i_x] = mag[i_y, i_x]
#             else:
#                 nms[i_y, i_x] = 0

#     result = np.zeros_like(nms)

#     strong = 255
#     weak = 75

#     for i_x in range(width):
#         for i_y in range(height):
#             val = nms[i_y, i_x]

#             if val >= strong_th:
#                 result[i_y, i_x] = strong
#             elif val >= weak_th:
#                 result[i_y, i_x] = weak
#             else:
#                 result[i_y, i_x] = 0

#     canny_img = Canny_detector(img)
#     plt.subplot(1, 2, 1)
#     plt.title('Input Image')
#     plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.subplot(1, 2, 2)
#     plt.title('Canny Edges')
#     plt.imshow(canny_img, cmap='gray')
#     plt.axis('off')
#     plt.show()
#     return result

# def find_contours(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edged = cv2.Canny(gray, 200, 300)
#     contours, hierarchy = cv2.findContours(edged,
#                                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     cv2.imshow('Canny Edges After Contouring', edged)
#     cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
#     cv2.imshow('Contours', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def detect_circles(img):
#     output = img.copy()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Reduce noise
#     gray = cv2.medianBlur(gray, 5)

#     # Detect circles
#     circles = cv2.HoughCircles(
#         gray,
#         cv2.HOUGH_GRADIENT,
#         dp=1,
#         minDist=100,
#         param1=100,
#         param2=40,
#         minRadius=30,
#         maxRadius=60
#     )

#     circles = np.uint16(np.around(circles))
#     x, y, r = circles[0][0]
#     cv2.circle(output, (x, y), r, (255, 0, 0), 2)  # Circle outline
#     cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  # Center point

#     # Show result
#     cv2.imshow('Detected Circle', output)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def colors_sum_no_green(img):
#     rows, cols = img.shape[:2]
#     sum_RGB = 0

#     for i in range(rows):
#         for j in range(cols):
#             sum_RGB += img[i, j][2] + img[i, j][0]
#             img[i, j][1] = 0

#     cv2.imshow('Image (no green)', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     print(sum_RGB)


def yolo_find_balls(img):
    results = model(img , conf = 0.4)
    result = results[0]
    result.names[0] = ""
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
        lst_images.append([sep_ball_img,box,box.conf.item()])   
        
    return lst_images

def find_center_box(box):
    corners = box.xyxy[0].int().tolist()
    x1 = corners[0]
    y1 = corners[1]
    x2 = corners[2]
    y2 = corners[3]
    center_x = (x1+x2)/2
    center_y = (y1+y2)/2
    return (center_x, center_y)

def shade_check(lst_ball_images_and_boxes):
    for img_and_box1 in lst_ball_images_and_boxes:
        ball_box1 = img_and_box1[1]
        center1 = find_center_box(ball_box1)
        for img_and_box2 in lst_ball_images_and_boxes:
            ball_box2 = img_and_box2[1]
            center2 = find_center_box(ball_box2)
            if center1 != center2 and abs(center1[0]-center2[0]) < 50 and abs(center1[1]-center2[1]) < 50:
                print("shade detected")
                if  ball_box1.conf.item() < img_and_box2[2]:
                    img_and_box1 = None
                else:
                    img_and_box2 = None

    return lst_ball_images_and_boxes
           
def convert_to_binary_img_white(img):
    rows, cols = img.shape[:2]
    img_RGB = img.copy()
    img = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2GRAY)
    for i in range(rows):
        for j in range(cols):
            sum_pixel= int(img_RGB[i, j][0]) + int(img_RGB[i, j][1]) + int(img_RGB[i, j][2])
            if sum_pixel > 550:
                img[i, j] = 255
            else:
                img[i, j] = 0

    return img

def convert_to_binary_img_black(img):
    rows, cols = img.shape[:2]
    img_RGB = img.copy()
    img = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2GRAY)
    for i in range(rows):
        for j in range(cols):
            sum_pixel= int(img_RGB[i, j][0]) + int(img_RGB[i, j][1]) + int(img_RGB[i, j][2])
            if sum_pixel > 150:
                img[i, j] = 255
            else:
                img[i, j] = 0

    return img

def stripes_full_or_white(lst_ball_images_and_boxes,lst_stripped_balls,lst_full_balls, black_ball):
    max_white = 0
    min_white = 100000000
    white_ball = (0,0)
    count_stripped = 0
    count_full = 0
    sum_white = 0

    for ball_img in lst_ball_images_and_boxes:
        bi_img = convert_to_binary_img_white(ball_img[0])  #only black and white
        sum_white = 0
        rows, cols = bi_img.shape[:2]
        for i in range(rows):
            for j in range(cols):
                sum_white += int(bi_img[i, j])
        if sum_white > max_white:
            max_white = sum_white
        if sum_white < min_white:
            min_white = sum_white

    for ball_img in lst_ball_images_and_boxes:
        bi_img = convert_to_binary_img_white(ball_img[0])  #only black and white
        sum_white = 0
        rows, cols = bi_img.shape[:2]
        for i in range(rows):
            for j in range(cols):
                sum_white += int(bi_img[i, j])

        ball_box = ball_img[1]
        center = find_center_box(ball_box)
        im = ball_img[0]

        if  sum_white >= max_white:
            white_ball = center
            print("sum_white: ", sum_white, "type: white" )

        elif min_white + 50000 <= sum_white < max_white:
            lst_stripped_balls[count_stripped] = center
            count_stripped += 1
            print("sum_white: ", sum_white, "type: stripped" , "center: ", center)

        elif center != black_ball:
            lst_full_balls[count_full] = center
            count_full += 1
            print("sum_white: ", sum_white, "type: full" , "center: ", center)

        else:
            print( "type: black" )

        
        cv2.imshow('binary Image (Weighted)', bi_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('binary Image (Weighted)', im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return  lst_stripped_balls, lst_full_balls, white_ball

def detect_black(lst_ball_images):
    min_white = 100000000
    black_ball = (0,0)
    for ball_img in lst_ball_images:
        bi_img = convert_to_binary_img_black(ball_img[0])  #only black and white
        sum_white = 0
        rows, cols = bi_img.shape[:2]
        for i in range(rows):
            for j in range(cols):
                sum_white += int(bi_img[i, j])
        ball_box = ball_img[1]
        corners = ball_box.xyxy[0].int().tolist()
        x1 = corners[0]
        y1 = corners[1]
        x2 = corners[2]
        y2 = corners[3]
        
        if  sum_white < min_white:
            min_white = sum_white
            black_ball = ((x1+x2)/2, (y1+y2)/2)


    return black_ball

def create_lst_balls():
    lst_ball_types = []
    for i in range(1, 10):
        lst_ball_types.append((0,0))
    return lst_ball_types

def combine_lists(white_ball, black_ball, lst_stripped_balls, lst_full_balls):
    lst_all_balls = []
    lst_all_balls.append(white_ball)
    for ball in lst_stripped_balls:
        lst_all_balls.append(ball)
    lst_all_balls.append(black_ball)
    for ball in lst_full_balls:
        lst_all_balls.append(ball)
    return lst_all_balls
# -------------------------------------------------------------------- #
img = cv2.imread("pool_magdad\Screenshot_1.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()
lst_stripped_balls = create_lst_balls()
lst_full_balls = create_lst_balls()
white_ball = (0,0)
black_ball = (0,0)

model = YOLO("yoloe-26n-seg.pt")
model.set_classes(["billiard circular ball on green background maybe near other balls"])


result = yolo_find_balls(img)
lst_ball_images_and_boxes = separate_balls(result)
lst_ball_images_and_boxes = shade_check(lst_ball_images_and_boxes)
black_ball = detect_black(lst_ball_images_and_boxes)
lst_stripped_balls, lst_full_balls, white_ball = stripes_full_or_white(lst_ball_images_and_boxes, lst_stripped_balls, lst_full_balls, black_ball)
print("white ball: ", white_ball)
print("stripped balls: ", lst_stripped_balls)
print("black ball: ", black_ball)
print("full balls: ", lst_full_balls)
# lst_all_balls_pixels = combine_lists(white_ball, black_ball, lst_stripped_balls, lst_full_balls)
# print(lst_all_balls_pixels)
