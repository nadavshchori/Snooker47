import cv2
import numpy as np
from ultralytics import YOLO
import copy
import time
from scipy import signal

import math

TABLE_WIDTH = 255  # (cm)
TABLE_HEIGHT = 129  # (cm)
SCREEN_WIDTH = TABLE_WIDTH * 4
SCREEN_HEIGHT = TABLE_HEIGHT * 4


# ---------------------------------------------
# Corner Detection
# ---------------------------------------------

# ->
def remove_no_green_pixels(img):
    mask = (img[:, :, 1] > 150) & (img[:, :, 0] > 60) & (img[:, :, 2] > 80) & (img[:, :, 2] < 150)
    out = np.zeros(img.shape[0:3], dtype=int)
    out[mask] = 255

    return out


def preProcessing(imgGray):
    """ A function used to pre-process gray images to make it suitable to work with.

        Parameters: gray img (matrix): matrix image
        Returns: imgThres(matrix): Processed matrix image
    """
    img_uint8 = imgGray.astype(np.uint8)
    imgGray = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)
    # 2 - Denoise image
    imgBlur = cv2.fastNlMeansDenoising(imgGray, h=10)
    # imgBlur2 = cv2.GaussianBlur(imgGray,(5,5),1)  # another way of denoising

    # 3 - Removing unnecessary details such as writing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = cv2.morphologyEx(imgBlur, cv2.MORPH_CLOSE, kernel, iterations=5)

    # 4 - Edge detection using canny
    imgCanny = cv2.Canny(morph, 75, 200)
    # imgCanny2 = cv2.Canny(imgBlur,75,200) # without morph

    # 5- Extra preprocessing Dilation and Erosion( Mostly we don't needs them )
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres


def cornerDetector(img):
    """ A function used to highlight all the contours in the image and
        find the corners of the biggest one.

        Parameters: img (matrix): matrix image
        Returns: corners(Array): return the biggest contours corner pixel points
    """
    # biggest contours corner points saved in an array
    corners = np.array([])
    # initalize the max area
    maxArea = 0
    # calling method find contours to find all contours in the image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # loop through each contour in the image
    for cnt in contours:
        # svae its area size
        area = cv2.contourArea(cnt)
        # if area smaller than 6k ignore it
        if area > 6000:
            # if we want to see all contours uncomment this
            # cv2.drawContours(imgCopy, cnt, -1, (0, 0, 255), 2)
            # Calculate the contour curves length we need it for approximation points (the bigger the accurate)
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            # An approximation of the edges points ( corner points)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # if this contour has bigger area and it has 4 points edges ( meaning it is a square such as paper)
            if area > maxArea and len(approx) == 4:
                corners = approx
                maxArea = area
    return corners


def sortCorners(corners):
    """ A function used to reorder the corners point to
        match the exact order required for the wrapper function.

        Parameters: corners (array): corners of the paper
        Returns: sortedCorners(array): returns sortedd array of corners pixel points
    """
    # Just reshaping before manuplations
    corners = corners.reshape((4, 2))
    # create empty new corners
    sortedCorners = np.zeros((4, 1, 2), np.int32)
    # sum each corner width + height => smallest is at the top left[0,0]
    # biggest will be bottom right [width, height]
    # we want to make the corner like this order [[0, 0], [width, 0], [0, height], [width, height]]
    add = corners.sum(1)
    # smallest
    sortedCorners[0] = corners[np.argmin(add)]
    # biggest
    sortedCorners[3] = corners[np.argmax(add)]
    # The difference between width - height for the remaining points:
    #  if the result (height - width) is negative(minimum) => will be at [width, 0]
    #  if the result (height - width) is positive(maximum) => will be at [0, height]
    diff = np.diff(corners, axis=1)
    sortedCorners[1] = corners[np.argmin(diff)]
    sortedCorners[2] = corners[np.argmax(diff)]

    return sortedCorners


def warper(img, corners, width, height):
    """ A function used to transform the document into fully flat in x and y directions.

        Parameter: img (matrix): matrix image
        Parameter: corners (array): corners of the paper

        Returns: corners(array): fully flat in x and y image
    """
    # Calling sort corner to satisfy wrapper function
    corners = sortCorners(corners)
    # set the points
    points1 = np.float32(corners)
    points2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Wrapper transform computer transform matrix
    matrix = cv2.getPerspectiveTransform(points1, points2)
    # result
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result


# ->
def binary_kernel(img):
    '''
    gets a gray scaled image
    finds areas with white
    returns gray scaled image (with binary values 0 or 255)
    '''

    PIXEL_AMOUNT_THRESHOLD = 5

    K_SIZE = 15

    kernel = np.ones((K_SIZE, K_SIZE, K_SIZE))

    sum_mat = signal.convolve(img, kernel, mode="same")
    # sum_mat = signal.convolve2d(img, kernel, mode="same")
    # sum_mat = (K_SIZE**2) * cv2.blur(img, K_SIZE)

    mask = sum_mat[:, :] >= PIXEL_AMOUNT_THRESHOLD * 255

    img[mask] = 255
    return img


def format_corners(corners):
    output = []
    for i in range(corners.shape[0]):
        output.append([int(corners[i, 0, 0]), int(corners[i, 0, 1])])

    output.sort(key=lambda x: x[0])
    return output


def corners_detection(img):
    # Output images size
    width = 960
    height = 1280
    # Reading input images
    new_img = remove_no_green_pixels(img)
    img = binary_kernel(new_img)
    # resizing image (for better quality)
    img1 = cv2.resize(img, (width, height))
    # Process image before detecting
    processedImg = preProcessing(img1)

    # Get corners points
    corners = cornerDetector(processedImg)
    return format_corners(corners)


# ---------------------------------------------
# Image Transformation
# ---------------------------------------------

def four_point_perspective_transformation(src_img, corners):
    pts1 = np.float32(corners)
    pts2 = np.float32([[0, 0], [SCREEN_WIDTH, 0], [0, SCREEN_HEIGHT], [SCREEN_WIDTH, SCREEN_HEIGHT]])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(src_img, M, (SCREEN_WIDTH, SCREEN_HEIGHT))
    return dst


# ---------------------------------------------
# Ball Detection
# ---------------------------------------------

def yolo_find_balls(img, model):
    results = model(img, conf=0.4)
    result = results[0]
    result.names[0] = ""
    result.show()
    return result


def separate_balls(result, img):
    lst_images = []
    for box in result.boxes:
        corners = box.xyxy[0].int().tolist()
        x1 = corners[0]
        y1 = corners[1]
        x2 = corners[2]
        y2 = corners[3]
        sep_ball_img = img[y1:y2, x1:x2]
        lst_images.append([sep_ball_img, box, box.conf.item()])

    return lst_images


def find_center_box(box):
    corners = box.xyxy[0].int().tolist()
    x1 = corners[0]
    y1 = corners[1]
    x2 = corners[2]
    y2 = corners[3]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return (center_x, center_y)


def shade_check(lst_ball_images_and_boxes):
    delete_dict = {}
    for i in range(len(lst_ball_images_and_boxes)):
        ball_box1 = lst_ball_images_and_boxes[i][1]
        center1 = find_center_box(ball_box1)
        for j in range(len(lst_ball_images_and_boxes)):
            ball_box2 = lst_ball_images_and_boxes[j][1]
            center2 = find_center_box(ball_box2)
            if center1 != center2 and abs(center1[0] - center2[0]) < 30 and abs(center1[1] - center2[1]) < 30:
                print("shade detected")
                if ball_box1.conf.item() < ball_box2.conf.item():
                    delete_dict[i] = lst_ball_images_and_boxes[i]
                else:
                    delete_dict[j] = lst_ball_images_and_boxes[j]

    for key in delete_dict:
        del lst_ball_images_and_boxes[key]

    return lst_ball_images_and_boxes


def convert_to_binary_img_white(img):
    rows, cols = img.shape[:2]
    img_RGB = img.copy()
    img = cv2.cvtColor(img_RGB, cv2.COLOR_BGR2GRAY)
    for i in range(rows):
        for j in range(cols):
            sum_pixel = int(img_RGB[i, j][0]) + int(img_RGB[i, j][1]) + int(img_RGB[i, j][2])
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
            sum_pixel = int(img_RGB[i, j][0]) + int(img_RGB[i, j][1]) + int(img_RGB[i, j][2])
            if sum_pixel > 150:
                img[i, j] = 255
            else:
                img[i, j] = 0

    return img


def stripes_full_or_white(lst_ball_images_and_boxes, lst_stripped_balls, lst_full_balls, black_ball):
    white_ball = (0, 0)
    count_stripped = 0
    count_full = 0
    sum_white = 0
    balls_white = []

    for ball_img in lst_ball_images_and_boxes:
        bi_img = convert_to_binary_img_white(ball_img[0])  # only black and white
        sum_white = 0
        rows, cols = bi_img.shape[:2]
        for i in range(rows):
            for j in range(cols):
                sum_white += int(bi_img[i, j])

        balls_white.append(sum_white)

    copy_whites = copy.deepcopy(balls_white)
    copy_whites.sort()
    max_white = copy_whites[-1]
    third_max = copy_whites[-3]
    min_white = copy_whites[0]
    if third_max < (max_white + min_white) / 2:
        third_max = max_white
    mid = (third_max + min_white) / 2

    for j in range(len(lst_ball_images_and_boxes)):
        ball_box = lst_ball_images_and_boxes[j][1]
        center = find_center_box(ball_box)
        im = lst_ball_images_and_boxes[j][0]

        if balls_white[j] >= max_white:
            white_ball = center
            print("sum_white: ", balls_white[j], "type: white")

        elif mid <= balls_white[j] < max_white:
            lst_stripped_balls[count_stripped] = center
            count_stripped += 1
            print("sum_white: ", balls_white[j], "type: stripped", "center: ", center)

        elif center != black_ball:
            lst_full_balls[count_full] = center
            count_full += 1
            print("sum_white: ", balls_white[j], "type: full", "center: ", center)

        # cv2.imshow('binary Image (Weighted)', convert_to_binary_img_white(im))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(im)
        # cv2.imshow('binary Image (Weighted)', im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    return lst_stripped_balls, lst_full_balls, white_ball


def detect_black(lst_ball_images):
    min_white = 100000000
    black_ball = (0, 0)
    for ball_img in lst_ball_images:
        bi_img = convert_to_binary_img_black(ball_img[0])  # only black and white
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

        if sum_white < min_white:
            min_white = sum_white
            black_ball = ((x1 + x2) / 2, (y1 + y2) / 2)

    return black_ball


def create_lst_balls():
    lst_ball_types = []
    for i in range(1, 20):
        lst_ball_types.append((0, 0))
    return lst_ball_types


def combine_lists(white_ball, black_ball, lst_stripped_balls, lst_full_balls):
    lst_all_balls = []
    lst_all_balls.append(white_ball)

    for i in range(0, 7):
        lst_all_balls.append(lst_full_balls[i])

    lst_all_balls.append(black_ball)

    for j in range(0, 7):
        lst_all_balls.append(lst_stripped_balls[j])

    return lst_all_balls


def ball_detection(img, model):
    lst_stripped_balls = create_lst_balls()
    lst_full_balls = create_lst_balls()
    white_ball = (0, 0)
    black_ball = (0, 0)

    result = yolo_find_balls(img, model)
    lst_ball_images_and_boxes = separate_balls(result, img)
    lst_ball_images_and_boxes = shade_check(lst_ball_images_and_boxes)
    black_ball = detect_black(lst_ball_images_and_boxes)
    lst_stripped_balls, lst_full_balls, white_ball = stripes_full_or_white(lst_ball_images_and_boxes,
                                                                           lst_stripped_balls,
                                                                           lst_full_balls, black_ball)
    lst_all_balls_pixels = combine_lists(white_ball, black_ball, lst_stripped_balls, lst_full_balls)
    return lst_all_balls_pixels


def ball_lst_from_pixel_to_cm(lst_all_balls_pixels):
    new_lst = []
    for i in range(len(lst_all_balls_pixels)):
        new_lst.append((lst_all_balls_pixels[i][0] / 4, lst_all_balls_pixels[i][1] / 4))
    return new_lst


# ---------------------------------------------
# Main
# ---------------------------------------------
# if __name__ == '__main__':


def main():
    tot_time = time.time()

    yolo_time = time.time()
    model = YOLO("yoloe-26n-seg.pt")
    model.set_classes(["billiard ball on green background maybe near other balls"])
    end_yolo_time = time.time()
    print(f'yolo time: {end_yolo_time - yolo_time:.2f} seconds')

    src_path = input("Enter the source image path: ")
    src_img = cv2.imread(src_path)

    # cv2.imshow("src_img", src_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    corners_time = time.time()
    corners = corners_detection(copy.deepcopy(src_img))
    end_corners_time = time.time()
    print(f'find corners time: {end_corners_time - corners_time:.2f} seconds')

    trian_time = time.time()
    stretched_img = four_point_perspective_transformation(src_img, corners)
    end_trian_time = time.time()
    print(f'triangulation time: {end_trian_time - trian_time:.2f} seconds')

    cv2.imshow("stretched_img", stretched_img)
    cv2.waitKey(0)

    detect_balls_time = time.time()
    ball_detected = ball_detection(stretched_img, model)
    print("ball detection: ", ball_detected)
    end_ball_detect_time = time.time()
    print(f'ball detect time: {end_ball_detect_time - detect_balls_time:.2f} seconds')

    end_tot_time = time.time()
    print(f'total algorithem time: {end_tot_time - tot_time:.2f} seconds')
    return ball_lst_from_pixel_to_cm(ball_detected)


if __name__ == '__main__':
    main()