from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np


def blur_image(img):
    image = cv2.blur(img, (40,40))
    # cv2.imshow("Image", image) 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image


def remove_no_green_pixels(img):
    mask = (img[:, :, 1] > 150) & (img[:, :, 0] > 60) & (img[:, :, 2] > 80) & (img[:, :, 2] < 150)
    out = np.zeros(img.shape)
    out[mask] = [255, 255, 255]

    return out  

def cornerDetector(img):

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

im = cv2.imread(r"C:\Users\TLP\OneDrive - huji.ac.il\Desktop\magdad_pool\Snooker47\pool_magdad\play1.jpg")
bi_im = remove_no_green_pixels(blur_image(im))
imgGray = cv2.cvtColor(bi_im, cv2.COLOR_RGB2GRAY)
corners = cornerDetector(imgGray)
print(corners)

normal_view = cv2.resize(bi_im, (480,640) )

cv2.imshow("Image", normal_view) 
cv2.waitKey(0)
cv2.destroyAllWindows()
