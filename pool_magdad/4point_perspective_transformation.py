import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from contuors_finder_full_code import *

TABLE_WIDTH = 255  # (cm)
TABLE_HEIGHT = 129  # (cm)
SCREEN_WIDTH = TABLE_WIDTH * 4
SCREEN_HEIGHT = TABLE_HEIGHT * 4

img1 = cv.imread('pool_magdad\dide_view.jpg')
img = cv.imread('pool_magdad\dide_view.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
rows, cols, ch = img.shape

corners = main_corners(img1)

pts1 = np.float32([corners[2][0], corners[1][0], corners[3][0], corners[0][0]])
pts2 = np.float32([[0, 0], [SCREEN_WIDTH, 0], [0, SCREEN_HEIGHT], [SCREEN_WIDTH, SCREEN_HEIGHT]])

M = cv.getPerspectiveTransform(pts1, pts2)

dst = cv.warpPerspective(img, M, (SCREEN_WIDTH, SCREEN_HEIGHT))

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()


def FAST():
    root = os.getcwd()
    # TODO : https://www.youtube.com/watch?v=zJUiZUNKhCI

