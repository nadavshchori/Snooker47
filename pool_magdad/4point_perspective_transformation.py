import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

TABLE_WIDTH = 255  # (cm)
TABLE_HEIGHT = 129  # (cm)
SCREEN_WIDTH = TABLE_WIDTH * 4
SCREEN_HEIGHT = TABLE_HEIGHT * 4

img = cv.imread('pool.jpg')
assert img is not None, "file could not be read, check with os.path.exists()"
rows, cols, ch = img.shape

pts1 = np.float32([[18, 171], [1217, 175], [36, 761], [1212, 738]])
pts2 = np.float32([[0, 0], [SCREEN_WIDTH, 0], [0, SCREEN_HEIGHT], [SCREEN_WIDTH, SCREEN_HEIGHT]])

M = cv.getPerspectiveTransform(pts1, pts2)

dst = cv.warpPerspective(img, M, (SCREEN_WIDTH, SCREEN_HEIGHT))
plt.imshow(dst)

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()


def FAST():
    root = os.getcwd()
    # TODO : https://www.youtube.com/watch?v=zJUiZUNKhCI

