from ultralytics import YOLO
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import ImageEnhance


def remove_no_green_pixels(img):
    rows, cols = img.shape[:2]
    for i in range(rows):
        for j in range(cols):
            if int(img[i, j][1]) > 150 and  60 < int(img[i, j][0]) < 120 and  80 < int(img[i, j][2]) < 150:
                img[i, j] = (255, 255, 255)
            else:
                img[i, j] = 0

    cv2.imshow("table_crop", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

def enhence(img):
    img1 = cv2.addWeighted(
    img,
    1,
    np.zeros(img.shape, img.dtype),
    15,
    1
    )
    
    kernel = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])

    sharpened_image = cv2.filter2D(img, -1, kernel)
    return sharpened_image

def YOLO_crop(img):
    model = YOLO("yoloe-26n-seg.pt")
    model.set_classes(["pool table from a 45 degree downward view"])    
    results = model(img)
    result = results[0]
    result.show()
    corners = result.boxes[0].xyxy[0].int().tolist()
    x1 = corners[0]
    y1 = corners[1]
    x2 = corners[2]
    y2 = corners[3]
    table_crop = img[y1:y2, x1:x2]     
    cv.imshow("table_crop", table_crop)
    cv.waitKey(0)
    cv.destroyAllWindows()

def harris_corner_detection(image):

    operatedImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    operatedImage = np.float32(operatedImage)

    dest = cv.cornerHarris(operatedImage, 17, 21, 5)
    dest = cv.dilate(dest, None)

    image[dest > 0.01 * dest.max()] = [0, 0, 255]
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    plt.imshow(image_rgb)
    plt.title('Harris Corner Detection')
    plt.axis('off')  
    plt.show()

def camscanner_corners(img):
    # Resize image to workable size
    dim_limit = 1080
    max_dim = max(img.shape)
    if max_dim > dim_limit:
        resize_scale = dim_limit / max_dim
        img = cv2.resize(img, None, fx=resize_scale, fy=resize_scale)
    # Create a copy of resized original image for later use
    orig_img = img.copy()
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=3)
    # GrabCut
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (20, 20, img.shape[1] - 20, img.shape[0] - 20)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]
 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection.
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
 
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
 
    # Detecting Edges through Contour approximation.
    # Loop over the contours.
    if len(page) == 0:
        return orig_img
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points.
        if len(corners) == 4:
            break
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
    # For 4 corner points being detected.
    corners = order_points(corners)
 
    destination_corners = find_dest(corners)
 
    h, w = orig_img.shape[:2]
    # Getting the homography.
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # Perspective transform using homography.
    final = cv2.warpPerspective(orig_img, M, (destination_corners[2][0], destination_corners[2][1]),
                                flags=cv2.INTER_LINEAR)
    return final

def find_dest(pts):
    (tl, tr, br, bl) = pts
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
 
    return order_points(destination_corners)

def find_contuors(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 300)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imshow('Canny Edges After Contouring', edged)
    cv2.waitKey(0)  
    return contours

def cornerDetector(contours):
    """ A function used to highlight all the contours in the image and 
        find the corners of the biggest one.
        
        Parameters: img (matrix): matrix image
        Returns: corners(Array): return the biggest contours corner pixel points
    """
    # biggest contours corner points saved in an array
    corners = np.array([])
    # initalize the max area
    maxArea = 0
    
    # loop through each contour in the image
    for cnt in contours:
        # svae its area size
        area = cv2.contourArea(cnt)
        # if area smaller than 6k ignore it 
        if area > 6000:
            # if we want to see all contours uncomment this
            # cv2.drawContours(imgCopy, cnt, -1, (0, 0, 255), 2)
            # Calculate the contour curves length we need it for approximation points (the bigger the accurate)
            epsilon  = 0.05*cv2.arcLength(cnt,True)
            # An approximation of the edges points ( corner points)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            # if this contour has bigger area and it has 4 points edges ( meaning it is a square such as paper)
            if area > maxArea and len(approx) == 4:
                corners = approx
                maxArea = area
    return corners

def detect_brown_corners(img):
    rows, cols = img.shape[:2]
    for i in range(rows):
        for j in range(cols):
            if   22 < int(img[i, j][0]) < 173    and    70 < int(img[i, j][1]) < 199     and      123 < int(img[i, j][2]) < 255:
                img[i, j] = (255, 255, 255)
            else:
                img[i, j] = 0

    cv2.imshow("table_crop", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

def brown_scale():
    img = cv2.imread("pool_magdad\many_brown.jpg")
    rows, cols = img.shape[:2]
    max = [0,0,0]
    min = [10000,10000,10000]
    for i in range(rows):
        for j in range(cols):
            red , green, blue = img[i,j][2] , img[i,j][1] , img[i,j][0]
            if red < min[2]:
                min[2] = red
            if red > max[2]:
                max[2] = red 

            if blue < min[0]:
                min[0] = blue
            if blue > max[0]:
                max[0] = blue 

            if green < min[1]:
                min[1] = green
            if green > max[1]:
                max[1] = green 

            print(img[i,j])
    print(max,min)

def transformation(image):
  image=image.copy()  
  height, width, channels = image.shape
  gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  image_size=gray.size
  
  threshold=blur_and_threshold(gray)
  # We need two threshold values, minVal and maxVal. Any edges with intensity gradient more than maxVal 
  # are sure to be edges and those below minVal are sure to be non-edges, so discarded. 
  #  Those who lie between these two thresholds are classified edges or non-edges based on their connectivity.
  # If they are connected to "sure-edge" pixels, they are considered to be part of edges. 
  #  Otherwise, they are also discarded
  edges = cv2.Canny(threshold,50,150,apertureSize = 7)
  contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  simplified_contours = []


  for cnt in contours:
      hull = cv2.convexHull(cnt)
      simplified_contours.append(cv2.approxPolyDP(hull,
                                0.001*cv2.arcLength(hull,True),True))
  simplified_contours = np.array(simplified_contours)
  biggest_n,approx_contour = biggest_contour(simplified_contours,image_size)

  threshold = cv2.drawContours(image, simplified_contours ,biggest_n, (0,255,0), 1)

  dst = 0
  if approx_contour is not None and len(approx_contour)==4:
      approx_contour=np.float32(approx_contour)
      dst=four_point_transform(threshold,approx_contour)
  croppedImage = dst
  return croppedImage

def biggest_contour(contours,min_area):
    biggest = None
    max_area = 0
    biggest_n=0
    approx_contour=None
    for n,i in enumerate(contours):
            area = cv2.contourArea(i)
         
            
            if area > min_area/10:
                    peri = cv2.arcLength(i,True)
                    approx = cv2.approxPolyDP(i,0.02*peri,True)
                    if area > max_area and len(approx)==4:
                            biggest = approx
                            max_area = area
                            biggest_n=n
                            approx_contour=approx
                            
                                                   
    return biggest_n,approx_contour

def binary_kernel(img):
    k_size = 15
    rows, cols = img.shape[:2]
    for i in range(0,rows,k_size):
        for j in range(0,cols,k_size):
            sum_w = 0 
            change = 255
            for r in range(0,k_size):
                for c in range(0,k_size):
                    if i+r < rows-1 and j+c < cols-1:
                        sum_w += int(img[i + r , j + c][0])
            if sum_w == 0:
                change = 0
            for r in range(0,k_size):
                for c in range(0,k_size):
                    if i+r < rows-1 and j+c < cols-1:
                        img[i + r , j + c] = (change,change,change)
    cv2.imshow("table_crop", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img 


            
    

img = cv2.imread("pool_magdad/side_view.jpg")
new_img = remove_no_green_pixels(img)
image = binary_kernel(new_img)


# image = detect_brown_corners(img)
# contours = find_contuors(new_img)

# corners = cornerDetector(contours)
# print(corners)
# brown_scale()