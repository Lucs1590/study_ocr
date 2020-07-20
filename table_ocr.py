import cv2
import numpy as np

from PIL import Image
from time import time


def detect_edges(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    cv2.imwrite('images/canny1.jpg', edges)
    minLineLength = 100
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi/180, threshold=100,
                            lines=np.array([]), minLineLength=minLineLength, maxLineGap=80)

    a, b, c = lines.shape
    for i in range(a):
        cv2.line(image, (lines[i][0][0], lines[i][0][1]), (lines[i]
                                                           [0][2], lines[i][0][3]), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite('houghlines.jpg', image)
        cv2.imshow('img', image)
        cv2.waitKey(0)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def erode(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel, iterations=3)


def dilate(img, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)


def remove_lines(image):
    # Color changes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh_val, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Making kernels
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    # Detect lines
    detected_h_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    detected_v_lines = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Remove lines
    h_cnts = cv2.findContours(
        detected_h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h_cnts = h_cnts[0] if len(h_cnts) == 2 else h_cnts[1]
    for c in h_cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

    v_cnts = cv2.findContours(
        detected_v_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    v_cnts = v_cnts[0] if len(v_cnts) == 2 else v_cnts[1]
    for c in v_cnts:
        cv2.drawContours(image, [c], -1, (255, 255, 255), 2)

    # Repair image
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    result = 255 - cv2.morphologyEx(255 - image,
                                    cv2.MORPH_CLOSE, repair_kernel, iterations=1)

    cv2.imwrite('tests/thresh1.png', thresh)
    # cv2.imwrite('tests/detected_lines0.png', detected_h_lines)
    # cv2.imwrite('tests/detected_lines1.png', detected_v_lines)
    cv2.imwrite('tests/image.png', image)
    # cv2.imwrite('tests/result1.png', result)
    cv2.waitKey()


# read image
image = cv2.imread('images/tabela.png')
# removing alpha chanel
image = image[:, :, :3]
# remove lines
remove_lines(image)
# increase image 4 times
image = image_resize(image, height=image.shape[0]*4)

cv2.imwrite('tests/output1.png', image)
cv2.waitKey(0)
