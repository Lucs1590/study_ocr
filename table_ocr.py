import cv2
import numpy as np

from PIL import Image
from time import time


""" def detect_edges(image):
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
        cv2.waitKey(0) """


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


# read image
image = cv2.imread('images/t1.png')
# removing alpha chanel
image = image[:, :, :3]
# increase image 4 times
image = image_resize(image, height=image.shape[0]*4)
# erode
image = erode(image, 3)
# dilate
image = dilate(image, 5)
cv2.imwrite('output1.png', image)
cv2.waitKey(0)
