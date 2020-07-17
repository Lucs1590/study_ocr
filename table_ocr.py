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
        

image = cv2.imread('images/tabela.png')
detect_edges(image)


cv2.imshow('output', image)
cv2.waitKey(0)
