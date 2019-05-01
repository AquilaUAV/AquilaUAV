from math import *
import numpy as np
import cv2 as cv

def getCorrectListRect(rect):
    new_rect = []
    new_rect.append(list(rect[0]))
    new_rect.append(list(rect[1]))
    new_rect.append(rect[2])
    if (new_rect[1][1] > new_rect[1][0]):
        new_rect[1][1], new_rect[1][0] = new_rect[1][0], new_rect[1][1]
        new_rect[2] = (new_rect[2] + 90) % 360
    return new_rect

def getCorrectTupleRect(rect):
    new_rect = []
    new_rect.append(tuple(rect[0]))
    new_rect.append(tuple(rect[1]))
    new_rect.append(rect[2])
    return tuple(new_rect)

def drawRect(rect):
    rect = getCorrectTupleRect(rect)
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(im, [box], 0, (0, 0, 255), 2)

def recognizeRect(contours, min_area, max_area, min_aspect_ratio, max_aspect_ratio):
    rectMap = []
    a = np.array([[max_area**2, (0.5*(max_area+min_area))**2, min_area**2],
                  [max_area, (0.5 * (max_area + min_area)), min_area],
                   [1, 1, 1]]).transpose()
    b = np.array([0, 1, 0])
    result = np.linalg.solve(a, b)
    print(result)

    for cnt in contours:
        try:
            rect = cv.minAreaRect(cnt)
            print(rect)
            rect = getCorrectListRect(rect)
            center_point = rect[0].copy()
            vector = rect[1].copy()
            angle = rect[2]

            vector[0], vector[1] = vector[0] * cos(angle * pi / 180) + vector[1] * cos((angle + 90) * pi / 180), vector[
                0] * sin(angle * pi / 180) + vector[1] * sin((angle + 90) * pi / 180)
            vector[0] /= 2
            vector[1] /= 2
            vector[0] += center_point[0]
            vector[1] += center_point[1]

            center_point = tuple(np.int0(center_point))
            vector = tuple(np.int0(vector))
            print(center_point)
            print(vector)
            cv.line(im, center_point, vector, (0, 255, 0), 2)
            drawRect(rect)
        except:
            pass

cap = cv.VideoCapture(0)
while (True):
    _, im = cap.read()
    #im = cv.imread('images/dash_line.jpg')
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(imgray, 200, 255, 0)
    #thresh = cv.adaptiveThreshold(imgray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    im2,contours,hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    print('\n'*10)
    rectMap = recognizeRect(contours, 20, 100, 3, 9)
    cv.imshow('im', im)
    cv.imshow('im2', im2)
    k = cv.waitKey(1)
    if (k != -1):
        break