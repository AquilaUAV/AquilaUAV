# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 06:32:39 2016
@author: celia
"""

import numpy as np
import cv2
import glob

from numpy import iinfo

cap = cv2.VideoCapture(0)

while (True):
    _, frame = cap.read()
    frame = cv2.imread('images/dash_line.jpg')
    # Blur image to remove noise
    frame = cv2.GaussianBlur(frame, (3, 3), 0)

    # Switch image from BGR colorspace to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of purple color in HSV
    purpleMin = (0, 0, 0)
    purpleMax = (120, 120, 120)

    # Sets pixels to white if in purple range, else will be set to black
    mask = cv2.inRange(hsv, purpleMin, purpleMax)

    # dilate makes the in range areas larger
    mask = cv2.dilate(mask, None, iterations=1)

    # Set up the SimpleBlobdetector with default parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0;
    params.maxThreshold = 256;

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 30

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.5

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.0

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    reversemask = 255 - mask
    keypoints = detector.detect(reversemask)

    """
    if keypoints:
        print("found %d blobs" % len(keypoints))
        if len(keypoints) > 4:
            # if more than four blobs, keep the four largest
            keypoints.sort(key=(lambda s: s.size))
            keypoints = keypoints[0:3]
    else:
        print("no blobs")
    """

    print('\n'*10)
    for i in keypoints:
        print(i.response, i.angle, i.class_id, i.octave, i.pt, i.size, i.hash())
    # Draw green circles around detected blobs
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 255, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # open windows with original image, mask, res, and image with keypoints marked
    cv2.imshow('mask', mask)
    cv2.imshow('im_with_keypoints', im_with_keypoints)

    k = cv2.waitKey(1)
    if k & 0xFF is ord('q'):
        break
cv2.destroyAllWindows()