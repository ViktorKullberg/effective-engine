from collections import deque
from matplotlib import pyplot as plt
import cv2
assert float(cv2.__version__.rsplit('.', 1)[0]) >= 3
import numpy as np
import sys
import imutils
import os
import glob
import Serial
import time

if __name__ == '__main__':


    def find_homograpgy(p_src, p_dst):
        print(len(p_src))
        if len(p_src) != 4 | len(p_dst) != 4:
            return 0
        hmatrix = cv2.findHomography(p_src, pts_dst)
        return hmatrix
    #find the correct pos of y

    #correctst the height of the led
    def heighCorrection2(x, y, ledheight, cameraheight, xStart, ystart):
        t = (cameraheight - ledheight)/cameraheight
        vx = x - 200
        vy = 400 - y - 30 #the camera is perceived to be closer than what it is hence -37
        xreal = xStart + t*vx
        yreal = ystart + t*vy
        return [xreal, yreal]

    pts_dst = np.array([
        np.array([[0.0], [0.0]]),
        np.array([[400], [0.0]]),
        np.array([[400], [400]]),
        np.array([[0.0], [400]])])

    #p1 = (217, 28)
    #p2 = (592, 31)
    #p3 = (786, 307)
    #p4 = (2, 310)

    p1 = (208, 41)
    p2 = (592, 40)
    p3 = (783, 348)
    p4 = (31, 349)

    pts_corners = np.array([
        np.array([p1]),
        np.array([p2]),
        np.array([p3]),
        np.array([p4])])

    #p1 = (227, 117)
    #p2 = (596, 116)
    #p3 = (785, 413)
    #p4 = (22, 413)

    a = np.array([[p1[0]], [p1[1]], [1.0]])
    b = np.array([[p2[0]], [p2[1]], [1.0]])
    c = np.array([[p3[0]], [p3[1]], [1.0]])
    d = np.array([[p4[0]], [p4[1]], [1.0]])



    b = find_homograpgy(pts_corners, pts_dst)
    #print(b[0])
   # image_corners = np.array([A[:-1], B[:-1], C[:-1], D[:-1]])
    #print(A[:-1])
    #print(image_corners)
    #image_corners = np.array([a,b,c,d])
    # Calculate homography
    #h, status = cv2.findHomography(image_corners, pts_dst)
    h, status = cv2.findHomography(pts_corners, pts_dst)
    print(h)
    xPixel = 399
    yPixel = 144
    # cv2.warpPerspective(im_src, im_src, h, 800,800);
    #test = np.matmul(h, flip_y_origo)
    # dst = cv2.warpPerspective(im_src, test, (400, 400))
    # Warp source image to destination based on homography
    # im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
    point = np.array([[p1[0]], [p1[1]], [1]])
    point_center = np.array([[xPixel], [yPixel], [1]])
    #position = np.matmul(test, point_center)
    position = np.matmul(h, point_center)
    # remove scaling issues
    x = position[0] / position[2]
    y = position[1] / position[2]
    print("x:", x, "y:", y)
    print(np.matmul(h, point_center))  # Display images
    realCoordinates = heighCorrection2(x, y, 42, 230, 200, 0)
    print(realCoordinates[0], realCoordinates[1])