from collections import deque
from matplotlib import pyplot as plt
import cv2
assert float(cv2.__version__.rsplit('.', 1)[0]) >= 3
import numpy as np
import calibrater
import sys
import imutils
import os
import glob
import Serial
import time
import ROI

ser = Serial.Serial('COM9', 115200)

def find_pixels(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP:
        print("X:" ,x, "Y:",y)
        global xCoordinate
        global yCoordinate
        first = 1
        xCoordinate = x
        yCoordinate = y

def heighCorrection2(x, y, ledheight, cameraheight, xStart, ystart):
    t = (cameraheight - ledheight)/cameraheight
    vx = x - 200
    vy = 400 - y - 43
    xreal = xStart + t*vx
    yreal = ystart + t*vy
    return [xreal, yreal]
cv2.namedWindow('image')
cv2.setMouseCallback('image', find_pixels)

lower_red = np.array([0,100,100], np.uint8)
upper_red = np.array([10, 255, 255], np.uint8)

lower_red2 = np.array([160,100,100], np.uint8)
upper_red2 = np.array([179, 255, 255], np.uint8)

lower_blue = np.array([110,125, 110], np.uint8)
upper_blue = np.array([145, 255, 255], np.uint8)

#lower_blue = np.array([110,80,100])
#upper_blue = np.array([130,255,255])

lower_yellow = np.array([10, 40, 50], np.uint8)
upper_yellow = np.array([20, 255, 255], np.uint8)


K = np.array([[575.9776309200677, 0.0, 407.82765771054426], [0.0, 576.6608930341262, 299.73790582663247],
                  [0.0, 0.0, 1.0]])
D = np.array([[-0.0030660809375308356], [-0.5970552167328254], [2.1997056097503904], [-2.6551971161522494]])
DIM = (800, 600)

balance = 1.0
kernel = np.ones((2,2), np.uint8)

cap = cv2.VideoCapture("http://root:pass@192.168.1.10/mjpg/video.mjpg")
cap.open("http://root:pass@192.168.1.10/mjpg/video.mjpg")
#cap.open("http://admin:Mk123456789@192.168.1.5/video.cgi?.mjpg")

#map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
map = calibrater.calibrateFisheye()
#if(cap.isOpened()== False):
  # print("error loading stream")

pts_dst = np.array([
    np.array([[0.0], [0.0]]),
    np.array([[400], [0.0]]),
    np.array([[400], [400]]),
    np.array([[0.0], [400]])])

    #p1 = (216, 107)
    #p2 = (584, 109)
    #p3 = (774, 398)
    #p4 = (11, 400)

p1 = (213, 50)
p2 = (592, 48)
p3 = (778, 349)
p4 = (33, 348)

#hörner vid upphöjning ändra till 390 i bredd
# p1 = (212, 23)
# p2 = (582, 24)
# p3 = (768, 289)
# p4 = (1, 295)

pts_corners = np.array([
        np.array([p1]),
        np.array([p2]),
        np.array([p3]),
        np.array([p4])])

# Calculate homography
h, status = cv2.findHomography(pts_corners, pts_dst)
#test = np.matmul(h, flip_y_origo)
RoiDimension = 50

while(cap.isOpened()):
    #timer1 = cv2.getTickCount()
    ret, img = cap.read()
    #img = ROI.getROI(img, (400, 149), RoiDimension)
    #img = cv2.imread('image.jpg')
    #ret = True
    if ret == True:
        #img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        img = calibrater.remapper(img, map[0], map[1])
        #img_blur = cv2.GaussianBlur(img, (1, 1), 0)
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_image = cv2.GaussianBlur(hsv_image, (13, 13), 0)
        img_blur = cv2.inRange(hsv_image, lower_blue, upper_blue)
        img_blur = cv2.erode(img_blur, kernel, iterations=2)
        img_blur = cv2.dilate(img_blur, kernel, iterations=3)
        #img_blur = cv2.GaussianBlur(hsv_image, (15, 15), 0)
        #opening = cv2.morphologyEx(hsv_image2 , cv2.MORPH_OPEN, kernel)
        x, y, w, height = cv2.boundingRect(img_blur)
        cv2.rectangle(img, (x, y), (x + w, y + height), (0, 255, 0), 1)
        #print(x+(w/2), y)
        if (w < 1) | (w > 35) | (height > 35):
            print("bad result", w, height)
            xCoordinate = 255*2
            yCoordinate = 255*2
            realCoordinates = (xCoordinate, yCoordinate)
        else:
            print("blue found!",w,height)
            xPixel = x + (w/2)
            yPixel = y + (height/2)
            point_center = np.array([[xPixel], [yPixel], [1]])
            position = np.matmul(h, point_center)
            # remove scaling issues
            xCoordinate = position[0] / position[2]
            yCoordinate = position[1] / position[2]
            print(xCoordinate)
            print(yCoordinate)
            realCoordinates = heighCorrection2(xCoordinate, yCoordinate, 41, 228, 200, 0)
            print("X:", realCoordinates[0], "\n", "Y:",realCoordinates[1])

        cv2.imshow('image2', img_blur)
        cv2.imshow('image', img)
        #realCoordinates = heighCorrection2(xCoordinate, yCoordinate, 42, 228, 200, 0)
        #print(realCoordinates)
        ser.sendData(int(realCoordinates[0]), int(realCoordinates[1]))
        #ser.sendData(int(xCoordinate), int(yCoordinate))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
