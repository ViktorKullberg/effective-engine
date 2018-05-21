from collections import deque
from matplotlib import pyplot as plt
import cv2
assert float(cv2.__version__.rsplit('.', 1)[0]) >= 3
import numpy as np
import ROI
import sys
import imutils
import os
import glob
import Serial
import time

#ser = Serial.Serial('COM9', 115200)
xCoordinate = 100
yCoordinate = 100
xCoordinate1 = 0
yCoordinate2 = 0
xtest = 555
ytest = 777
ready = False

global detect
def find_pixels(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP:
        #cv2.circle(img,(x,y),100,(255,0,0),-1)
        print("X:" ,x, "Y:",y)
        global xCoordinate1
        global yCoordinate2
        xCoordinate1 = x
        yCoordinate2 = y
# Create a black image, a window and bind the function to window

cv2.namedWindow('image')
cv2.setMouseCallback('image',find_pixels)
#ser = serial.Serial('COM9',115200)
#ser.parity = serial.PARITY_NONE #set parity check: no parity
#ser.stopbits = serial.STOPBITS_ONE #number of stop bits


lower_red = np.array([0,100,100], np.uint8)
upper_red = np.array([10, 255, 255],np.uint8)

lower_red2 = np.array([160,100,100], np.uint8)
upper_red2 = np.array([179, 255, 255], np.uint8)

lower_blue = np.array([110,180,180], np.uint8)
upper_blue = np.array([160, 255, 255], np.uint8)


lower_yellow = np.array([10, 40, 50], np.uint8)
upper_yellow = np.array([20, 255, 255], np.uint8)
K = np.array([[575.9776309200677, 0.0, 407.82765771054426], [0.0, 576.6608930341262, 299.73790582663247],
                  [0.0, 0.0, 1.0]])
D = np.array([[-0.0030660809375308356], [-0.5970552167328254], [2.1997056097503904], [-2.6551971161522494]])
DIM = (800, 600)

balance = 1.0


cap = cv2.VideoCapture("http://root:pass@192.168.1.10/mjpg/video.mjpg")
cap.open("http://root:pass@192.168.1.10/mjpg/video.mjpg")
#cap = cv2.VideoCapture(0)
#cap.open(0)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
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

p1 = (208, 42)
p2 = (591, 41)
p3 = (778, 345)
p4 = (28, 345)

a = np.array([[p1[0]], [p1[1]], [1.0]])
b = np.array([[p2[0]], [p2[1]], [1.0]])
c = np.array([[p3[0]], [p3[1]], [1.0]])
d = np.array([[p4[0]], [p4[1]], [1.0]])

flip_y_origo = np.array([[1, 0, -p1[0]],
                         [0, -1, p1[1]],
                         [0, 0, 1]])

A = np.matmul(flip_y_origo, a)
B = np.matmul(flip_y_origo, b)
C = np.matmul(flip_y_origo, c)
D = np.matmul(flip_y_origo, d)

image_corners = np.array([A[:-1], B[:-1], C[:-1], D[:-1]])
# Calculate homography
h, status = cv2.findHomography(image_corners, pts_dst)
#test = np.matmul(h, flip_y_origo)
test = h
point11 = 185;
point22 = 13;
RoiDimension = 50
global point1
##point1 = (185, 13)
##point2 = (407, 224)
while(cap.isOpened()):
    #timer1 = cv2.getTickCount()
    ret, img = cap.read()
    point1 = (xCoordinate1, yCoordinate2)
    #img = cv2.imread('imageholo.jpeg')
    #img = ROI.getROI(img, point1, RoiDimension)
    #ret = True
    if ret == True:
        #point11+=1
        #point22+=1
        #point33+=1
        #point44+=1
        img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        #img = imutils.resize(img, width=1000, height = 2000)
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #kernel = np.ones((2, 2), np.uint8)  # Sets the size of the kernel used for erode and dilate

        #img_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
        #img_blue = cv2.erode(img_blue, kernel, iterations=2)
        #img_blue = cv2.dilate(img_blue, kernel, iterations=3)

        #img_blur = cv2.GaussianBlur(hsv_image, (15, 15), 0)
        hsv_image2 = cv2.inRange(hsv_image, lower_blue, upper_blue);
        #opening = cv2.morphologyEx(hsv_image2 , cv2.MORPH_OPEN, kernel)
        x, y, w, h, = cv2.boundingRect(hsv_image2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        #print(x+(w/2), y+(h/2))
        xPixel = x+(w/2)
        yPixel = y+(h/2)
        point_center = np.array([[xPixel], [yPixel], [1]])
        position = np.matmul(test, point_center)
        #print(position)
        # remove scaling issues
        xCoordinate = position[0] / position[2]
        yCoordinate = position[1] / position[2]
        #cv2.circle(img, (x + w / 2, y + h / 2), 5, (0, 0, 255), -1)

        #im = cv2.imread("imageholo.jpeg")
        # Select ROI
        #r = cv2.selectROI(im)
        # Crop image
        #imgThreshold =  cv2.inRange(imgHSV, cvScalar(0, 0, 0, 0), cvScalar(180, 255, 30, 0));
        #mask = cv2.inRange(hsv_image, lower_red, upper_red)
        #lower_red_hue = cv2.bitwise_and(hsv_image, hsv_image, mask=mask)
        #mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        #upper_red_hue = cv2.bitwise_and(img, img, mask=mask2)

        #mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        #full_image = cv2.bitwise_and(img, img, mask = mask)

        #full_image = cv2.addWeighted(lower_red_hue, 1.0, upper_red_hue,1.0,0.0)

        #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        #img_gray = cv2.GaussianBlur(img_gray, (11,11),0)
       # img_gray = cv2.blur(img_gray, (11,11),0)
        #cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY,img_gray)
        #cv2.erode(img_gray, kernel, iterations=1)
        #cv2.dilate(img_gray, kernel, iterations=1)
        #cv2.bitwise_not(img_gray, img_gray);
        #mask0 = cv2.inRange(img_hsv, lower_red, upper_red)
        #mask1 = cv2.inRange(img_hsv, lower_red2, upper_red2)
        #mask = mask0+mask1
       # img_gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        #circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT,1.2,100)
        #circles = np.round(circles[0, :]).astype("int")
        ##for(center_x, center_y, radius) in circles:
        #    cv2.circle(img, (center_x, center_y), radius , (0,255,0))
       # cv2.imshow('image', full_image)

        #im2, contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.drawContours(full_img, contours, -1, (0, 255, 0), 3)

        #dst = cv2.resize(full_img, None, fx = 1.3, fy = 1.3, interpolation=cv2.INTER_CUBIC)

        cv2.imshow('image', img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()