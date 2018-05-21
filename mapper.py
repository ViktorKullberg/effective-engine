#Run this to show the picture you want to map
#Clicking on the image will display the pixel-coordinates
#Use these coordinates in homography to create the h-matrix and map them

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
import serial
import Serial

xCoordinate = 0
yCoordinate = 0

ser = Serial.Serial('COM9',115200)
#ser.parity = serial.PARITY_NONE #set parity check: no parity
#ser.stopbits = serial.STOPBITS_ONE #number of stop bits

#Stores the coordiantes you clicked
def find_pixels(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        # cv2.circle(img,(x,y),100,(255,0,0),-1)
        print("X:", x, "Y:", y)
        global xCoordinate
        global yCoordinate
        xCoordinate = x
        yCoordinate = y


cv2.namedWindow('image')
cv2.setMouseCallback('image', find_pixels)
im_src = cv2.imread('imageholo.jpeg')
print(bytes([43]))
print(bytes(str(43).encode()))
cap = cv2.VideoCapture("http://root:pass@192.168.1.3/mjpg/video.mjpg")
cap.open("http://root:pass@192.168.1.3/mjpg/video.mjpg")

while(True):

    cv2.imshow('image', im_src)
    #test = ser.write(b'hello')
    #ser.sendData(xCoordinate, yCoordinate)
    # ser.write(bytes(b'V'))
    # print(xCoordinate)
    # print(yCoordinate)
    # testY = yCoordinate//4
    # testX = xCoordinate//4
    #print(testX)
    #print(testY)
    # ser.write(bytes([(xCoordinate//4)]))
    # ser.write(bytes([(yCoordinate//4)]))
    #ser.write(bytes([testX]))
    #ser.write(bytes([testY]))
    #if xCoordinate < 100:
        #ser.write(str(0).encode())
        #if(xCoordinate < 10):
            #ser.write(str(0).encode())
    #ser.write(str(xCoordinate).encode())
    #if yCoordinate < 100:
        #ser.write(str(0).encode())
        #if(yCoordinate < 10):
           # ser.write(str(0).encode())
    #ser.write(str(yCoordinate).encode())
    time.sleep(0.05)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break