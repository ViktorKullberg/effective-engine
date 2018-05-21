import cv2
import numpy as np




def getROI(img, p1,dimension):
    black = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    black1 = cv2.rectangle(black, (p1[0] - dimension, p1[1] - dimension), (p1[0] + dimension, p1[1] + dimension), (255, 255, 255), -1)  # ---the dimension of the ROI
    gray = cv2.cvtColor(black1, cv2.COLOR_BGR2GRAY)  # ---converting to gray
    #ret, b_mask = cv2.threshold(gray, 127, 255, 0)  # ---converting to binary image
    image = cv2.bitwise_and(img, img, mask=gray)
    return image