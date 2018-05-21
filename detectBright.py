from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

imageToRead = "bluetest.jpg"
image = cv2.imread(imageToRead)
image = cv2.resize(image, (800, 600))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (15, 15), 0)

thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]

thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)

cv2.imshow("Image", thresh)
cv2.waitKey(0)





