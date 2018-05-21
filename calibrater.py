
##USE THIS TO REMOVE FISHEYE-EFFECT ON CAMERA##
#K, D and DIM is calculated from calibrate.py using 13 chessboard images following
#guides online https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0#



import numpy as np
import cv2

assert float(cv2.__version__.rsplit('.', 1)[0]) >= 3

K = np.array([[575.9776309200677, 0.0, 407.82765771054426], [0.0, 576.6608930341262, 299.73790582663247],
              [0.0, 0.0, 1.0]])
D = np.array([[-0.0030660809375308356], [-0.5970552167328254], [2.1997056097503904], [-2.6551971161522494]])
DIM = (800, 600)

#call this one time to save the parameters map1 and map2.

def calibrateFisheye():
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    return [map1, map2]

#undistort your image with parameters received from calibrateFisheye()
def remapper(image, map1, map2):
    mappedimage = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return mappedimage
