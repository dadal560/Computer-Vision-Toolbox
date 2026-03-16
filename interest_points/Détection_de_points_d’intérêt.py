import sys
import cv2 as cv
import numpy as np
import math
 
 
def main():
    cam = 0  
    cap = cv.VideoCapture(cam)
    cap.set(cv.CAP_PROP_FPS, 25)
    while True:
        ret, src = cap.read()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        # masque de taille 21*21
        gray = cv.GaussianBlur(gray, (21, 21), 0)
        orb = cv.ORB_create()
        keyPoint, des = orb.detectAndCompute(gray, None)
        src = cv.drawKeypoints(src, keyPoint, None, color=(0, 255, 0), flags=0)
        cv.imshow("detect point interet", src)
        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()


    