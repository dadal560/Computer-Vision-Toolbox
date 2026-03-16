import sys
import cv2 as cv
import numpy as np
import math
 
 
def main():
    cam = 0  # capture from camera at location 0
    cap = cv.VideoCapture(cam)
    cap.set(cv.CAP_PROP_FPS, 25)
    while True:
    
        ret, src = cap.read()

        src = cv.GaussianBlur(src, (5,5),0)
        canny = cv.Canny(src, 100, 200)
        contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        for zone in contours:
            area = cv.contourArea(zone)
            perimeter = cv.arcLength(zone, True)
            if perimeter > 0:
                q = (4 * math.pi * area) / (perimeter**2)
                if q > 0.85 :
                    (x, y), radius = cv.minEnclosingCircle(zone)
                    cv.circle(src, (int(x), int(y)), int(radius), (0, 255, 0), 3)

        cv.imshow("grey", canny)            
        cv.imshow("detected contours", src)
        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()


    