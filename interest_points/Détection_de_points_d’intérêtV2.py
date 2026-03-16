import sys
import cv2 as cv
import numpy as np
import math

FRAME_100 = "/home/henry/Bureau/L3/S6/Vision embarquée et Intelligence artificielle/Tp7/points_interet.jpg"

def main():
    cam = 0  
    cap = cv.VideoCapture(cam)
    cap.set(cv.CAP_PROP_FPS, 25)
    count_frame = 0
    orb = cv.ORB_create(10000)
    while True:
        ret, src = cap.read()
        count_frame += 1
        if count_frame == 50:
            cv.imwrite("Tp7/points_interet.jpg", src)
            print ("Frame 50 saved as points_interet.jpg")
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (21, 21), 0)
        keyPoint, des = orb.detectAndCompute(gray, None)
        img_src = cv.drawKeypoints(src, keyPoint, None, color=(0, 255, 0), flags=0)
        if count_frame >= 101 and count_frame <= 200:
            # frame100
            frame100 = cv.imread(FRAME_100)
            gray100 = cv.cvtColor(frame100, cv.COLOR_BGR2GRAY)
            gray100 = cv.GaussianBlur(gray100, (21, 21), 0)
            keyPoint100, des = orb.detectAndCompute(gray100, None)

            img_out = cv.drawKeypoints(img_src, keyPoint100, None, color=(255, 0, 0), flags=0)
            cv.imshow("detect point interet", img_out)
        elif count_frame > 200:
            cv.destroyWindow("detect point interet")
            cv.imshow("detect point interet 200", img_src)
        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()
 
 
if __name__ == "__main__":
    print(FRAME_100)
    main()


    # Bruno Lescalier systel de la par de menard
    # vision singal analyse de données 
    