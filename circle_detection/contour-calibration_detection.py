import sys
import cv2 as cv
import numpy as np
import math
 
 
def main():
    cam = 0  # capture from camera at location 0
    cap = cv.VideoCapture(cam)
    cap.set(cv.CAP_PROP_FPS, 25)
    data = np.load("./circle_detection/charuco_calibration.npz")

    camera_matrix = data['camera_matrix'] 
    dist_coeff = data['dist_coeff']    
    ret, src = cap.read()
    h, w = src.shape[:2]
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeff, (w,h), 1, (w,h))
    map1, map2 = cv.initUndistortRectifyMap(new_camera_matrix, dist_coeff, None, None, (w, h), cv.CV_32FC1) 

    f = (new_camera_matrix[0,0] + new_camera_matrix[1,1]) / 2

    DIAMETRE_REEL_CM = 4.0 

    print(f"Focale calculée : {f:.2f} pixels")
    while True:
        rect, src = cap.read()


        src = cv.remap(src, map1, map2, cv.INTER_LINEAR)
        
        hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        mask_jaune = cv.inRange(hsv, (25, 100, 100), (35, 255, 255))
    
        # Appliquer un flou pour réduire le bruit
        img_maskJaune = cv.GaussianBlur(mask_jaune, (5,5),0)

        canny = cv.Canny(img_maskJaune, 100, 200)
        contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        for zone in contours:
            area = cv.contourArea(zone)
            perimeter = cv.arcLength(zone, True)
            if perimeter > 0:
                q = (4 * math.pi * area) / (perimeter**2)
                if q > 0.85 :
                    distance_cm = (DIAMETRE_REEL_CM * f) / (2 * math.sqrt(area / math.pi))
                    print(f"Distance estimée : {distance_cm:.2f} cm")
                    (x, y), radius = cv.minEnclosingCircle(zone)
                    text = f"{distance_cm:.2f} cm"
                    cv.putText(src, text, (int(x - radius), int(y - radius - 10)),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv.circle(src, (int(x), int(y)), int(radius), (0, 255, 0), 3)
                    
        cv.imshow("hsv", hsv)
        cv.imshow("grey", canny)            
        cv.imshow("detected contours", src)
        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()


    