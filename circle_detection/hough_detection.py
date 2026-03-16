import cv2 as cv
import numpy as np
 
 
def main():
    cam = 0  # capture from camera at location 0
    cap = cv.VideoCapture(cam)
    cap.set(cv.CAP_PROP_FPS, 25)
    while True:
    
        ret, src = cap.read()

        
        hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)
        mask_jaune = cv.inRange(hsv, (25, 100, 100), (35, 255, 255))
    
        
        img_maskJaune = cv.GaussianBlur(mask_jaune, (5,5),0)
        
    
        rows = img_maskJaune.shape[0]
        circles = cv.HoughCircles(img_maskJaune, cv.HOUGH_GRADIENT, 1, rows / 8,
                                param1=100, param2=15,
                                minRadius=1, maxRadius=100)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                center = (i[0], i[1])
                cv.circle(src, center, 1, (0, 100, 100), 3)
                radius = i[2]
                cv.circle(src, center, radius, (255, 0, 255), 3)
        
    
        cv.imshow("mask_jaune", mask_jaune)            
        cv.imshow("detected circles", src)
        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()


    