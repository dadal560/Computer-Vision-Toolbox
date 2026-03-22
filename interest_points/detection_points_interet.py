import cv2 as cv

 
def main():
    cam = 0  
    cap = cv.VideoCapture(cam)
    cap.set(cv.CAP_PROP_FPS, 25)
    orb = cv.ORB_create(nfeatures=500, scoreType=cv.ORB_FAST_SCORE)
    while True:
        ret, src = cap.read()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        # masque de taille 21*21
        gray = cv.GaussianBlur(gray, (21, 21), 0)
        keypoints, descripteurs = orb.detectAndCompute(gray, None)
        src = cv.drawKeypoints(src, keypoints, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
        cv.imshow("detect point interet", src)
        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
