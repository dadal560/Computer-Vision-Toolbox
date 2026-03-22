import cv2 as cv


FRAME_100 = "Tp7/detected_frame.jpg"

def main():
    cam = 0  
    cap = cv.VideoCapture(cam)
    cap.set(cv.CAP_PROP_FPS, 25)
    count_frame = 0
    orb = cv.ORB_create()
    frame100 = cv.imread(FRAME_100)
    while True:
        ret, src = cap.read()
        count_frame += 1
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (21, 21), 0)
        keypoints, descripteurs = orb.detectAndCompute(gray, None)
        img_src = cv.drawKeypoints(src, keypoints, None, color=(0, 255, 0), flags=0)
        
        if count_frame >= 101 and count_frame <= 200:
            # frame100
            gray100 = cv.cvtColor(frame100, cv.COLOR_BGR2GRAY)
            gray100 = cv.GaussianBlur(gray100, (21, 21), 0)
            keypoints100, descripteurs100 = orb.detectAndCompute(gray100, None)

            img_out = cv.drawKeypoints(img_src, keypoints100, None, color=(255, 0, 0), flags=0)
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
    main()
