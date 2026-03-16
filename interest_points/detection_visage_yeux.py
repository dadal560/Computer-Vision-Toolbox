import cv2 as cv


FRAME_100 = "/home/henry/Bureau/L3/S6/Vision embarquée et Intelligence artificielle/Tp7/cascade.jpg"

def frame_cascade(src, face_cascade, eyes_cascade, gray, colorface=(255, 0, 0), color=(0, 255, 0)):
    frame_gray = cv.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    for (x,y,w,h) in faces:
        frame = cv.rectangle(src, (x,y), (x+w, y+h), colorface, 2)
        faceROI = frame_gray[y:y+h,x:x+w]
        eyeRoi = faceROI[0:int(h/2), :]
        eyes = eyes_cascade.detectMultiScale(eyeRoi)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, color, 2)



def main():
    cam = 0 
    cap = cv.VideoCapture(cam)
    cap.set(cv.CAP_PROP_FPS, 25)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyes_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
    count_frame = 0
    while True:
        rect, src = cap.read()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        count_frame += 1
        if count_frame == 50:
            cv.imwrite("Tp7/cascade.jpg", src)
            print ("Frame 50 saved as cascade.jpg")
        if count_frame >= 101 and count_frame <= 200:
            frame100 = cv.imread(FRAME_100)
            gray100 = cv.cvtColor(frame100, cv.COLOR_BGR2GRAY)
            frame_cascade(src, face_cascade, eyes_cascade, gray, colorface=(255, 0, 0), color=(0, 255, 0))
            frame_cascade(src, face_cascade, eyes_cascade, gray100, colorface=(0, 255, 0), color=(255, 0, 0))
            cv.imshow("detected100", src)
        elif count_frame > 200:
            cv.destroyWindow("detected100")
            frame_cascade(src, face_cascade, eyes_cascade, gray, colorface=(255, 0, 0), color=(0, 255, 0))
            cv.imshow("detected", src)
        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()


    