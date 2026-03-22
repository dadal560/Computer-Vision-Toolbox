import cv2 as cv


def point_interet(orb, grayface, src, x, y):
    keypoints, descripteurs = orb.detectAndCompute(grayface, None)
    if not keypoints:
        return
    for kp in keypoints:
        kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
    cv.drawKeypoints(src, keypoints, src, color=(0, 255, 0),flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)


def frame_cascade(src, face_cascade, eyes_cascade, gray, orb,colorface=(255, 0, 0), color=(0, 0, 255)):
    frame_gray = cv.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.2,minNeighbors=5, minSize=(60, 60))
    for (x, y, w, h) in faces:
        cv.rectangle(src, (x, y), (x + w, y + h), colorface, 2)
        faceROI = frame_gray[y:y + h, x:x + w]
        point_interet(orb, faceROI, src, x, y)
        eyeRoi = faceROI[0:int(h / 2), :]
        eyes = eyes_cascade.detectMultiScale(eyeRoi)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            cv.circle(src, eye_center, radius, color, 2)


def main():
    cam = 0
    cap = cv.VideoCapture(cam)
    cap.set(cv.CAP_PROP_FPS, 25)
    orb = cv.ORB_create(nfeatures=500, scoreType=cv.ORB_FAST_SCORE)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyes_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")

    while True:
        ret, src = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        frame_cascade(src, face_cascade, eyes_cascade, gray, orb)
        cv.imshow("detected points", src)
        if cv.waitKey(1) == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()