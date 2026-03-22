import cv2 as cv


def point_interet(orb, grayface, src, x, y):
    keypoints, descripteurs = orb.detectAndCompute(grayface, None)
    if not keypoints:
        return [], None
    for kp in keypoints:
        kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
    cv.drawKeypoints(src, keypoints, src, color=(0, 255, 0),flags=cv.DRAW_MATCHES_FLAGS_DEFAULT)
    return keypoints, descripteurs


def frame_cascade(src, face_cascade, eyes_cascade, gray, orb,colorface=(255, 0, 0), color=(0, 0, 255)):
    frame_gray = cv.equalizeHist(gray)
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    kp_list, des_list = [], []
    for (x, y, w, h) in faces:
        cv.rectangle(src, (x, y), (x + w, y + h), colorface, 2)
        faceROI = frame_gray[y:y + h, x:x + w]
        kp, des = point_interet(orb, faceROI, src, x, y)
        kp_list.append(kp)
        des_list.append(des)
        eyeRoi = faceROI[0:int(h / 2), :]
        eyes = eyes_cascade.detectMultiScale(eyeRoi)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            cv.circle(src, eye_center, int(round((w2 + h2) * 0.25)), color, 2)
    return kp_list, des_list


def main():
    cam = 0
    cap = cv.VideoCapture(cam)
    cap.set(cv.CAP_PROP_FPS, 25)
    orb = cv.ORB_create(nfeatures=500, scoreType=cv.ORB_FAST_SCORE)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades +"haarcascade_frontalface_default.xml")
    eyes_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
    while True:
        ret, src = cap.read()
        if not ret:
            break
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        kp_list, des_list = frame_cascade(src, face_cascade, eyes_cascade,gray, orb)
        
        # Matching entre les deux premiers visages détectés
        if len(des_list) >= 2 and des_list[0] is not None and des_list[1] is not None:
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_list[0], des_list[1])
            matches = sorted(matches, key=lambda x: x.distance)
            im3 = cv.drawMatches(src, kp_list[0], src, kp_list[1], matches[:10], None, flags=2)
            cv.imshow("matching", im3)
        else:
            cv.imshow("detected", src)
        if cv.waitKey(1) == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()