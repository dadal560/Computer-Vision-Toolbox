import sys
import cv2 as cv
import numpy as np
import math
 
def alpha_mask(frame, img_with_alpha, orig=None, threshold=0):
    """Insert an image with alpha channel into another image. """
    # frame: the image to insert into
    # img_with_alpha: the image to be inserted
    # orig: the position to insert the image
    # threshold: the threshold to apply the mask
    # return: the modified image

    # get the size of the frame
    w, h, _ = frame.shape

    # if orig is None, set it to the top left of the frame
    if orig is None:
        orig = [0, 0]
    # if the position is out of the frame, return the frame
    if not (0 <= orig[0] < w and 0 <= orig[1] < h):
        return frame

    # get the start position to insert the image
    x_start, y_start = orig
    # if the position is out of the frame, set it to the top left of the frame
    x_start, y_start = max(x_start, 0), max(y_start, 0)

    # idem for the end position
    x_end, y_end = x_start + img_with_alpha.shape[0], y_start + img_with_alpha.shape[1]
    x_end, y_end = min(x_end, w), min(y_end, h)

    # create the mask
    mask = np.zeros((w, h, 4), dtype=np.uint8)
    # insert the image into the mask
    mask[x_start:x_end, y_start:y_end, :] = img_with_alpha[:x_end - x_start, :y_end - y_start, :]

    # apply the mask to the frame
    np.copyto(frame, mask[:, :, :3], where=(mask[:, :, 3] > threshold)[:, :, None])
    return frame

 
def main():
    cam = 0 
    cap = cv.VideoCapture(cam)
    cap.set(cv.CAP_PROP_FPS, 25)

    img_mustache = cv.imread('face_filters/mustache.png', cv.IMREAD_UNCHANGED)

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyes_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
    nose_cascade = cv.CascadeClassifier('face_filters/haarcascade_mcs_nose.xml')
    mouth_cascade = cv.CascadeClassifier('face_filters/haarcascade_mcs_mouth.xml')
    if nose_cascade.empty():
        raise IOError("Impossible de charger le classificateur de nez")    
    if mouth_cascade.empty():
        raise IOError("Impossible de charger le classificateur de bouche")    
    while True:
        rect, src = cap.read()
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
        for (x,y,w,h) in faces:
            frame = cv.rectangle(src, (x,y), (x+w, y+h), (255, 0, 255), 2)
            faceROI = frame_gray[y:y+h,x:x+w]

            mouthROI = faceROI[int(h/1.5):h, :] 
            mouths = mouth_cascade.detectMultiScale(mouthROI)
            for (mx, my, mw, mh) in mouths:
                # Calcul du facteur de redimensionnement basé sur la largeur de la bouche
                resize_factor = mw / img_mustache.shape[1]
                resized_mustache = cv.resize(img_mustache, (None, None), fx=resize_factor, fy=resize_factor)
                
                mus_h, mus_w, _ = resized_mustache.shape

                target_x = x + mx + (mw // 2) - (mus_w // 2)
                
                target_y = y + my + int(h/1.5) - (mus_h // 2)

                frame = alpha_mask(frame, resized_mustache, orig=[target_y, target_x])
            noseRoi = faceROI[int(h/3):int(h/1.5), :]
            noses = nose_cascade.detectMultiScale(noseRoi)
            for (nx, ny, nw, nh) in noses:
                cv.rectangle(frame, (x + nx, y + ny + int(h/3)), 
                            (x + nx + nw, y + ny + nh + int(h/3)), (0, 255, 0), 2)
                
            eyeRoi = faceROI[0:int(h/2), :]
            eyes = eyes_cascade.detectMultiScale(eyeRoi)
            for (x2,y2,w2,h2) in eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int(round((w2 + h2)*0.25))
                frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 2)
        cv.imshow("detected", src)
        key = cv.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()


    