import cv2 as cv
import numpy as np

def get_resources():
    """Charge les modèles Haar Cascades et ORB une seule fois"""
    orb = cv.ORB_create(nfeatures=500, scoreType=cv.ORB_FAST_SCORE)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyes_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_eye.xml")
    return orb, face_cascade, eyes_cascade

def process_face_analysis(src, face_cascade, eyes_cascade, orb):
    """Détection visage + yeux + points ORB restreints au visage"""
    output = src.copy()
    gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    
    kp_list, des_list = [], []
    for (x, y, w, h) in faces:
        # Boîte englobante du visage
        cv.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        faceROI = frame_gray[y:y+h, x:x+w]
        
        # Extraction ORB restreinte à la ROI
        kp, des = orb.detectAndCompute(faceROI, None)
        if kp:
            for p in kp:
                p.pt = (p.pt[0] + x, p.pt[1] + y)
            cv.drawKeypoints(output, kp, output, color=(0, 255, 0))
            
        kp_list.append(kp)
        des_list.append(des)
        
    return kp_list, des_list, output

def apply_mustache_filter(src, face_cascade, mouth_cascade):
    """Incruste la moustache (face_filters)"""
    output = src.copy()
    gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
    
    # Charger l'image de la moustache avec transparence (canal alpha)
    try:
        mustache = cv.imread('face_filters/mustache.png', cv.IMREAD_UNCHANGED)
    except:
        return output # Si l'image n'est pas trouvée
        
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    
    for (x, y, w, h) in faces:
        faceROI = gray[y:y+h, x:x+w]
        mouthROI = faceROI[int(h/1.5):h, :] 
        mouths = mouth_cascade.detectMultiScale(mouthROI)
        
        for (mx, my, mw, mh) in mouths:
            # Redimensionner la moustache
            resize_factor = mw / mustache.shape[1]
            resized_mustache = cv.resize(mustache, (None, None), fx=resize_factor, fy=resize_factor)
            mus_h, mus_w, _ = resized_mustache.shape

            target_x = x + mx + (mw // 2) - (mus_w // 2)
            target_y = y + my + int(h/1.5) - (mus_h // 2)

            # --- Fonction Alpha Mask (adaptée de ton code) ---
            orig = [max(target_y, 0), max(target_x, 0)]
            x_end = min(orig[0] + mus_h, output.shape[0])
            y_end = min(orig[1] + mus_w, output.shape[1])
            
            mask = np.zeros((output.shape[0], output.shape[1], 4), dtype=np.uint8)
            mask[orig[0]:x_end, orig[1]:y_end, :] = resized_mustache[:x_end-orig[0], :y_end-orig[1], :]
            
            np.copyto(output, mask[:,:,:3], where=(mask[:,:,3] > 0)[:,:,None])

    return output

def detect_hsv_circles(src, hue_min, hue_max):
    """Détection via espace HSV (circle_detection)"""
    output = src.copy()
    hsv = cv.cvtColor(output, cv.COLOR_BGR2HSV)
    
    # Création du masque selon la teinte choisie au slider
    mask = cv.inRange(hsv, (hue_min, 100, 100), (hue_max, 255, 255))
    blurred = cv.GaussianBlur(mask, (5,5), 0)
    
    canny = cv.Canny(blurred, 100, 200)
    contours, _ = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    import math
    circles_found = 0
    for zone in contours:
        area = cv.contourArea(zone)
        perimeter = cv.arcLength(zone, True)
        if perimeter > 0:
            q = (4 * math.pi * area) / (perimeter**2)
            if q > 0.85: # Ton critère de circularité !
                (cx, cy), radius = cv.minEnclosingCircle(zone)
                cv.circle(output, (int(cx), int(cy)), int(radius), (0, 255, 0), 3)
                circles_found += 1
                
    return output, mask, circles_found