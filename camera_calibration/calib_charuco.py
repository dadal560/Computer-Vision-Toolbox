import cv2
import numpy as np

# ==============================
# PARAMÈTRES À ADAPTER
# ==============================

# Nombre de cases du plateau (X, Y)
squaresX = 5
squaresY = 7

# Taille d'une case (mètres)
squareLength = 0.04

# Taille du marqueur ArUco (mètres)
markerLength = 0.02

MIN_IMAGES = 20

# ==============================
# DICTIONNAIRE + PLATEAU
# ==============================

aruco_dict = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

board = cv2.aruco.CharucoBoard(
    (squaresX, squaresY),
    squareLength,
    markerLength,
    aruco_dict
)

# Création du détecteur ChArUco (Nouvelle API OpenCV)
charuco_detector = cv2.aruco.CharucoDetector(board)

# ==============================
# STOCKAGE
# ==============================

all_charuco_corners = []
all_charuco_ids = []
image_size = None

# ==============================
# CAPTURE CAMERA
# ==============================

cap = cv2.VideoCapture(0)

print("Montrez le plateau ChArUco sous différents angles.")
print("Appuyez sur ESPACE pour capturer.")
print("Appuyez sur q pour terminer.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # La nouvelle méthode detectBoard s'occupe de tout (marqueurs + coins ChArUco)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    # Dessiner les marqueurs ArUco
    if marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

    # Dessiner les coins de l'échiquier ChArUco
    if charuco_ids is not None and len(charuco_ids) > 3:
        cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

    cv2.imshow("Charuco Calibration", frame)
    key = cv2.waitKey(1)

    if key == ord(' '):  # capture manuelle
        if charuco_ids is not None and len(charuco_ids) > 3:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            image_size = gray.shape[::-1]
            print(f"Image capturée : {len(all_charuco_corners)}")
        else:
            print("Pas assez de coins détectés sur cette image.")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ==============================
# CALIBRATION
# ==============================

if len(all_charuco_corners) >= 5:
    print("\nCalcul de la calibration en cours...")
    
    all_obj_points = []
    all_img_points = []
    
    # Associer les points 2D de l'image aux coordonnées 3D du plateau
    for img_corners, img_ids in zip(all_charuco_corners, all_charuco_ids):
        obj_points, img_points = board.matchImagePoints(img_corners, img_ids)
        if obj_points is not None and len(obj_points) > 3:
            all_obj_points.append(obj_points)
            all_img_points.append(img_points)

    # Calibration standard OpenCV
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_points, 
        all_img_points, 
        image_size, 
        None, 
        None
    )

    print("\nErreur RMS :", ret)
    print("\nMatrice intrinsèque :\n", camera_matrix)
    print("\nDistorsion :\n", dist_coeffs)

    np.savez("charuco_calibration.npz",
             camera_matrix=camera_matrix,
             dist_coeff=dist_coeffs)

    print("\nCalibration sauvegardée dans 'charuco_calibration.npz'.")

else:
    print("Pas assez d'images valides (minimum 5 requises).")

    

    