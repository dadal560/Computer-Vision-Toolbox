import cv2
import cv2.aruco as aruco
import numpy as np


cam=0 # capture from camera at location 0 

cap = cv2.VideoCapture(cam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)


def loadCameraCalibration(w, h):
    cpf = f"calibration{w}x{h}.yaml"
    fs = cv2.FileStorage(cpf, cv2.FILE_STORAGE_READ)
    if fs.isOpened():
        camMatrix = fs.getNode("camera_matrix").mat()
        distCoeffs = fs.getNode("distortion_coefficients").mat()
        print(f"Camera parameters loaded from file [{cpf}]")
    else:
        print(f"Failed to load camera parameters from file [{cpf}]")
        camMatrix = None
        distCoeffs = None
    return camMatrix, distCoeffs

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    https://stackoverflow.com/questions/75750177/solve-pnp-or-estimate-pose-single-markerswhich-is-better
    This will estimate the rvec and tvec for each of the marker corners detected by:
    corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
    [marker_size / 2, marker_size / 2, 0],
    [marker_size / 2, -marker_size / 2, 0],
    [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs,  tvecs, trash

def incrustation_image(pts,image_path):
    image = cv2.imread(image_path)
    corners = np.array([[0, 0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]])
    mtx_homographique = cv2.getPerspectiveTransform(corners.astype(np.float32), pts.astype(np.float32))
    incrustation = cv2.warpPerspective(image, mtx_homographique, (640, 480))
    return incrustation


taille_marqueur_m = 0.006
s = 0.003 
coordonnees_cube = np.array([
    [s, s,  0], [-s, s,  0], [-s,  -s,  0], [s,  -s,  0], 
    [s, s, 0.006], [-s, s, 0.006], [-s,  -s, 0.006], [s,  -s, 0.006] 
])
while True:
    ret, img = cap.read()

    dico = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters=aruco.DetectorParameters()

    inimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = aruco.ArucoDetector(dico,parameters)
    corners, ids, rejectedImgPos = detector.detectMarkers(img)

    intrinsèques , coefficients_distorsion, = loadCameraCalibration(640, 480)
    print(intrinsèques)
    rvecs, tvecs, trash = my_estimatePoseSingleMarkers(corners, taille_marqueur_m, intrinsèques, coefficients_distorsion)


    outimg = aruco.drawDetectedMarkers(img, corners, ids)

    for i in range(len(rvecs)):
        outimg = cv2.drawFrameAxes(outimg, intrinsèques, coefficients_distorsion, rvecs[i], tvecs[i], 0.003)
        imagePoints, _ = cv2.projectPoints(coordonnees_cube, rvecs[i], tvecs[i], intrinsèques, coefficients_distorsion)
        pts = imagePoints.reshape(-1, 2).astype(int)
        cv2.polylines(outimg, [pts[:4]], True, (0, 0, 255), 2)
        cv2.polylines(outimg, [pts[4:]], True, (0, 0, 255), 2)
        for j in range(4):
            cv2.line(outimg, pts[j], pts[j+4], (0, 0, 255), 2)
        outimg = cv2.addWeighted(outimg, 1, incrustation_image(pts[4:], "GreenEye.png"), 1, 0)
        
    cv2.imshow("output", outimg)
    key = cv2.waitKey(1)
    if key == 27:
        break
