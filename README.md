# Computer-Vision-Toolbox

## Modules

### `camera_calibration/`
Calibration intrinsèque d'une caméra avec un plateau **ChArUco** (marqueurs ArUco + échiquier).

- Capture interactive : appuyer sur `ESPACE` pour capturer une image, `q` pour terminer.
- Utilise la nouvelle API `cv2.aruco.CharucoDetector`.
- Sauvegarde les paramètres dans `charuco_calibration.npz` (`camera_matrix`, `dist_coeff`).

**Usage :**
```bash
python camera_calibration/calib_charuco.py
```

### `circle_detection/`
Détection en temps réel d'un **objet circulaire jaune** via segmentation couleur en espace HSV.

**Pipeline commun :**
1. Conversion BGR → HSV
2. Masque couleur jaune : `H ∈ [25, 35]`, `S ∈ [100, 255]`, `V ∈ [100, 255]`
3. Flou gaussien 5×5 pour réduire le bruit
4. Détection du cercle (Canny+contours ou Hough)

| Script | Méthode | Fonctionnalité |
|--------|---------|----------------|
| `contour_detection.py` | Canny + contours | Détection brute sans calibration |
| `contour-calibration_detection.py` | Canny + contours | + estimation de distance réelle |
| `hough_detection.py` | Transformée de Hough | Détection directe de cercles |

**Critère de circularité** (scripts contours) :
$$q = \frac{4\pi \cdot A}{P^2} > 0.85$$
Un contour n'est retenu que s'il est suffisamment rond.

**Estimation de distance** (script avec calibration) :
$$d = \frac{D_{réel} \times f}{2\sqrt{A/\pi}}$$
avec $D_{réel}$ = 4 cm et $f$ la focale extraite de `charuco_calibration.npz`.

Requiert `charuco_calibration.npz` pour la version avec estimation de distance.

---

### `color_detection/`
Interface interactive pour calibrer un masque de couleur en espace **HSV**.

- Six trackbars pour ajuster H/S/V min et max en temps réel.
- Utile pour trouver les plages HSV avant de les coder en dur dans un autre script.

---

### `aruco_pose/`
Détection de marqueurs ArUco avec estimation de pose 6DoF et réalité augmentée.

- Estimation de pose via `solvePnP` (méthode `IPPE_SQUARE`).
- Dessin d'un cube 3D sur le marqueur.
- Incrustation d'une image (`GreenEye.png`) sur la face supérieure du cube via homographie.
- Requiert un fichier de calibration YAML : `calibration640x480.yaml`.

### `face_filters/`
Détection de visage avec incrustation d'éléments graphiques via canal alpha.

**`haar_face_filters.py`** — filtre facial en temps réel :
- Détecte visage, yeux, nez et bouche via **Haar Cascades**
- Incruste une moustache PNG redimensionnée dynamiquement sur la bouche détectée
- Utilise `alpha_mask()` pour respecter la transparence du PNG
- Requiert `mustache.png` et les cascades custom `haarcascade_mcs_nose.xml` / `haarcascade_mcs_mouth.xml`

> Les cascades nez/bouche ne sont pas incluses dans OpenCV par défaut —
> les placer dans le dossier `face_filters/` avant d'exécuter.

---

### `interest_points/`
Détection de points d'intérêt et reconnaissance de visages.

| Script | Description |
|--------|-------------|
| `detection_points_interet.py` | ORB sur flux caméra avec flou gaussien |
| `detection_points_interet_v2.py` | Compare les points ORB du flux en direct avec une frame de référence (frame 50) |
| `detection_visage_yeux.py` | Haar Cascade frontal face + eyes, avec comparaison live vs frame figée |
| `detection_points_visage_roi.py` | ORB restreint à la région d'intérêt (ROI) du visage détecté avec Haar Cascade frontal face + eyes |
| `face_points_matching.py` | Détection multi-visages et matching (appariement) des points d'intérêt entre les deux premiers visages détectés via BFMatcher |
| `lena_demo.py` | Démo statique sur `lena.jpg` : Haar Cascade (visage + yeux) + ORB sur ROI |

---

#### `detection_points_visage_roi.py` — Paramètres détaillés

Ce script combine la détection de visage par **Haar Cascade** et l'extraction de points d'intérêt **ORB** restreinte à la ROI du visage.

**Haar Cascade — `detectMultiScale`**
| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| `scaleFactor` | `1.2` | Facteur de réduction de l'image à chaque échelle. Plus proche de 1.0 = plus précis mais plus lent |
| `minNeighbors` | `5` | Nombre de détections voisines requises pour valider un visage. Augmenter réduit les faux positifs |
| `minSize` | `(60, 60)` | Taille minimale du visage détecté en pixels. Ignorer les visages trop petits |

**ORB — `ORB_create`**
| Paramètre | Valeur par défaut | Description |
|-----------|-------------------|-------------|
| `nfeatures` | `500` | Nombre maximum de points d'intérêt à détecter. Réduire pour alléger l'affichage |
| `scoreType` | `ORB_FAST_SCORE` | Méthode de scoring : `ORB_FAST_SCORE` (rapide) ou `ORB_HARRIS_SCORE` (plus précis) |
| `scaleFactor` | `1.2` | Facteur de l'image pyramidale entre chaque niveau |
| `nlevels` | `8` | Nombre de niveaux de la pyramide. Réduire pour ignorer les petits détails |

**Flags de dessin — `drawKeypoints`**
| Flag | Rendu |
|------|-------|
| `DRAW_MATCHES_FLAGS_DEFAULT` | Petits points fixes, sans orientation ni taille |
| `DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS` | Cercles avec taille et orientation (plus verbeux) |