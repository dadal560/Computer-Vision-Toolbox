import streamlit as st
import numpy as np
from PIL import Image
import cv2 as cv

# Import de tes modules personnalisés
import vision_engine as ve
import ui_components as ui

# Configuration de la page
st.set_page_config(page_title="Gwendal Henry - CV Toolbox", layout="wide")
st.title("🚀 Computer Vision Dashboard")
st.markdown("Interface interactive pour mon dépôt `Computer-Vision-Toolbox`.")

# 1. Initialisation des modèles (Cache pour la performance)
@st.cache_resource
def init_models():
    return ve.get_resources()

orb, face_cv, eyes_cv = init_models()

# 2. Chargement de l'interface
algo, file, params = ui.sidebar_settings()

# 3. Exécution si un fichier est chargé
if file:
    # Conversion Image PIL -> OpenCV BGR
    img = np.array(Image.open(file))
    img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    
    # --- ROUTAGE DES ALGORITHMES ---
    if algo == "Analyse de Visage (Haar + ORB)":
        kp, des, res_img = ve.process_face_analysis(img_bgr, face_cv, eyes_cv, orb)
        ui.display_results(img_bgr, res_img, algo, f"Visages détectés : {len(kp)}")
        
    elif algo == "Canny Edge Detection":
        gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        res_img = cv.Canny(gray, params['low'], params['high'])
        ui.display_results(img_bgr, res_img, algo)
        
    elif algo == "Filtre Gris":
        res_img = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
        ui.display_results(img_bgr, res_img, algo)
        
    elif algo == "Filtre Facial (Moustache)":
        mouth_cascade = cv.CascadeClassifier('face_filters/haarcascade_mcs_mouth.xml')
        res_img = ve.apply_mustache_filter(img_bgr, face_cv, mouth_cascade)
        ui.display_results(img_bgr, res_img, algo)
        
    elif algo == "Détection de Cercles (HSV)":
        res_img, mask, count = ve.detect_hsv_circles(img_bgr, params['hue_min'], params['hue_max'])
        ui.display_results(img_bgr, res_img, algo, f"Cercles parfaits détectés : {count}")

else:
    st.info("👈 Charge une image dans la barre latérale pour tester les algorithmes.")