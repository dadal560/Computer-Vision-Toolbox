import streamlit as st

def sidebar_settings():
    """Gère le menu latéral de configuration"""
    st.sidebar.header("Configuration")
    algo = st.sidebar.selectbox("Choisir l'outil", [
        "Analyse de Visage (Haar + ORB)",
        "Filtre Facial (Moustache)",         
        "Détection de Cercles (HSV)",        
        "Canny Edge Detection",
        "Filtre Gris"
    ])
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader("Charger une image", type=['jpg', 'png', 'jpeg'])
        
    params = {}
    if algo == "Canny Edge Detection":
        params['low'] = st.sidebar.slider("Seuil Bas", 0, 255, 100)
        params['high'] = st.sidebar.slider("Seuil Haut", 0, 255, 200)
    elif algo == "Détection de Cercles (HSV)":
        st.sidebar.markdown("**Filtre Couleur (Hue)**")
        params['hue_min'] = st.sidebar.slider("Teinte Min", 0, 179, 25)
        params['hue_max'] = st.sidebar.slider("Teinte Max", 0, 179, 35)
        
    return algo, uploaded_file, params

def display_results(original, processed, algo_name, metrics=None):
    """Gère l'affichage en deux colonnes avant/après"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image Originale")
        st.image(original, channels="BGR", use_container_width=True)
        
    with col2:
        st.subheader(f"Résultat : {algo_name}")
        # Gestion des images en Niveaux de gris vs Couleur
        if len(processed.shape) == 2:
            st.image(processed, channels="GRAY", use_container_width=True)
        else:
            st.image(processed, channels="BGR", use_container_width=True)
            
        if metrics:
            st.success(metrics)