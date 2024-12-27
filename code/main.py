import streamlit as st
import tempfile
from models import load_player_detection_model, load_field_detection_model
from team_classifier import show_crops, extract_player_crops, fit_team_classifier
from PIL import Image, ImageDraw
from frame_process import process_frame
import cv2

# Interface Streamlit
st.title("Football Video Processing")

# Sélection du menu
menu = st.sidebar.selectbox("Sélectionner l'option", ["Visualiser la vidéo", "Collecter, classifier et visualiser les équipes", "Visualiser une frame annotée", "Traitement en temps réel"])

# 1. **Menu pour visualiser la vidéo**
if menu == "Visualiser la vidéo":
    uploaded_file = st.file_uploader("Déposez votre vidéo .mp4", type=["mp4"])

    if uploaded_file is not None:
        # Créer un fichier temporaire pour la vidéo téléchargée
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Lire et afficher la vidéo
        st.video(tmp_file_path)

# 2. **Menu pour collecter, classifier et visualiser les équipes**
elif menu == "Collecter, classifier et visualiser les équipes":
    uploaded_file = st.file_uploader("Déposez votre vidéo .mp4", type=["mp4"])

    if uploaded_file is not None:
        # Créer un fichier temporaire pour la vidéo téléchargée
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Charger les modèles de détection des joueurs et du terrain
        player_detection_model = load_player_detection_model()  # Chargement du modèle des joueurs
        field_detection_model = load_field_detection_model()    # Chargement du modèle du terrain

        # Afficher un bouton pour démarrer la détection des crops
        if st.button("Détecter les joueurs"):
            # Afficher un indicateur de chargement pendant la détection
            with st.spinner("Détection en cours..."):
                # Extraire les crops des joueurs
                crops = extract_player_crops(tmp_file_path, player_detection_model)
                
                # Vérifier si des crops ont été détectés
                if crops:
                    # Afficher les crops dans l'interface
                    show_crops(crops)
                    st.write(f"Nombre de joueurs détectés : {len(crops)}")  # Affiche le nombre de crops détectés

                    with st.spinner("Classification en cours..."):
                        try:
                            # Entraîner le classifieur
                            team_classifier = fit_team_classifier(crops, device="cpu")
                            st.session_state.team_classifier = team_classifier  # Sauvegarder dans session_state
                            st.success("Le modèle de classification des équipes est prêt!")
                            st.write("Vous pouvez maintenant visualiser les équipes.")
                        except Exception as e:
                            st.error(f"Erreur lors de la classification : {str(e)}")
                else:
                    st.warning("Aucun joueur détecté. Veuillez réessayer avec une autre vidéo.")

# 3. **Menu pour visualiser une frame annotée**
elif menu == "Visualiser une frame annotée":
    uploaded_file = st.file_uploader("Déposez votre vidéo .mp4", type=["mp4"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Vérifier si le modèle de classification des équipes est disponible
        if "team_classifier" not in st.session_state:
            st.warning("Veuillez d'abord classifier les joueurs dans l'onglet précédent.")
        else:
            team_classifier = st.session_state.team_classifier  # Charger le modèle de classification des équipes

            # Sélectionner l'indice de la frame
            frame_index = st.slider("Sélectionner l'indice de la frame", 0, 100, 0)  # Ajuste la plage selon la longueur de la vidéo

            # Lire la vidéo et sélectionner la frame correspondante
            cap = cv2.VideoCapture(tmp_file_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            if ret:
                # Afficher la frame originale
                st.image(frame, channels="BGR", caption=f"Frame {frame_index}")

                # Traiter la frame et obtenir l'annotée
                annotated_frame = process_frame(frame, team_classifier)

                # Afficher la frame annotée
                st.image(annotated_frame, channels="BGR", caption=f"Frame annotée {frame_index}")
            cap.release()

# 4. **Menu pour le traitement en temps réel**
elif menu == "Traitement en temps réel":
    uploaded_file = st.file_uploader("Déposez votre vidéo .mp4", type=["mp4"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Charger le modèle de classification des équipes
        if "team_classifier" not in st.session_state:
            st.warning("Veuillez d'abord classifier les joueurs dans l'onglet 'Collecter, classifier et visualiser les équipes'.")
        else:
            team_classifier = st.session_state.team_classifier

            # Bouton pour démarrer la détection en temps réel
            if st.button("Start Detection"):
                cap = cv2.VideoCapture(tmp_file_path)
                frame_placeholder = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Traiter la frame et obtenir l'annotée
                    annotated_frame = process_frame(frame, team_classifier)

                    # Combiner les frames côte à côte
                    combined_frame = cv2.hconcat([frame, annotated_frame])

                    # Mettre à jour la même image dans Streamlit
                    frame_placeholder.image(combined_frame, channels="BGR")

                cap.release()

# Fonction pour visualiser les équipes
def visualize_teams(crops, team_labels, cols=5):
    """
    Visualise les crops des joueurs avec des étiquettes pour indiquer leur équipe.
    
    Args:
    - crops (list): Liste des images crops des joueurs détectés.
    - team_labels (list): Liste des étiquettes de classification pour chaque joueur.
    - cols (int): Nombre de colonnes dans la grille.
    """
    if not crops:
        st.warning("Aucun crop détecté pour visualisation.")
        return

    # Définir les couleurs des équipes
    team_colors = {0: "blue", 1: "red"}  # Équipe 0 : bleu, Équipe 1 : rouge

    # Calcul du nombre de lignes nécessaires pour afficher toutes les images
    rows = len(crops) // cols + (1 if len(crops) % cols != 0 else 0)

    # Afficher les images dans un format de grille
    for i in range(rows):
        row_crops = crops[i * cols:(i + 1) * cols]
        row_labels = team_labels[i * cols:(i + 1) * cols]

        cols_layout = st.columns(cols)
        for j, (crop, label) in enumerate(zip(row_crops, row_labels)):
            if j < len(cols_layout):
                # Convertir le crop en format Image PIL
                pil_image = Image.fromarray(crop)

                # Ajouter un rectangle coloré pour indiquer l'équipe
                draw = ImageDraw.Draw(pil_image)
                color = team_colors.get(label, "gray")  # Couleur par défaut : gris si aucune étiquette valide
                draw.rectangle([0, 0, pil_image.width, pil_image.height], outline=color, width=10)

                # Afficher le crop avec étiquette
                cols_layout[j].image(
                    pil_image,
                    use_column_width=True,
                    caption=f"Équipe {label + 1}" if label in team_colors else "Inconnue",
                )
