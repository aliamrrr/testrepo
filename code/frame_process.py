import numpy as np
import supervision as sv

def resolve_goalkeepers_team_id(
    players: sv.Detections,
    goalkeepers: sv.Detections
) -> np.ndarray:
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players.class_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players.class_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)

    return np.array(goalkeepers_team_id)

# frame_process.py
import supervision as sv
import cv2
from ultralytics import RTDETR
import numpy as np
from models import load_player_detection_model  # Importer le modèle de détection des joueurs

# Définir les IDs des classes (joueurs, arbitres, etc.)
PLAYER_ID = 0
GOALKEEPER_ID = 1
BALL_ID = 2
REFEREE_ID = 3
SIDE_REFEREE_ID = 4
STAFF_MEMBER_ID = 5

model = load_player_detection_model()

def process_frame(frame, team_classifier):
    """
    Fonction qui effectue l'inférence sur une seule frame, détecte les objets,
    puis les annote avec leurs couleurs respectives.

    Args:
    - frame (ndarray): La frame de la vidéo à traiter.
    - team_classifier (object): Le modèle de classification des équipes pour les joueurs.

    Retourne:
    - frame_annotated (ndarray): La frame annotée avec les détections.
    """
    # Appliquer l'inférence avec le modèle RT-DETR pour détecter les objets
    results = model.predict(frame, conf=0.3)
    detections = sv.Detections(
        xyxy=results[0].boxes.xyxy.detach().cpu().numpy(),
        class_id=results[0].boxes.cls.detach().cpu().numpy(),
        confidence=results[0].boxes.conf.detach().cpu().numpy()
    )

    # Appliquer NMS (Non-Maximum Suppression) pour réduire les faux positifs
    detections = detections.with_nms(threshold=0.5, class_agnostic=True)

    # Regrouper les détections par type (joueurs, arbitres, etc.)
    referees_detections = detections[detections.class_id == REFEREE_ID]
    side_referees_detections = detections[detections.class_id == SIDE_REFEREE_ID]
    ball_detections = detections[detections.class_id == BALL_ID]
    staff_members_detections = detections[detections.class_id == STAFF_MEMBER_ID]
    players_detections = detections[detections.class_id == PLAYER_ID]
    goalkeepers_detections = detections[detections.class_id == GOALKEEPER_ID]

    # Extraire les crops des joueurs et classifier leurs équipes
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    # Résoudre les gardiens de but en fonction de l'équipe
    goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
        players_detections, goalkeepers_detections
    )

    # Fusionner les détections des joueurs et gardiens de but
    all_detections = sv.Detections.merge([players_detections, goalkeepers_detections])

    # Annoter la frame originale
    annotated_frame = frame.copy()

    # Définir les couleurs d'annotation pour chaque type d'objet
    class_colors = {
        "referee": (0, 255, 255),  # Couleur pour les arbitres (orange)
        "side_referee": (128, 0, 128),  # Couleur pour les arbitres latéraux (violet)
        "ball": (255, 0, 0),  # Couleur pour le ballon (rouge)
        "staff_member": (255, 255, 153)  # Couleur pour les membres du staff (jaune clair)
    }

    # Annoter les arbitres
    for i in range(len(referees_detections)):
        x1, y1, x2, y2 = map(int, referees_detections.xyxy[i])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_colors["referee"], 2)
        label = f"Referee ({referees_detections.confidence[i]:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["referee"], 2)

    # Annoter les arbitres latéraux
    for i in range(len(side_referees_detections)):
        x1, y1, x2, y2 = map(int, side_referees_detections.xyxy[i])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_colors["side_referee"], 2)
        label = f"Side Referee ({side_referees_detections.confidence[i]:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["side_referee"], 2)

    # Annoter les ballons
    for i in range(len(ball_detections)):
        x1, y1, x2, y2 = map(int, ball_detections.xyxy[i])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_colors["ball"], 2)
        label = f"Ball ({ball_detections.confidence[i]:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["ball"], 2)

    # Annoter les membres du staff
    for i in range(len(staff_members_detections)):
        x1, y1, x2, y2 = map(int, staff_members_detections.xyxy[i])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_colors["staff_member"], 2)
        label = f"Staff Member ({staff_members_detections.confidence[i]:.2f})"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_colors["staff_member"], 2)

    # Annoter les joueurs dynamiquement en fonction de leur équipe
    for i in range(len(all_detections)):
        x1, y1, x2, y2 = map(int, all_detections.xyxy[i])
        team_color = (0, 255, 0) if all_detections.class_id[i] == 1 else (255, 0, 0)  # Vert pour l'équipe 1, Bleu pour l'équipe 0
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), team_color, 2)
        label = f"{'Team 1' if all_detections.class_id[i] == 1 else 'Team 0'}"
        cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)

    return annotated_frame
