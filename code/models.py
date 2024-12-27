import os
import torch
from ultralytics import RTDETR
import inference
from inference import get_model

# Function to load the RBFA Player Detection Model
def load_player_detection_model(model_path="player_detect.pt"):
    """
    Loads the player detection model (RBFA Detection Model).
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RTDETR(model_path)
    model.to(device)
    print(f"Player detection model loaded on {device}")
    return model

# Function to load the Field Detection Model from Roboflow
def load_field_detection_model(api_key="cxtZ0KX74eCWIzrKBNkM", model_id="football-field-detection-f07vi/14"):
    """
    Loads the field detection model from Roboflow using the given API key and model ID.
    """
    field_detection_model = get_model(model_id=model_id, api_key=api_key)
    print("Field detection model loaded from Roboflow")
    return field_detection_model

# Example usage of the functions
if __name__ == "__main__":
    # Load models
    player_detection_model = load_player_detection_model()
    field_detection_model = load_field_detection_model()
    
    # You can now use these models for inference
    # For example, you can call player_detection_model.predict(image) or field_detection_model.predict(image)
