import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import MultiModalNet
import joblib  # Used to save/load Scikit-Learn models like RF/MLP

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GLOBAL_MODEL_PATH = "global_model.pth"
CLASSIFIER_PATH = "local_mlp_head.pkl" # We need to save this first!

def load_global_model():
    print(f"Loading Global Feature Extractor from {GLOBAL_MODEL_PATH}...")
    model = MultiModalNet().to(DEVICE)
    model.load_state_dict(torch.load(GLOBAL_MODEL_PATH, map_location=DEVICE))
    model.eval() # Freeze layers
    return model

def preprocess_image(image_path):
    """
    Loads an image, converts to grayscale, and resizes to 128x128.
    Returns a tensor shape [1, 1, 128, 128].
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert('L') # Force Grayscale
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension [1, 1, 128, 128]
    return img_tensor.to(DEVICE)

def predict(mri_path, ct_path, encoder, classifier_head):
    # 1. Preprocess
    mri_tensor = preprocess_image(mri_path)
    ct_tensor = preprocess_image(ct_path)
    
    # 2. Step 1: Extract Features (The "Eyes")
    with torch.no_grad():
        feat_mri = encoder.mri_cnn(mri_tensor)
        feat_ct = encoder.ct_cnn(ct_tensor)

        # --- DEBUG PRINT ---
        print(f"MRI Feature Max: {feat_mri.max().item():.4f} (Should be > 0.5)")
        print(f"CT Feature Max:  {feat_ct.max().item():.4f}  (Should be > 0.5)")
        # -------------------

        # Fuse
        features = torch.cat((feat_mri, feat_ct), dim=1)
        # Convert to Numpy for Scikit-Learn
        features_np = features.cpu().numpy()

    # 3. Step 2: Classify (The "Brain")
    # Returns 0 (Healthy) or 1 (Tumor)
    prediction = classifier_head.predict(features_np)[0]
    probability = classifier_head.predict_proba(features_np)[0] # Get confidence scores
    
    label = "TUMOR" if prediction == 1 else "HEALTHY"
    confidence = probability[prediction] * 100
    
    return label, confidence

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Models
    try:
        encoder = load_global_model()
        # Note: You need to run save_head.py (below) first to create this file!
        classifier = joblib.load(CLASSIFIER_PATH) 
        print("Models loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Did you run 'save_head.py' yet?")
        exit()

    # 2. Define your test images
    # REPLACE THESE PATHS with actual file paths from your dataset
    test_mri = "./test_images/ct_tumor.png" # Example
    test_ct = "./test_images/mri_tumor.jpg"   # Example

    print(f"\n--- Diagnosing Patient ---")
    print(f"MRI: {test_mri}")
    print(f"CT:  {test_ct}")
    
    try:
        diagnosis, conf = predict(test_mri, test_ct, encoder, classifier)
        print(f"\nFINAL DIAGNOSIS:  >>> {diagnosis} <<<")
        print(f"Confidence:       {conf:.2f}%")
    except Exception as e:
        print(f"\nError processing images: {e}")