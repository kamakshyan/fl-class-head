# Save this as save_head.py
import torch
import numpy as np
import joblib # Library for saving Scikit-Learn models
from sklearn.neural_network import MLPClassifier
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MultiModalNet
from dataset import BrainTumorDataset
from task import DEVICE

MODEL_PATH = "global_model.pth"

def get_features(model, dataloader):
    model.eval()
    features = []
    labels_list = []
    print("Extracting features...")
    with torch.no_grad():
        for mri, ct, labels in dataloader:
            mri, ct = mri.to(DEVICE), ct.to(DEVICE)
            feat_mri = model.mri_cnn(mri)
            feat_ct = model.ct_cnn(ct)
            features.append(torch.cat((feat_mri, feat_ct), dim=1).cpu().numpy())
            labels_list.append(labels.numpy())
    return np.vstack(features), np.hstack(labels_list)

def main():
    # 1. Load Feature Extractor
    model = MultiModalNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    # 2. Get Data
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    # Use the full dataset to train the "Deployment" head
    dataset = BrainTumorDataset(root_dir='./Dataset', transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    X, y = get_features(model, loader)
    
    # 3. Train the MLP Head (The winner)
    print("Training Final MLP Head...")
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)
    mlp.fit(X, y)
    
    # 4. Save to disk
    joblib.dump(mlp, "local_mlp_head.pkl")
    print("✅ Saved 'local_mlp_head.pkl' successfully!")

if __name__ == "__main__":
    main()