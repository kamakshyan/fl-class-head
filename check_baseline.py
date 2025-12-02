import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MultiModalNet
from dataset import BrainTumorDataset
from task import DEVICE

MODEL_PATH = "global_model.pth"

def main():
    print(f"Checking Baseline Feature Values on {DEVICE}...")
    model = MultiModalNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Load just a few training images
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = BrainTumorDataset(root_dir='./Dataset', transform=transform)
    # Get a batch
    loader = DataLoader(dataset, batch_size=10, shuffle=True)
    mri, ct, labels = next(iter(loader))
    
    mri, ct = mri.to(DEVICE), ct.to(DEVICE)

    with torch.no_grad():
        feat_mri = model.mri_cnn(mri)
        feat_ct = model.ct_cnn(ct)
        
        # Analyze Tumor vs Healthy activations
        for i in range(len(labels)):
            label_str = "TUMOR" if labels[i].item() == 1 else "HEALTHY"
            mri_val = feat_mri[i].max().item()
            ct_val = feat_ct[i].max().item()
            print(f"Image {i} [{label_str}]: MRI_Max={mri_val:.2f} | CT_Max={ct_val:.2f}")

if __name__ == "__main__":
    main()