import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Path to the 'Dataset' folder.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # EXACT FOLDER NAMES FROM YOUR SCREENSHOT
        # Note: 'scan Images' vs 'images' - distinct capitalization matter!
        self.ct_path = os.path.join(root_dir, 'Brain Tumor CT scan Images')
        self.mri_path = os.path.join(root_dir, 'Brain Tumor MRI images')

        # Load file lists
        # We assume standard structure inside: /Healthy and /Tumor
        self.mri_healthy = self._get_files(os.path.join(self.mri_path, 'Healthy'))
        self.mri_tumor = self._get_files(os.path.join(self.mri_path, 'Tumor'))
        
        self.ct_healthy = self._get_files(os.path.join(self.ct_path, 'Healthy'))
        self.ct_tumor = self._get_files(os.path.join(self.ct_path, 'Tumor'))

        self.samples = []
        
        # Pair MRI images with random CT images of the same class
        # Label: 0 = Healthy, 1 = Tumor
        
        # 1. Healthy Pairs
        for mri_file in self.mri_healthy:
            self.samples.append((mri_file, 0))
            
        # 2. Tumor Pairs
        for mri_file in self.mri_tumor:
            self.samples.append((mri_file, 1))

    def _get_files(self, dir_path):
        """Helper to safely get files"""
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"CRITICAL ERROR: Could not find folder: {dir_path}")
            
        files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        return files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mri_file, label = self.samples[idx]
        
        # Load MRI
        mri_img = Image.open(mri_file).convert('L') # Grayscale
        
        # Load Random CT of same class
        if label == 0:
            ct_file = random.choice(self.ct_healthy)
        else:
            ct_file = random.choice(self.ct_tumor)
            
        ct_img = Image.open(ct_file).convert('L') # Grayscale

        if self.transform:
            mri_img = self.transform(mri_img)
            ct_img = self.transform(ct_img)

        return mri_img, ct_img, label

# --- SIMPLE TEST BLOCK ---
# This block only runs if you run "python dataset.py" directly.
if __name__ == "__main__":
    print("Testing Dataset...")
    
    # Define simple transform
    t = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    
    # POINTING TO YOUR 'Dataset' FOLDER
    # Since dataset.py is next to the Dataset folder, we use './Dataset'
    try:
        ds = BrainTumorDataset(root_dir='./Dataset', transform=t)
        print(f"Successfully loaded {len(ds)} paired samples.")
        
        img_mri, img_ct, lbl = ds[0]
        print(f"Sample 0 Shapes -> MRI: {img_mri.shape}, CT: {img_ct.shape}, Label: {lbl}")
        print("Data Pipeline is ready!")
    except Exception as e:
        print(f"Error: {e}")