import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalNet(nn.Module):
    def __init__(self):
        super(MultiModalNet, self).__init__()
        
        # --- BRANCH 1: MRI Feature Extractor ---
        # Input: [1, 128, 128]
        self.mri_cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32), # Replaces BatchNorm for DP safety
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: [32, 64, 64]
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2), # Output: [64, 32, 32]
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # Force output to fixed size [128, 4, 4]
            nn.Flatten() # Vector size: 128 * 4 * 4 = 2048
        )

        # --- BRANCH 2: CT Feature Extractor ---
        # Same architecture, but independent weights
        self.ct_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(16, 128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), 
            nn.Flatten()
        )

        # --- FUSION & CLASSIFIER HEAD ---
        # Input size = MRI features (2048) + CT features (2048) = 4096
        self.classifier = nn.Sequential(
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # Output: 2 classes (Healthy vs Tumor)
        )

    def forward(self, x_mri, x_ct):
        # 1. Extract Features Independently
        feat_mri = self.mri_cnn(x_mri)
        feat_ct = self.ct_cnn(x_ct)
        
        # 2. Fuse (Concatenate)
        combined = torch.cat((feat_mri, feat_ct), dim=1)
        
        # 3. Classify
        logits = self.classifier(combined)
        return logits

# --- VERIFICATION BLOCK ---
if __name__ == "__main__":
    # Create the model
    model = MultiModalNet()
    
    # Create dummy data matching your shapes [Batch, Channel, H, W]
    dummy_mri = torch.randn(4, 1, 128, 128)
    dummy_ct = torch.randn(4, 1, 128, 128)
    
    # Forward pass
    output = model(dummy_mri, dummy_ct)
    
    print("Model Architecture Test:")
    print(f"Input Shapes: {dummy_mri.shape}")
    print(f"Output Shape: {output.shape}") # Should be [4, 2]
    
    # Verify DP compatibility
    from opacus.validators import ModuleValidator
    errors = ModuleValidator.validate(model, strict=False)
    if not errors:
        print("✅ Model is DP-Compatible (No BatchNorm detected)")
    else:
        print("❌ Model has DP issues:", errors)