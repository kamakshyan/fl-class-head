import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless mode for server
import matplotlib.pyplot as plt
import seaborn as sns

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier # <--- NEW: Gradient-Based Head

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms

from model import MultiModalNet
from dataset import BrainTumorDataset
from task import DEVICE, BATCH_SIZE

MODEL_PATH = "global_model.pth"

def get_features_and_labels(model, dataloader):
    model.eval()
    features = []
    labels_list = []
    print(f"Extracting features on {DEVICE}...")
    with torch.no_grad():
        for i, (mri, ct, labels) in enumerate(dataloader):
            mri, ct = mri.to(DEVICE), ct.to(DEVICE)
            
            # Extract features from the frozen "Eyes"
            feat_mri = model.mri_cnn(mri)
            feat_ct = model.ct_cnn(ct)
            combined = torch.cat((feat_mri, feat_ct), dim=1)
            
            features.append(combined.cpu().numpy())
            labels_list.append(labels.numpy())
            
    return np.vstack(features), np.hstack(labels_list)

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Tumor'], yticklabels=['Healthy', 'Tumor'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

def main():
    # 1. Load Model & Data
    print(f"Loading model from {MODEL_PATH}...")
    model = MultiModalNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except FileNotFoundError:
        print("Run 'flwr run .' first!")
        return

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = BrainTumorDataset(root_dir='./Dataset', transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 2. Extract Features
    X, y = get_features_and_labels(model, dataloader)
    print(f"Features shape: {X.shape}")
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    results = {}

    print("\n--- COMPARISON STARTED ---")

    # --- A. Random Forest (Tree-based) ---
    print("1. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['Random Forest'] = accuracy_score(y_test, rf_pred)
    plot_confusion_matrix(y_test, rf_pred, "Random Forest CM", "cm_rf.png")

    # --- B. SVM (Margin-based) ---
    print("2. Training SVM...")
    svm = SVC(kernel='rbf', C=1.0)
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    results['SVM'] = accuracy_score(y_test, svm_pred)
    plot_confusion_matrix(y_test, svm_pred, "SVM CM", "cm_svm.png")

    # --- C. KNN (Distance-based) ---
    print("3. Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    results['KNN'] = accuracy_score(y_test, knn_pred)
    plot_confusion_matrix(y_test, knn_pred, "KNN CM", "cm_knn.png")

    # --- D. MLP (Gradient-based) --- <--- THIS IS NEW
    print("4. Training MLP (Gradient-Based)...")
    # A simple neural network head: Input -> 256 -> ReLU -> Output
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=500, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    results['MLP (Gradient)'] = accuracy_score(y_test, mlp_pred)
    plot_confusion_matrix(y_test, mlp_pred, "MLP Confusion Matrix", "cm_mlp.png")

    # --- PRINT RESULTS ---
    print("\nFinal Results:")
    for name, acc in results.items():
        print(f"{name}: {acc*100:.2f}%")

    # --- PLOT COMPARISON BAR CHART ---
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), [v*100 for v in results.values()], 
                   color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    plt.ylabel('Accuracy (%)')
    plt.title('Classifier Head Comparison (Non-IID + Differential Privacy)')
    plt.ylim(0, 100)
    
    # Add numbers on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}%", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("comparison_chart.png")
    print("\nSaved comparison_chart.png")

if __name__ == "__main__":
    main()