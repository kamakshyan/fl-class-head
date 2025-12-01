import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms
from dataset import BrainTumorDataset
from model import MultiModalNet

# --- CONFIGURATION ---
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DIRICHLET_ALPHA = 0.5 
NUM_CLIENTS = 2

def get_model():
    return MultiModalNet().to(DEVICE)

def partition_data():
    """
    1. Splits total data into: 80% Client Pool, 10% Global Val, 10% Global Test.
    2. Partitions the 80% Client Pool into N clients using Dirichlet (Non-IID).
    """
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    full_dataset = BrainTumorDataset(root_dir='./Dataset', transform=transform)
    
    # 1. Global Split (80% Train, 10% Val, 10% Test)
    total_len = len(full_dataset)
    val_len = int(0.1 * total_len)
    test_len = int(0.1 * total_len)
    train_len = total_len - val_len - test_len
    
    # Use a fixed generator for reproducibility
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_len, val_len, test_len], generator=torch.Generator().manual_seed(42)
    )
    
    # 2. Dirichlet Partitioning for Clients (using only the train_dataset)
    # We need to extract indices and labels from the Subset 'train_dataset'
    train_indices = train_dataset.indices
    all_labels = np.array([full_dataset.samples[i][1] for i in train_indices])
    
    num_classes = len(np.unique(all_labels))
    min_size = 0
    
    # Retry loop for Dirichlet
    while min_size < 10:
        idx_batch = [[] for _ in range(NUM_CLIENTS)]
        for k in range(num_classes):
            idx_k = np.where(all_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(DIRICHLET_ALPHA, NUM_CLIENTS))
            proportions = np.array([p * (len(idx_j) < len(all_labels) / NUM_CLIENTS) 
                                    for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])

    # Map relative indices back to original dataset indices
    client_subsets = []
    for relative_indices in idx_batch:
        # relative_indices points to train_dataset, we need indices for full_dataset
        original_indices = [train_indices[i] for i in relative_indices]
        client_subsets.append(Subset(full_dataset, original_indices))
        
    return client_subsets, val_dataset, test_dataset

# Execute split once globally
print("Partitioning data...")
client_partitions, global_val, global_test = partition_data()

def load_client_data(partition_id):
    return DataLoader(client_partitions[partition_id], batch_size=BATCH_SIZE, shuffle=True)

def load_global_test_data():
    return DataLoader(global_test, batch_size=BATCH_SIZE)

def load_global_val_data():
    return DataLoader(global_val, batch_size=BATCH_SIZE)