import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset():
    api = KaggleApi()
    api.authenticate()  # Uses ~/.kaggle/kaggle.json
    dataset_slug = 'murtozalikhon/brain-tumor-multimodal-image-ct-and-mri'

    # Compute repo root (two levels up from this file: src/data -> src -> repo)
    repo_root = os.path.dirname(os.path.realpath(__file__))
    download_path = os.path.join(repo_root)
    os.makedirs(download_path, exist_ok=True)
    print(f"Downloading dataset to {download_path}...")
    api.dataset_download_files(dataset_slug, path=download_path, unzip=True)
    print(f"Download complete! Check {download_path} for files.")

if __name__ == "__main__":
    download_dataset()