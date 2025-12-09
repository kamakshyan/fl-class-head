import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Safe for remote server (Headless)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from glob import glob

# --- CONFIGURATION ---
DATASET_ROOT = "./Dataset"
OUTPUT_DIR = "eda_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- SEABORN STYLING (The Professor's Preference) ---
# 'whitegrid' is standard for scientific papers (clean background, gridlines for reading values)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)

def get_image_paths():
    """Crawls the dataset and returns a DataFrame of file info."""
    data = []
    
    modalities = {
        "MRI": "Brain Tumor MRI images",
        "CT": "Brain Tumor CT scan Images"
    }
    classes = ["Healthy", "Tumor"]
    
    for mod_name, folder_name in modalities.items():
        for label in classes:
            path = os.path.join(DATASET_ROOT, folder_name, label)
            if not os.path.exists(path):
                continue
                
            files = glob(os.path.join(path, "*"))
            files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            
            for f in files:
                data.append({
                    "path": f,
                    "modality": mod_name,
                    "label": label
                })
    
    return pd.DataFrame(data)

def save_plot(filename):
    """Helper to save in both PNG (Web) and PDF (LaTeX/Paper) formats."""
    # PNG for slides/quick view
    plt.savefig(f"{OUTPUT_DIR}/{filename}.png", dpi=300, bbox_inches='tight')
    # PDF for the research paper (Vector graphics)
    plt.savefig(f"{OUTPUT_DIR}/{filename}.pdf", format='pdf', bbox_inches='tight')
    print(f"Saved {filename}.png & {filename}.pdf")
    plt.close()

def analyze_distributions(df):
    """Generates a bar chart of data counts using Seaborn."""
    plt.figure(figsize=(8, 6))
    
    # Use Seaborn's countplot
    ax = sns.countplot(
        data=df, 
        x="modality", 
        hue="label", 
        palette="viridis",
        edgecolor="black", # Adds a nice border to bars
        linewidth=1
    )
    
    ax.set_title("Class Distribution by Modality", fontweight='bold')
    ax.set_xlabel("Modality")
    ax.set_ylabel("Number of Images")
    
    # Add counts on top of bars
    for container in ax.containers:
        ax.bar_label(container, padding=3)
        
    save_plot("distribution_chart")

def visualize_samples(df):
    """Saves a grid of 4 random images."""
    # We temporarily turn off the grid for images because it looks messy
    with sns.axes_style("white"): 
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        scenarios = [
            ("MRI", "Healthy", axes[0,0]),
            ("MRI", "Tumor", axes[0,1]),
            ("CT", "Healthy", axes[1,0]),
            ("CT", "Tumor", axes[1,1])
        ]
        
        for mod, label, ax in scenarios:
            subset = df[(df["modality"] == mod) & (df["label"] == label)]
            if not subset.empty:
                sample = subset.sample(1).iloc[0]
                img = Image.open(sample["path"]).convert('L')
                ax.imshow(img, cmap='gray')
                ax.set_title(f"{mod} - {label}", fontweight='bold')
                ax.axis('off') # Hide axes ticks for images
                
        plt.tight_layout()
        save_plot("sample_grid")

def analyze_image_properties(df, sample_n=500):
    """Analyzes image sizes and pixel intensities using Seaborn distributions."""
    print(f"Analyzing image properties (Sampling {sample_n} images)...")
    
    if len(df) > sample_n:
        df_sample = df.sample(sample_n, random_state=42)
    else:
        df_sample = df
        
    widths = []
    heights = []
    mean_intensities = []
    
    for _, row in df_sample.iterrows():
        try:
            img = Image.open(row["path"]).convert('L')
            w, h = img.size
            arr = np.array(img)
            
            widths.append(w)
            heights.append(h)
            mean_intensities.append(arr.mean())
        except:
            pass
            
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Dimensions (Scatter)
    sns.scatterplot(
        x=widths, 
        y=heights, 
        alpha=0.6, 
        color="#2ecc71", 
        edgecolor="black", 
        linewidth=0.5, 
        ax=ax1
    )
    ax1.set_title("Image Resolution Distribution", fontweight='bold')
    ax1.set_xlabel("Width (px)")
    ax1.set_ylabel("Height (px)")
    
    # Plot 2: Intensities (Histogram + KDE)
    sns.histplot(
        mean_intensities, 
        bins=30, 
        kde=True, 
        color="#9b59b6", 
        edgecolor="black", 
        line_kws={'linewidth': 2}, # Make the KDE line thicker
        ax=ax2
    )
    ax2.set_title("Pixel Intensity Distribution (0-255)", fontweight='bold')
    ax2.set_xlabel("Mean Pixel Value")
    ax2.set_ylabel("Frequency")
    
    save_plot("image_properties")

def main():
    print("--- Starting Exploratory Data Analysis (Seaborn Mode) ---")
    df = get_image_paths()
    
    if df.empty:
        print("Error: No images found! Check DATASET_ROOT in the script.")
        return

    print(f"Found {len(df)} total images.")
    
    # 1. Bar Chart
    analyze_distributions(df)
    
    # 2. Sample Grid
    visualize_samples(df)
    
    # 3. Sizes & Intensities
    analyze_image_properties(df)
    
    print("\nEDA Complete. Check the 'eda_results' folder for PDF and PNG files.")

if __name__ == "__main__":
    main()