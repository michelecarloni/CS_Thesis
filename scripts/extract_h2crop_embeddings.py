import os
import sys

# Ensure project root is accessible for custom modules
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn.functional as F
import numpy as np
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from H2Crop.data_structures import h2crop_band_mapping
from H2Crop.H2CropTileDataset import H2CropTileDataset

def get_valid_wavelengths():
    """
    Parses the h2crop_band_mapping to extract the 218 valid wavelengths
    and converts them from nanometers (nm) to micrometers (um) for DOFA.
    """
    wave_list_um = []
    for idx in range(224):
        wave = h2crop_band_mapping.get(idx, "NaN")
        if wave != "NaN":
            wave_list_um.append(wave / 1000.0)
    return wave_list_um

def extract_dofa_embeddings(data_loader, model, device, wave_list_um, desc="Extracting"):
    model.eval()
    embeddings_list = []
    
    with torch.no_grad():
        for batch_X, _ in tqdm(data_loader, desc=desc, leave=False):
            batch_X = batch_X.to(device)
            
            if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                continue
                
            b_mean = batch_X.mean(dim=(2, 3), keepdim=True)
            b_std = batch_X.std(dim=(2, 3), keepdim=True)
            batch_X = (batch_X - b_mean) / (b_std + 1e-5)
            
            # Resize from 64x64 to 224x224 (DOFA's native Vision Transformer grid)
            batch_X = F.interpolate(batch_X, size=(224, 224), mode='bilinear', align_corners=False)
            
            features = model.forward_features(batch_X, wave_list=wave_list_um)
            
            if len(features.shape) == 3:
                img_embeddings = features[:, 0, :] 
            elif len(features.shape) == 4:
                img_embeddings = features.mean(dim=(2, 3))
            else:
                img_embeddings = features
                
            embeddings_list.append(img_embeddings.cpu().numpy())
            
    return np.vstack(embeddings_list)

def analyze_subset_domain_shift(subset_id, subset_classes, dofa_model, wave_list_um, device, umap_base_dir, batch_size=8):
    # Set the dataset directory targeting patch size 64
    dataset_dir = os.path.join(project_root, f"ds/H2Crop_tiles_ds_subset_{subset_id}/hyperspectral_taxonomy_3_pSize_64")
    
    print(f"\n{'='*60}")
    print(f"PROCESSING SUBSET {subset_id} | Classes: {subset_classes}")
    print(f"{'='*60}")
    
    if not os.path.exists(dataset_dir):
        print(f"Directory missing: {dataset_dir}. Skipping...")
        return

    print("--- Initializing DataLoaders ---")
    train_dataset = H2CropTileDataset(os.path.join(dataset_dir, "train"), subset_classes=subset_classes)
    test_dataset = H2CropTileDataset(os.path.join(dataset_dir, "test"), subset_classes=subset_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # REVERSED ORDER: Process Test embeddings first
    print("--- Extracting Test Embeddings ---")
    test_embeddings = extract_dofa_embeddings(test_loader, dofa_model, device, wave_list_um, f"Test S{subset_id}")

    # Process Train embeddings second
    print("--- Extracting Train Embeddings ---")
    train_embeddings = extract_dofa_embeddings(train_loader, dofa_model, device, wave_list_um, f"Train S{subset_id}")

    print("--- Fitting UMAP Projection ---")
    # Stack array with Test first, Train second to maintain index consistency
    all_embeddings = np.vstack((test_embeddings, train_embeddings))
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    all_2d = reducer.fit_transform(all_embeddings)
    
    # Split back based on the new array order (Test occupies the first indices)
    test_2d = all_2d[:len(test_embeddings)]
    train_2d = all_2d[len(test_embeddings):]

    print("--- Generating Latent Space Scatter Plot ---")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot Test first (drawn underneath)
    ax.scatter(test_2d[:, 0], test_2d[:, 1], alpha=0.3, s=20, label='Test Set Embeddings', c='#ff7f0e', marker='X')
    # Plot Train second (drawn on top)
    ax.scatter(train_2d[:, 0], train_2d[:, 1], alpha=0.8, s=15, label='Train Set Embeddings', c='#1f77b4', edgecolors='none')
    
    ax.set_title(f'DOFA Latent Space UMAP: Train vs Test Distribution (Subset {subset_id})', fontsize=16, fontweight='bold')
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=12, markerscale=2)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    # Save to the root-level shared directory
    plot_path = os.path.join(umap_base_dir, f"dofa_umap_distribution_subset_{subset_id}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Success! UMAP visualization saved to {plot_path}")

if __name__ == "__main__":
    # Define the 4 target subsets
    subsets = {
        # 1: [8, 11, 23, 56],
        2: [13, 30, 38, 53, 61],
        # 3: [1, 2, 10, 17],
        # 4: [29, 50, 64, 76]
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Global Execution Device: {device} ---")
    
    # Create the central UMAP image directory cleanly mapped to your project root
    umap_base_dir = os.path.join(project_root, "umap_img")
    os.makedirs(umap_base_dir, exist_ok=True)
    print(f"--- Saving outputs to central directory: {umap_base_dir} ---")

    # Load model once to save VRAM
    print("\n--- Loading DOFA Foundation Model (Global) ---")
    try:
        dofa_model = torch.hub.load('zhu-xlab/DOFA', 'vit_base_dofa', pretrained=True)
        dofa_model = dofa_model.to(device)
        dofa_model.eval()
    except Exception as e:
        print(f"Error loading DOFA. Did you run 'pip install timm==0.9.2'? Error: {e}")
        sys.exit(1)

    wave_list_um = get_valid_wavelengths()
    print(f"Parsed {len(wave_list_um)} valid bands for DOFA.")

    # Iterate through all configured subsets
    for subset_id, classes in subsets.items():
        analyze_subset_domain_shift(subset_id, classes, dofa_model, wave_list_um, device, umap_base_dir)