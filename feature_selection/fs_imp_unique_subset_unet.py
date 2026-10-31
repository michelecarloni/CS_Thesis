import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
import gc

# Setup project root path
project_root = os.path.abspath('../')
if project_root not in sys.path:
    sys.path.append(project_root)

from models.unet import UNet  
from H2Crop.data_structures import h2crop_taxonomy_dict

# ==========================================
# CONFIGURATION
# ==========================================
patch_size = 64
taxonomy = 3
num_splits = 3

dataset_base = "../ds"
unet_checkpoints_base = "/media/michele/T7/checkpoints/checkpoints_9_unet_pSize64_uniqueSub_NoGlobalSetTiles_ScalerYes_random_aug"
scalers_path = "/media/michele/T7/scalers/scalers_unique_subset" 
out_base_dir = "../feature_importance_unique_subset_unet_results"

modalities = ["hyperspectral", "multispectral"]
unet_encoders = ["resnet18"]

subset_classes = [8, 11, 23, 30, 38, 1, 2, 10, 29, 50]
unique_classes = [0] + sorted(subset_classes)
num_classes = len(unique_classes)

# Load Taxonomy Names
taxonomy_key = f'Taxonomy_{taxonomy}'
class_names = {c: h2crop_taxonomy_dict.get(taxonomy_key, {}).get(c, f"Class {c}") for c in unique_classes}
class_names[0] = "Background"

# Mapping logic for PyTorch
label_mapping = {raw_label: idx for idx, raw_label in enumerate(unique_classes)}
inverse_label_mapping = {idx: raw_label for raw_label, idx in label_mapping.items()}


def find_unet_model(base_dir, split_id, modality, use_best_model=True):
    """Recursively searches the base_dir to find the U-Net model for the given split and modality."""
    search_pattern = os.path.join(base_dir, '**', 'best_model.pth') if use_best_model else os.path.join(base_dir, '**', 'epoch_*.pth')
    all_models = glob.glob(search_pattern, recursive=True)
    if not use_best_model:
        all_models = sorted(all_models)
        
    for model_path in reversed(all_models): 
        if modality in model_path and f"split_{split_id}" in model_path:
            return model_path
    return None


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")
    
    for split_id in range(1, num_splits + 1):
        print(f"\n{'='*70}")
        print(f"STARTING SPLIT {split_id}")
        print(f"{'='*70}")
        
        for mod in modalities:
            in_channels = 218 if mod == "hyperspectral" else 10
            
            print(f"\n  --- Evaluating {mod.upper()} ---")
            
            # 1. Load Scaler to GPU
            scaler_path = os.path.join(scalers_path, "unique_subset", mod, f"scaler_split_{split_id}_tax_{taxonomy}_pSize_{patch_size}.joblib")
            if not os.path.exists(scaler_path):
                print(f"      [Warning] Scaler missing at {scaler_path}. Skipping.")
                continue
                
            scaler = joblib.load(scaler_path)
            scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device).view(1, in_channels, 1, 1)
            scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).view(1, in_channels, 1, 1)
            
            for encoder in unet_encoders:
                print(f"      >> Analyzing Encoder: {encoder.upper()}")
                
                # 2. Load U-Net Model
                model_path = find_unet_model(unet_checkpoints_base, split_id, mod, use_best_model=True)
                if not model_path:
                    print(f"      [Warning] Model missing for {mod} split {split_id}. Skipping.")
                    continue
                
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                
                # Infer depth dynamically
                block_indices = [int(k.split("model.decoder.blocks.")[1].split(".")[0]) for k in state_dict.keys() if "model.decoder.blocks." in k]
                inferred_depth = max(block_indices) + 1 if block_indices else 4
                
                model = UNet(in_channels=in_channels, num_classes=num_classes, encoder_name=encoder, encoder_depth=inferred_depth).to(device)
                model.load_state_dict(state_dict)
                model.eval()
                
                out_dir = os.path.join(out_base_dir, mod, f"split_{split_id}", f"unet_{encoder}")
                os.makedirs(out_dir, exist_ok=True)
                
                # 3. Setup Test Files and Trackers
                test_dir = os.path.join(dataset_base, "H2Crop_tiles_unique_subset", f"split_{split_id}", f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}", "test")
                test_files = glob.glob(os.path.join(test_dir, "*.npz"))
                
                if not test_files:
                    print(f"      [Warning] No test files found in {test_dir}. Skipping.")
                    continue
                
                # Tracking dictionaries
                baseline_correct_counts = {c: 0 for c in unique_classes}
                lost_pixels_per_band = {c: np.zeros(in_channels, dtype=np.int64) for c in unique_classes}
                
                # 4. Process Test Set Tile-by-Tile on GPU
                file_loop = tqdm(test_files, desc="      Computing Feature Ablation")
                
                with torch.no_grad():
                    for f in file_loop:
                        with np.load(f) as data:
                            X_raw = data['X']
                            y_raw = data['y']
                        
                        # Safely map true labels to PyTorch indices (defaulting unknown to 0)
                        y_mapped = np.vectorize(lambda x: label_mapping.get(x, 0))(y_raw)
                        y_true = torch.tensor(y_mapped, dtype=torch.long, device=device)
                        
                        X_tensor = torch.tensor(X_raw, dtype=torch.float32, device=device).unsqueeze(0)
                        
                        # Apply Standardization
                        X_scaled = torch.nan_to_num(X_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                        X_scaled = (X_scaled - scaler_mean) / scaler_scale
                        
                        # Baseline Prediction
                        out_baseline = model(X_scaled)
                        pred_baseline = torch.max(out_baseline, 1)[1].squeeze(0)
                        
                        # Track originally correct pixels
                        mask_correct = (pred_baseline == y_true)
                        
                        # Get the mapped indices present in the current tile
                        present_mapped_classes = torch.unique(y_true).cpu().numpy()
                        
                        for c_idx in present_mapped_classes:
                            c_raw = inverse_label_mapping[c_idx]
                            baseline_correct_counts[c_raw] += (mask_correct & (y_true == c_idx)).sum().item()
                            
                        # Ablation Loop over all bands
                        for b in range(in_channels):
                            X_ablated = X_scaled.clone()
                            # 0.0 mathematically represents the exact global average for standard scaled data
                            X_ablated[:, b, :, :] = 0.0 
                            
                            out_ablated = model(X_ablated)
                            pred_ablated = torch.max(out_ablated, 1)[1].squeeze(0)
                            
                            # For each class, how many originally correct pixels became incorrect?
                            for c_idx in present_mapped_classes:
                                c_raw = inverse_label_mapping[c_idx]
                                originally_correct_for_c = mask_correct & (y_true == c_idx)
                                
                                # Pixels that were right before, but are now wrong
                                newly_wrong_for_c = originally_correct_for_c & (pred_ablated != c_idx)
                                
                                lost_pixels_per_band[c_raw][b] += newly_wrong_for_c.sum().item()

                # 5. Calculate Final Importance & Generate Outputs
                df_importance = pd.DataFrame({'Band': np.arange(1, in_channels + 1)})
                
                for cls in unique_classes:
                    total_correct = baseline_correct_counts[cls]
                    
                    if total_correct == 0:
                        importances = np.zeros(in_channels)
                    else:
                        # Drop = Lost Pixels / Total Originally Correct Pixels
                        importances = lost_pixels_per_band[cls] / total_correct
                        
                    df_importance[class_names[cls]] = importances
                    
                    # Sort and save class-specific CSV
                    df_class_sorted = pd.DataFrame({
                        'band': np.arange(1, in_channels + 1),
                        'drop': importances
                    }).sort_values(by='drop', ascending=False)
                    
                    csv_class_path = os.path.join(out_dir, f"importance_class_{cls}_sorted.csv")
                    df_class_sorted.to_csv(csv_class_path, index=False)
                    
                    # Generate Plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(df_importance['Band'], importances, color='darkred', linewidth=1.5)
                    ax.fill_between(df_importance['Band'], importances, alpha=0.3, color='salmon')
                    
                    ax.set_title(f"Band Importance (Zero-Ablation) for {class_names[cls]}\n(U-Net {encoder.title()} | {mod.title()} | Split {split_id})", fontsize=12, pad=10)
                    ax.set_xlabel("Band Index", fontsize=10)
                    ax.set_ylabel("Drop in Accuracy (Importance)", fontsize=10)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"importance_class_{cls}.png"), dpi=200)
                    plt.close(fig)

                # Save Summary CSV containing all classes side-by-side
                csv_path = os.path.join(out_dir, "feature_importance_summary.csv")
                df_importance.to_csv(csv_path, index=False)
                
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"\n--- U-Net Feature Importance Extraction Complete! Data safely saved to {out_base_dir} ---")