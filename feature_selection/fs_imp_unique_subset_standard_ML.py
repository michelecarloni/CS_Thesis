import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc

# Setup project root path
project_root = os.path.abspath('../')
if project_root not in sys.path:
    sys.path.append(project_root)

from H2Crop.data_structures import h2crop_taxonomy_dict

# ==========================================
# CONFIGURATION
# ==========================================
patch_size = 64
taxonomy = 3
num_splits = 3

dataset_base = "../ds"
ml_checkpoints_base = "/run/media/michele/T7/checkpoints/checkpoints_9_standard_ML_uniqueSub_split_same_tot_tiles"
scalers_path = "/run/media/michele/T7/scalers/scalers_unique_subset" 
out_base_dir = "../feature_importance_unique_subset_standard_ML_results"

modalities = ["hyperspectral", "multispectral"]
ml_models = ["logistic_regression", "linear_svm", "decision_tree", "random_forest"]

subset_classes = [8, 11, 23, 30, 38, 1, 2, 10, 29, 50]
unique_classes = [0] + sorted(subset_classes)

# Memory Control Parameters
max_test_pool_per_class = 5000  
max_permutation_samples = 2000  

# Load Taxonomy Names
taxonomy_key = f'Taxonomy_{taxonomy}'
class_names = {c: h2crop_taxonomy_dict.get(taxonomy_key, {}).get(c, f"Class {c}") for c in unique_classes}
class_names[0] = "Background"


def extract_synchronized_test_pool(split_id, target_classes, patch_size):
    print(f"    -> Scanning test files for Split {split_id}...")
    
    hyper_dir = os.path.join(dataset_base, "H2Crop_tiles_unique_subset", f"split_{split_id}", f"hyperspectral_taxonomy_3_pSize_{patch_size}", "test")
    multi_dir = os.path.join(dataset_base, "H2Crop_tiles_unique_subset", f"split_{split_id}", f"multispectral_taxonomy_3_pSize_{patch_size}", "test")
    
    hyper_files = set(os.path.basename(f) for f in glob.glob(os.path.join(hyper_dir, "*.npz")))
    multi_files = set(os.path.basename(f) for f in glob.glob(os.path.join(multi_dir, "*.npz")))
    common_files = sorted(list(hyper_files.intersection(multi_files)))
    
    if not common_files:
        print(f"       [Error] No common files found for Split {split_id}!")
        return None, None, None

    y_arrays = []
    pixels_per_file = []
    for f in common_files:
        with np.load(os.path.join(hyper_dir, f)) as data:
            y_flat = data['y'].flatten().astype(np.int32)
            y_arrays.append(y_flat)
            pixels_per_file.append(len(y_flat))
            
    y_global = np.concatenate(y_arrays)
    
    np.random.seed(42)
    selected_global_indices = []
    for cls in target_classes:
        cls_idx = np.where(y_global == cls)[0]
        if len(cls_idx) > 0:
            sampled = np.random.choice(cls_idx, size=min(max_test_pool_per_class, len(cls_idx)), replace=False)
            selected_global_indices.append(sampled)
            
    selected_global_indices = np.concatenate(selected_global_indices)
    selected_global_indices.sort()
    
    del y_global, y_arrays
    gc.collect()

    total_selected = len(selected_global_indices)
    print(f"    -> Loading exactly {total_selected} synchronized pixels into memory...")
    
    X_hyper_pool = np.zeros((total_selected, 218), dtype=np.float32)
    X_multi_pool = np.zeros((total_selected, 10), dtype=np.float32)
    y_pool = np.zeros(total_selected, dtype=np.int32)
    
    current_offset = 0
    extracted = 0
    sg_idx = 0
    
    for f, num_pixels in zip(common_files, pixels_per_file):
        file_start = current_offset
        file_end = current_offset + num_pixels
        
        local_indices = []
        while sg_idx < total_selected and selected_global_indices[sg_idx] < file_end:
            local_indices.append(selected_global_indices[sg_idx] - file_start)
            sg_idx += 1
            
        if local_indices:
            with np.load(os.path.join(hyper_dir, f)) as data_h:
                X_h = data_h['X'].reshape(218, -1).T.astype(np.float32)
                X_hyper_pool[extracted : extracted + len(local_indices)] = X_h[local_indices]
                y_pool[extracted : extracted + len(local_indices)] = data_h['y'].flatten().astype(np.int32)[local_indices]
                
            with np.load(os.path.join(multi_dir, f)) as data_m:
                X_m = data_m['X'].reshape(10, -1).T.astype(np.float32)
                X_multi_pool[extracted : extracted + len(local_indices)] = X_m[local_indices]
                
            extracted += len(local_indices)
            
        current_offset += num_pixels
        if sg_idx >= total_selected:
            break
            
    return X_hyper_pool, X_multi_pool, y_pool


if __name__ == "__main__":
    
    for split_id in range(1, num_splits + 1):
        print(f"\n{'='*70}")
        print(f"STARTING SPLIT {split_id}")
        print(f"{'='*70}")
        
        X_hyper_raw, X_multi_raw, y_test = extract_synchronized_test_pool(split_id, unique_classes, patch_size)
        if y_test is None:
            continue
            
        for mod in modalities:
            in_channels = 218 if mod == "hyperspectral" else 10
            X_mod_raw = X_hyper_raw if mod == "hyperspectral" else X_multi_raw
            
            print(f"\n  --- Evaluating {mod.upper()} ---")
            
            scaler_path = os.path.join(scalers_path, "unique_subset", mod, f"scaler_split_{split_id}_tax_{taxonomy}_pSize_{patch_size}.joblib")
            
            if not os.path.exists(scaler_path):
                print(f"      [Warning] Scaler missing for {mod} at {scaler_path}. Skipping.")
                continue
                
            scaler = joblib.load(scaler_path)
            X_test_scaled = scaler.transform(X_mod_raw)
            
            for ml_algo in ml_models:
                model_path = os.path.join(ml_checkpoints_base, "unique_subset", ml_algo, mod, f"{ml_algo}_split_{split_id}_tax_{taxonomy}_pSize_{patch_size}.joblib")
                if not os.path.exists(model_path):
                    continue
                    
                print(f"      >> Analyzing Model: {ml_algo.upper()}")
                model = joblib.load(model_path)
                
                out_dir = os.path.join(out_base_dir, mod, f"split_{split_id}", ml_algo)
                os.makedirs(out_dir, exist_ok=True)
                
                y_pred = model.predict(X_test_scaled)
                
                # Shifted to 0-based indexing to align perfectly with the LaTeX mapping script
                df_importance = pd.DataFrame({'Band': np.arange(0, in_channels)})
                
                for cls in unique_classes:
                    correct_indices = np.where((y_test == cls) & (y_pred == cls))[0]
                    
                    if len(correct_indices) == 0:
                        df_importance[class_names[cls]] = 0.0
                        continue
                        
                    np.random.seed(42)
                    sampled_indices = np.random.choice(
                        correct_indices, 
                        size=min(max_permutation_samples, len(correct_indices)), 
                        replace=False
                    )
                    
                    X_correct = X_test_scaled[sampled_indices]
                    y_correct = y_test[sampled_indices]
                    
                    importances = np.zeros(in_channels)
                    baseline_acc = 1.0 
                    
                    for b in tqdm(range(in_channels), desc=f"         Shuffling bands for {class_names[cls]}", leave=False):
                        X_shuffled = X_correct.copy()
                        np.random.shuffle(X_shuffled[:, b]) 
                        
                        y_pred_shuffled = model.predict(X_shuffled)
                        shuffled_acc = np.mean(y_pred_shuffled == cls)
                        importances[b] = baseline_acc - shuffled_acc
                        
                    df_importance[class_names[cls]] = importances
                    
                    df_class_sorted = pd.DataFrame({
                        'band': np.arange(0, in_channels),
                        'drop': importances
                    }).sort_values(by='drop', ascending=False)
                    
                    csv_class_path = os.path.join(out_dir, f"importance_class_{cls}_sorted.csv")
                    df_class_sorted.to_csv(csv_class_path, index=False)
                    
                    # Generate Histogram Plot instead of a Line Plot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    # Ensure positive values for visualization by clipping at 0 (noise can sometimes cause tiny negative drops)
                    positive_importances = np.clip(importances, a_min=0, a_max=None)
                    
                    # Create the bar chart
                    bars = ax.bar(df_importance['Band'], positive_importances, color='steelblue', edgecolor='black', linewidth=0.5)
                    
                    # Add a horizontal line at 0 for visual baseline
                    ax.axhline(y=0, color='black', linewidth=1)
                    
                    ax.set_title(f"Band Importance Histogram: {class_names[cls]}\n({ml_algo.replace('_', ' ').title()} | {mod.title()} | Split {split_id})", fontsize=14, pad=15, fontweight='bold')
                    ax.set_xlabel("Band Index", fontsize=12)
                    ax.set_ylabel("Drop in Accuracy", fontsize=12)
                    
                    # Format grid lines behind the bars
                    ax.set_axisbelow(True)
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Adjust X-axis ticks for readability depending on modality
                    if mod == "hyperspectral":
                        ax.set_xlim(-1, in_channels)
                        ax.set_xticks(np.arange(0, in_channels, 10))
                        plt.xticks(rotation=45, ha='right', fontsize=9)
                    else:
                        ax.set_xlim(-0.5, in_channels - 0.5)
                        ax.set_xticks(np.arange(0, in_channels))
                        plt.xticks(fontsize=10)
                        
                    # Highlight the top 2 most important bands with a different color (e.g., Red)
                    top_2_indices = df_class_sorted['band'].head(2).values
                    for b_idx in top_2_indices:
                        if 0 <= b_idx < len(bars):
                            bars[b_idx].set_color('firebrick')
                            bars[b_idx].set_edgecolor('black')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"importance_class_{cls}_histogram.png"), dpi=300)
                    plt.close(fig)

                csv_path = os.path.join(out_dir, "feature_importance_summary.csv")
                df_importance.to_csv(csv_path, index=False)
                
                del model, y_pred
                gc.collect()
            
            del X_test_scaled, scaler
            gc.collect()

        del X_hyper_raw, X_multi_raw, y_test
        gc.collect()

    print(f"\n--- Feature Importance Extraction Complete! Data safely saved to {out_base_dir} ---")