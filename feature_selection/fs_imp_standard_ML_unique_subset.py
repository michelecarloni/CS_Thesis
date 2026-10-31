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
ml_checkpoints_base = "/media/michele/T7/checkpoints/checkpoints_9_standard_ML_uniqueSub_split_same_tot_tiles"
scalers_path = "/media/michele/T7/scalers/scalers_unique_subset" 
out_base_dir = "../feature_importance_unique_subset_standard_ML_results"

modalities = ["hyperspectral", "multispectral"]
ml_models = ["logistic_regression", "linear_svm", "decision_tree", "random_forest"]

subset_classes = [8, 11, 23, 30, 38, 1, 2, 10, 29, 50]
unique_classes = [0] + sorted(subset_classes)

# Memory Control Parameters
# 1. How many raw pixels per class to pull into RAM initially (safeguard pool)
max_test_pool_per_class = 5000  
# 2. How many CORRECTLY PREDICTED pixels to use for the permutation (to calculate the actual drop in accuracy)
max_permutation_samples = 2000  

# Load Taxonomy Names
taxonomy_key = f'Taxonomy_{taxonomy}'
class_names = {c: h2crop_taxonomy_dict.get(taxonomy_key, {}).get(c, f"Class {c}") for c in unique_classes}
class_names[0] = "Background"


def extract_synchronized_test_pool(split_id, target_classes, patch_size):
    """
    Two-pass memory-safe extraction. 
    Finds common geographical files, maps the location of all classes, 
    and extracts a strict maximum number of pixels synchronously for both modalities.
    """
    print(f"    -> Scanning test files for Split {split_id}...")
    
    hyper_dir = os.path.join(dataset_base, "H2Crop_tiles_unique_subset", f"split_{split_id}", f"hyperspectral_taxonomy_3_pSize_{patch_size}", "test")
    multi_dir = os.path.join(dataset_base, "H2Crop_tiles_unique_subset", f"split_{split_id}", f"multispectral_taxonomy_3_pSize_{patch_size}", "test")
    
    hyper_files = set(os.path.basename(f) for f in glob.glob(os.path.join(hyper_dir, "*.npz")))
    multi_files = set(os.path.basename(f) for f in glob.glob(os.path.join(multi_dir, "*.npz")))
    common_files = sorted(list(hyper_files.intersection(multi_files)))
    
    if not common_files:
        print(f"       [Error] No common files found for Split {split_id}!")
        return None, None, None

    # Pass 1: Global count based on the Y labels (identical for both)
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

    # Pass 2: Exact Extraction
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
        
        # 1. LOAD SYNCHRONIZED POOL ONCE PER SPLIT
        X_hyper_raw, X_multi_raw, y_test = extract_synchronized_test_pool(split_id, unique_classes, patch_size)
        if y_test is None:
            continue
            
        for mod in modalities:
            in_channels = 218 if mod == "hyperspectral" else 10
            X_mod_raw = X_hyper_raw if mod == "hyperspectral" else X_multi_raw
            
            print(f"\n  --- Evaluating {mod.upper()} ---")
            
            # UPDATED: Pulling the scaler directly from your dedicated external scalers path
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
                
                # Predict on the pool
                y_pred = model.predict(X_test_scaled)
                
                df_importance = pd.DataFrame({'Band': np.arange(1, in_channels + 1)})
                
                for cls in unique_classes:
                    # Find indices where model was CORRECT
                    correct_indices = np.where((y_test == cls) & (y_pred == cls))[0]
                    
                    if len(correct_indices) == 0:
                        df_importance[class_names[cls]] = 0.0
                        continue
                        
                    # Sample correctly predicted pixels for the permutation test
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
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(df_importance['Band'], importances, color='darkred', linewidth=1.5)
                    ax.fill_between(df_importance['Band'], importances, alpha=0.3, color='salmon')
                    
                    ax.set_title(f"Band Importance for {class_names[cls]}\n({ml_algo.replace('_', ' ').title()} | {mod.title()} | Split {split_id})", fontsize=12, pad=10)
                    ax.set_xlabel("Band Index", fontsize=10)
                    ax.set_ylabel("Drop in Accuracy (Importance)", fontsize=10)
                    ax.grid(True, linestyle='--', alpha=0.6)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, f"importance_class_{cls}.png"), dpi=200)
                    plt.close(fig)

                # Save CSV for this algorithm
                csv_path = os.path.join(out_dir, "feature_importance_summary.csv")
                df_importance.to_csv(csv_path, index=False)
                
                # PURGE THE MODEL
                del model, y_pred
                gc.collect()
            
            # PURGE SCALED DATA
            del X_test_scaled, scaler
            gc.collect()

        # PURGE THE SPLIT'S MASTER RAW POOL
        del X_hyper_raw, X_multi_raw, y_test
        gc.collect()

    print(f"\n--- Feature Importance Extraction Complete! Data safely saved to {out_base_dir} ---")