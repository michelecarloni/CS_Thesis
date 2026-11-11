import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import gc

# Setup project root path
project_root = os.path.abspath('../../')
if project_root not in sys.path:
    sys.path.append(project_root)

from models.unet import UNet  
from H2Crop.data_structures import h2crop_taxonomy_dict, h2crop_multispectral_band_mapping, h2crop_band_mapping_no_water_absorption_bands

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
split_id = 1
patch_size = 64
taxonomy = 3

dataset_base = "../../ds"
ml_checkpoints_base = "/run/media/michele/T7/checkpoints/checkpoints_9_standard_ML_uniqueSub_split_same_tot_tiles/unique_subset"
unet_checkpoints_base = "/run/media/michele/T7/checkpoints/checkpoints_9_unet_pSize64_uniqueSub_NoGlobalSetTiles_ScalerYes_random_aug/unique_subset"
scalers_path = "/run/media/michele/T7/scalers/scalers_unique_subset/unique_subset" 

out_dir = "../../final_analysis_results/pairwise_separability_results"
os.makedirs(out_dir, exist_ok=True)

models_to_evaluate = {
    "decision_tree": "ml",
    "random_forest": "ml",
    "logistic_regression": "ml",
    "linear_svm": "ml",
    "unet_enc_resnet18": "unet"
}

subset_classes = [8, 11, 23, 30, 38, 1, 2, 10, 29, 50]
unique_classes = [0] + sorted(subset_classes)
num_classes = len(unique_classes)

taxonomy_key = f'Taxonomy_{taxonomy}'
class_names = {c: h2crop_taxonomy_dict.get(taxonomy_key, {}).get(c, f"Class {c}") for c in unique_classes}
class_names[0] = "Background"

label_mapping = {raw_label: idx for idx, raw_label in enumerate(unique_classes)}
inverse_mapping = {idx: raw_label for raw_label, idx in label_mapping.items()}

max_test_pool_per_class = 5000  

def get_common_test_files(split_id, patch_size):
    hyper_dir = os.path.join(dataset_base, "H2Crop_tiles_unique_subset", f"split_{split_id}", f"hyperspectral_taxonomy_3_pSize_{patch_size}", "test")
    multi_dir = os.path.join(dataset_base, "H2Crop_tiles_unique_subset", f"split_{split_id}", f"multispectral_taxonomy_3_pSize_{patch_size}", "test")
    
    hyper_files = set(os.path.basename(f) for f in glob.glob(os.path.join(hyper_dir, "*.npz")))
    multi_files = set(os.path.basename(f) for f in glob.glob(os.path.join(multi_dir, "*.npz")))
    common_files = sorted(list(hyper_files.intersection(multi_files)))
    return common_files, hyper_dir, multi_dir


def extract_synchronized_test_pool(split_id, target_classes, patch_size):
    print(f"\n[+] Synchronizing and Loading 1D Test Pool for Standard ML (Split {split_id})...")
    common_files, hyper_dir, multi_dir = get_common_test_files(split_id, patch_size)
    if not common_files: raise ValueError("No common test files found!")

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
                X_hyper_pool[extracted : extracted + len(local_indices)] = data_h['X'].reshape(218, -1).T.astype(np.float32)[local_indices]
                y_pool[extracted : extracted + len(local_indices)] = data_h['y'].flatten().astype(np.int32)[local_indices]
            with np.load(os.path.join(multi_dir, f)) as data_m:
                X_multi_pool[extracted : extracted + len(local_indices)] = data_m['X'].reshape(10, -1).T.astype(np.float32)[local_indices]
            extracted += len(local_indices)
        current_offset += num_pixels
        if sg_idx >= total_selected: break
            
    return X_hyper_pool, X_multi_pool, y_pool


def calculate_ml_pairwise_band_importance(model, X_scaled, y_true, cls_a, cls_b, in_channels):
    """Permutation importance on flattened 1D arrays for Standard ML."""
    pair_mask = (y_true == cls_a) | (y_true == cls_b)
    X_pair = X_scaled[pair_mask]
    y_pair = y_true[pair_mask]
    if len(y_pair) == 0: return np.zeros(in_channels)
        
    y_pred_baseline = model.predict(X_pair)
    baseline_acc = np.mean(y_pred_baseline == y_pair)
    
    importances = np.zeros(in_channels)
    for b in range(in_channels):
        X_ablated = X_pair.copy()
        np.random.shuffle(X_ablated[:, b])
        y_pred_ablated = model.predict(X_ablated)
        importances[b] = baseline_acc - np.mean(y_pred_ablated == y_pair)
        
    return importances


def calculate_unet_pairwise_band_importance(model, common_files, target_dir, scaler_mean, scaler_scale, cls_a, cls_b, in_channels, device):
    """Zero-Ablation importance on full 64x64 spatial tiles for U-Net."""
    importances = np.zeros(in_channels)
    original_correct = 0
    lost_pixels = np.zeros(in_channels)
    
    # Evaluate a subset of 100 tiles containing the classes to keep processing fast
    sample_files = common_files[:100]
    
    with torch.no_grad():
        for f in sample_files:
            with np.load(os.path.join(target_dir, f)) as data:
                X_raw = data['X']
                y_raw = data['y']
                
            pair_mask = (y_raw == cls_a) | (y_raw == cls_b)
            if not np.any(pair_mask): continue
                
            y_tensor = torch.tensor(np.vectorize(label_mapping.get)(y_raw), dtype=torch.long, device=device)
            pair_mask_tensor = torch.tensor(pair_mask, dtype=torch.bool, device=device)
            
            X_tensor = torch.tensor(X_raw, dtype=torch.float32, device=device).unsqueeze(0)
            X_scaled = torch.nan_to_num(X_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = (X_scaled - scaler_mean) / scaler_scale
            
            # Baseline
            out_baseline = model(X_scaled)
            pred_baseline = torch.max(out_baseline, 1)[1].squeeze(0)
            correct_mask = pair_mask_tensor & (pred_baseline == y_tensor)
            original_correct += correct_mask.sum().item()
            
            # Ablate each band spatially
            for b in range(in_channels):
                X_ablated = X_scaled.clone()
                X_ablated[:, b, :, :] = 0.0
                
                out_ablated = model(X_ablated)
                pred_ablated = torch.max(out_ablated, 1)[1].squeeze(0)
                
                newly_wrong = correct_mask & (pred_ablated != y_tensor)
                lost_pixels[b] += newly_wrong.sum().item()
                
    if original_correct == 0: return np.zeros(in_channels)
    return lost_pixels / original_correct


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Global extractions
    common_files, hyper_dir, multi_dir = get_common_test_files(split_id, patch_size)
    X_hyper_raw, X_multi_raw, y_test_pool = extract_synchronized_test_pool(split_id, unique_classes, patch_size)
    
    for model_name, model_type in models_to_evaluate.items():
        print(f"\n{'='*70}\nANALYZING PAIRWISE SEPARABILITY: {model_name.upper()}\n{'='*70}")
        
        preds_cm = {}
        y_true_cm = {}
        loaded_models = {}
        unet_scalers = {}
        
        # 1. Run Inference for Both Modalities
        for mod in ["multispectral", "hyperspectral"]:
            in_channels = 10 if mod == "multispectral" else 218
            scaler_path = os.path.join(scalers_path, mod, f"scaler_split_{split_id}_tax_{taxonomy}_pSize_{patch_size}.joblib")
            scaler = joblib.load(scaler_path)
            
            if model_type == "ml":
                X_mod_raw = X_multi_raw if mod == "multispectral" else X_hyper_raw
                X_scaled = scaler.transform(X_mod_raw)
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                
                model_path = os.path.join(ml_checkpoints_base, model_name, mod, f"{model_name}_split_{split_id}_tax_{taxonomy}_pSize_{patch_size}.joblib")
                model = joblib.load(model_path)
                loaded_models[mod] = (model, X_scaled)
                
                preds_cm[mod] = model.predict(X_scaled)
                y_true_cm[mod] = y_test_pool
                
            elif model_type == "unet":
                scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device).view(1, in_channels, 1, 1)
                scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).view(1, in_channels, 1, 1)
                unet_scalers[mod] = (scaler_mean, scaler_scale)
                
                model_path = os.path.join(unet_checkpoints_base, model_name, mod, f"split_{split_id}_pSize_{patch_size}", "best_model.pth")
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                
                depth = max([int(k.split("model.decoder.blocks.")[1].split(".")[0]) for k in state_dict.keys() if "model.decoder.blocks." in k]) + 1
                encoder = "resnet18" if "resnet18" in model_name else "resnet34"
                
                model = UNet(in_channels=in_channels, num_classes=num_classes, encoder_name=encoder, encoder_depth=depth).to(device)
                model.load_state_dict(state_dict)
                model.eval()
                loaded_models[mod] = model
                
                # Spatial inference for Confusion Matrix
                print(f"      Running Spatial Inference on test tiles for {mod}...")
                target_dir = multi_dir if mod == "multispectral" else hyper_dir
                y_pred_mod = []
                y_true_mod = []
                
                with torch.no_grad():
                    for f in tqdm(common_files, desc="      Inference", leave=False):
                        with np.load(os.path.join(target_dir, f)) as data:
                            X_raw = data['X']
                            y_raw = data['y']
                        X_tensor = torch.tensor(X_raw, dtype=torch.float32, device=device).unsqueeze(0)
                        X_scaled = torch.nan_to_num(X_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                        X_scaled = (X_scaled - scaler_mean) / scaler_scale
                        
                        out = model(X_scaled)
                        pred_mapped = torch.max(out, 1)[1].squeeze(0).cpu().numpy()
                        y_pred_mod.extend(np.vectorize(inverse_mapping.get)(pred_mapped.flatten()))
                        y_true_mod.extend(y_raw.flatten())
                        
                preds_cm[mod] = np.array(y_pred_mod)
                y_true_cm[mod] = np.array(y_true_mod)

        # 2. Compute Confusion Matrices
        cm_multi = confusion_matrix(y_true_cm["multispectral"], preds_cm["multispectral"], labels=unique_classes)
        cm_hyper = confusion_matrix(y_true_cm["hyperspectral"], preds_cm["hyperspectral"], labels=unique_classes)
        
        # 3. Analyze Pairwise Separation & Deltas
        pairwise_records = []
        for i, cls_a in enumerate(unique_classes):
            for j, cls_b in enumerate(unique_classes):
                if i >= j: continue 
                
                errors_multi = cm_multi[i, j] + cm_multi[j, i]
                errors_hyper = cm_hyper[i, j] + cm_hyper[j, i]
                
                total_samples_pair = np.sum(y_true_cm["multispectral"] == cls_a) + np.sum(y_true_cm["multispectral"] == cls_b)
                if total_samples_pair == 0: continue
                
                confusion_rate_multi = errors_multi / total_samples_pair
                confusion_rate_hyper = errors_hyper / total_samples_pair
                delta = confusion_rate_multi - confusion_rate_hyper
                
                pairwise_records.append({
                    "Class A": class_names[cls_a],
                    "Class B": class_names[cls_b],
                    "Class A ID": cls_a,
                    "Class B ID": cls_b,
                    "Multi Confusion": confusion_rate_multi,
                    "Hyper Confusion": confusion_rate_hyper,
                    "Hyper Benefit Delta": delta
                })
                
        df_pairs = pd.DataFrame(pairwise_records)
        df_crops_only = df_pairs[(df_pairs["Class A ID"] != 0) & (df_pairs["Class B ID"] != 0)]
        
        df_hyper_req = df_crops_only.sort_values(by="Hyper Benefit Delta", ascending=False).head(5)
        df_multi_sep = df_crops_only.sort_values(by="Multi Confusion", ascending=True).head(5)
        
        # 4. Extract Top Bands
        print(f"\n[+] Extracting Top Bands for Pairwise Ablation...")
        multi_sep_results = []
        for _, row in tqdm(df_multi_sep.iterrows(), total=len(df_multi_sep), desc="      Multispectral"):
            if model_type == "ml":
                importances = calculate_ml_pairwise_band_importance(loaded_models["multispectral"][0], loaded_models["multispectral"][1], y_test_pool, row["Class A ID"], row["Class B ID"], 10)
            else:
                importances = calculate_unet_pairwise_band_importance(loaded_models["multispectral"], common_files, multi_dir, unet_scalers["multispectral"][0], unet_scalers["multispectral"][1], row["Class A ID"], row["Class B ID"], 10, device)
                
            top_2_idx = np.argsort(importances)[-2:][::-1]
            top_bands = [f"{h2crop_multispectral_band_mapping[b]['band']} ({h2crop_multispectral_band_mapping[b]['wavelength_nm']}nm)" for b in top_2_idx]
            multi_sep_results.append({
                "Pair": f"{row['Class A']} vs {row['Class B']}", "Confusion Rate": f"{row['Multi Confusion']:.2%}", "Top Band 1": top_bands[0], "Top Band 2": top_bands[1]
            })
            
        hyper_req_results = []
        for _, row in tqdm(df_hyper_req.iterrows(), total=len(df_hyper_req), desc="      Hyperspectral"):
            if model_type == "ml":
                importances = calculate_ml_pairwise_band_importance(loaded_models["hyperspectral"][0], loaded_models["hyperspectral"][1], y_test_pool, row["Class A ID"], row["Class B ID"], 218)
            else:
                importances = calculate_unet_pairwise_band_importance(loaded_models["hyperspectral"], common_files, hyper_dir, unet_scalers["hyperspectral"][0], unet_scalers["hyperspectral"][1], row["Class A ID"], row["Class B ID"], 218, device)
                
            top_2_idx = np.argsort(importances)[-2:][::-1]
            top_bands = []
            for b in top_2_idx:
                val = h2crop_band_mapping_no_water_absorption_bands.get(b, b)
                top_bands.append(f"{val:.1f}nm" if isinstance(val, (int, float)) else f"Idx:{b}")
                
            hyper_req_results.append({
                "Pair": f"{row['Class A']} vs {row['Class B']}", "Confusion Rate Drop": f"{row['Hyper Benefit Delta']:.2%}", "Top Band 1": top_bands[0], "Top Band 2": top_bands[1]
            })
            
        # 5. Save Reports
        df_pairs.to_csv(os.path.join(out_dir, f"{model_name}_full_pairwise_analysis.csv"), index=False)
        report_path = os.path.join(out_dir, f"{model_name}_separability_report.txt")
        with open(report_path, "w") as f:
            f.write(f"PAIRWISE SEPARABILITY REPORT: {model_name.upper()}\n")
            f.write("="*70 + "\n\n")
            f.write("--- 1. STRICTLY REQUIRE HYPERSPECTRAL (Highest Multi-to-Hyper Confusion Drop) ---\n")
            for _, row in pd.DataFrame(hyper_req_results).iterrows():
                f.write(f"{row['Pair']}:\n   Hyperspectral Improvement (Delta): {row['Confusion Rate Drop']}\n   Top Hyperspectral Band 1:          {row['Top Band 1']}\n   Top Hyperspectral Band 2:          {row['Top Band 2']}\n\n")
            f.write("\n--- 2. EASILY SEPARATED BY MULTISPECTRAL ALONE ---\n")
            for _, row in pd.DataFrame(multi_sep_results).iterrows():
                f.write(f"{row['Pair']}:\n   Sentinel-2 Confusion Rate: {row['Confusion Rate']}\n   Top Sentinel-2 Band 1:     {row['Top Band 1']}\n   Top Sentinel-2 Band 2:     {row['Top Band 2']}\n\n")
        print(f"    Saved report for {model_name} to {out_dir}")

    print("\n--- Pairwise Separability Analysis Complete! ---")