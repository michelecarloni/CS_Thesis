import os
import sys
import glob
import joblib
import numpy as np
import pandas as pd
import torch
import itertools
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import gc

# Setup project root path
project_root = os.path.abspath('../../')
if project_root not in sys.path:
    sys.path.append(project_root)

from models.unet import UNet  
from H2Crop.data_structures import h2crop_taxonomy_dict, h2crop_band_mapping_no_water_absorption_bands

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

out_dir = "../../final_analysis_results/empirical_vs_theoretical_separability"
os.makedirs(out_dir, exist_ok=True)

models_to_evaluate = {
    "decision_tree": "ml",
    "random_forest": "ml",
    "logistic_regression": "ml",
    "linear_svm": "ml",
    "unet_enc_resnet18": "unet"
}

# 10 pure crop classes (excluding 0/Background) -> 45 unique pairs
subset_classes = [1, 2, 8, 10, 11, 23, 29, 30, 38, 50]
subset_classes.sort()
unique_classes = [0] + subset_classes
num_classes = len(unique_classes)

taxonomy_key = f'Taxonomy_{taxonomy}'
class_names = {c: h2crop_taxonomy_dict.get(taxonomy_key, {}).get(c, f"Class {c}") for c in subset_classes}
class_names[0] = "Background"

label_mapping = {raw_label: idx for idx, raw_label in enumerate(unique_classes)}
inverse_mapping = {idx: raw_label for raw_label, idx in label_mapping.items()}

max_test_pool_per_class = 5000  

# ==========================================
# MACRO-GROUP HARDWARE MAPPING
# ==========================================
def get_macro_group(wavelength_nm):
    if wavelength_nm < 520: return "Blue"
    elif wavelength_nm < 600: return "Green"
    elif wavelength_nm < 680: return "Red"
    elif wavelength_nm < 720: return "Lower Red Edge"
    elif wavelength_nm < 760: return "Upper Red Edge"
    elif wavelength_nm < 950: return "NIR-1"
    elif wavelength_nm < 1400: return "NIR-2"
    elif wavelength_nm < 1800: return "SWIR-1"
    else: return "SWIR-2"

# ==========================================
# DATA LOADING & ABLATION FUNCTIONS
# ==========================================
def get_common_test_files(split_id, patch_size):
    hyper_dir = os.path.join(dataset_base, "H2Crop_tiles_unique_subset", f"split_{split_id}", f"hyperspectral_taxonomy_3_pSize_{patch_size}", "test")
    multi_dir = os.path.join(dataset_base, "H2Crop_tiles_unique_subset", f"split_{split_id}", f"multispectral_taxonomy_3_pSize_{patch_size}", "test")
    hyper_files = set(os.path.basename(f) for f in glob.glob(os.path.join(hyper_dir, "*.npz")))
    multi_files = set(os.path.basename(f) for f in glob.glob(os.path.join(multi_dir, "*.npz")))
    common_files = sorted(list(hyper_files.intersection(multi_files)))
    return common_files, hyper_dir, multi_dir

def extract_synchronized_test_pool(split_id, target_classes, patch_size):
    common_files, hyper_dir, multi_dir = get_common_test_files(split_id, patch_size)
    y_arrays, pixels_per_file = [], []
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

def calculate_ml_pairwise_band_importance(model, X_scaled, y_true, cls_a, cls_b):
    pair_mask = (y_true == cls_a) | (y_true == cls_b)
    X_pair = X_scaled[pair_mask]
    y_pair = y_true[pair_mask]
    if len(y_pair) == 0: return np.zeros(218)
        
    y_pred_baseline = model.predict(X_pair)
    baseline_acc = np.mean(y_pred_baseline == y_pair)
    
    importances = np.zeros(218)
    for b in range(218):
        X_ablated = X_pair.copy()
        np.random.shuffle(X_ablated[:, b])
        y_pred_ablated = model.predict(X_ablated)
        importances[b] = baseline_acc - np.mean(y_pred_ablated == y_pair)
        
    return importances

def calculate_unet_pairwise_band_importance(model, common_files, target_dir, scaler_mean, scaler_scale, cls_a, cls_b, device):
    importances = np.zeros(218)
    original_correct = 0
    lost_pixels = np.zeros(218)
    sample_files = common_files[:100] 
    
    with torch.no_grad():
        for f in sample_files:
            with np.load(os.path.join(target_dir, f)) as data:
                X_raw, y_raw = data['X'], data['y']
                
            pair_mask = (y_raw == cls_a) | (y_raw == cls_b)
            if not np.any(pair_mask): continue
                
            y_tensor = torch.tensor(np.vectorize(label_mapping.get)(y_raw), dtype=torch.long, device=device)
            pair_mask_tensor = torch.tensor(pair_mask, dtype=torch.bool, device=device)
            
            X_tensor = torch.tensor(X_raw, dtype=torch.float32, device=device).unsqueeze(0)
            X_scaled = torch.nan_to_num(X_tensor, nan=0.0, posinf=0.0, neginf=0.0)
            X_scaled = (X_scaled - scaler_mean) / scaler_scale
            
            out_baseline = model(X_scaled)
            pred_baseline = torch.max(out_baseline, 1)[1].squeeze(0)
            correct_mask = pair_mask_tensor & (pred_baseline == y_tensor)
            original_correct += correct_mask.sum().item()
            
            for b in range(218):
                X_ablated = X_scaled.clone()
                X_ablated[:, b, :, :] = 0.0
                out_ablated = model(X_ablated)
                pred_ablated = torch.max(out_ablated, 1)[1].squeeze(0)
                newly_wrong = correct_mask & (pred_ablated != y_tensor)
                lost_pixels[b] += newly_wrong.sum().item()
                
    if original_correct == 0: return np.zeros(218)
    return lost_pixels / original_correct

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    common_files, hyper_dir, multi_dir = get_common_test_files(split_id, patch_size)
    X_hyper_raw, X_multi_raw, y_test_pool = extract_synchronized_test_pool(split_id, unique_classes, patch_size)
    
    for model_name, model_type in models_to_evaluate.items():
        print(f"\n{'='*70}\nEMPIRICAL VS THEORETICAL ANALYSIS: {model_name.upper()}\n{'='*70}")
        
        preds_cm = {}
        y_true_cm = {}
        
        # 1. Run Empirical Inference for Both Modalities
        for mod in ["multispectral", "hyperspectral"]:
            in_channels = 10 if mod == "multispectral" else 218
            scaler_path = os.path.join(scalers_path, mod, f"scaler_split_{split_id}_tax_{taxonomy}_pSize_{patch_size}.joblib")
            scaler = joblib.load(scaler_path)
            
            if model_type == "ml":
                X_mod_raw = X_multi_raw if mod == "multispectral" else X_hyper_raw
                X_scaled = scaler.transform(X_mod_raw)
                X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
                
                model_path = os.path.join(ml_checkpoints_base, model_name, mod, f"{model_name}_split_{split_id}_tax_{taxonomy}_pSize_{patch_size}.joblib")
                model_inst = joblib.load(model_path)
                
                preds_cm[mod] = model_inst.predict(X_scaled)
                y_true_cm[mod] = y_test_pool
                
                if mod == "hyperspectral":
                    hs_model = model_inst
                    hs_X_scaled = X_scaled
                    
            else:
                scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device).view(1, in_channels, 1, 1)
                scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).view(1, in_channels, 1, 1)
                
                model_path = os.path.join(unet_checkpoints_base, model_name, mod, f"split_{split_id}_pSize_{patch_size}", "best_model.pth")
                state_dict = torch.load(model_path, map_location=device, weights_only=True)
                depth = max([int(k.split("model.decoder.blocks.")[1].split(".")[0]) for k in state_dict.keys() if "model.decoder.blocks." in k]) + 1
                encoder = "resnet18" if "resnet18" in model_name else "resnet34"
                
                model_inst = UNet(in_channels=in_channels, num_classes=num_classes, encoder_name=encoder, encoder_depth=depth).to(device)
                model_inst.load_state_dict(state_dict)
                model_inst.eval()
                
                if mod == "hyperspectral":
                    hs_model = model_inst
                    hs_scaler_mean = scaler_mean
                    hs_scaler_scale = scaler_scale
                
                print(f"   Running Spatial Inference for {mod}...")
                target_dir = multi_dir if mod == "multispectral" else hyper_dir
                y_pred_mod, y_true_mod = [], []
                
                with torch.no_grad():
                    for f in tqdm(common_files, desc=f"   {mod.title()}", leave=False):
                        with np.load(os.path.join(target_dir, f)) as data:
                            X_raw, y_raw = data['X'], data['y']
                        X_tensor = torch.tensor(X_raw, dtype=torch.float32, device=device).unsqueeze(0)
                        X_scaled = torch.nan_to_num(X_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                        X_scaled = (X_scaled - scaler_mean) / scaler_scale
                        out = model_inst(X_scaled)
                        pred_mapped = torch.max(out, 1)[1].squeeze(0).cpu().numpy()
                        y_pred_mod.extend(np.vectorize(inverse_mapping.get)(pred_mapped.flatten()))
                        y_true_mod.extend(y_raw.flatten())
                        
                preds_cm[mod] = np.array(y_pred_mod)
                y_true_cm[mod] = np.array(y_true_mod)

        # 2. Compute Empirical Confusion
        cm_multi = confusion_matrix(y_true_cm["multispectral"], preds_cm["multispectral"], labels=unique_classes)
        cm_hyper = confusion_matrix(y_true_cm["hyperspectral"], preds_cm["hyperspectral"], labels=unique_classes)
        
        pairwise_records = []
        pairs = list(itertools.combinations(subset_classes, 2))
        
        # 3. Analyze Empirical Delta vs Theoretical Macro-Groups
        for cls_a, cls_b in tqdm(pairs, desc="Calculating Empirical Delta & HS Features"):
            idx_a, idx_b = unique_classes.index(cls_a), unique_classes.index(cls_b)
            
            errors_multi = cm_multi[idx_a, idx_b] + cm_multi[idx_b, idx_a]
            errors_hyper = cm_hyper[idx_a, idx_b] + cm_hyper[idx_b, idx_a]
            
            total_samples = np.sum(y_true_cm["multispectral"] == cls_a) + np.sum(y_true_cm["multispectral"] == cls_b)
            if total_samples == 0: continue
            
            conf_multi = errors_multi / total_samples
            conf_hyper = errors_hyper / total_samples
            empirical_delta = conf_multi - conf_hyper
            
            if model_type == "ml":
                importances = calculate_ml_pairwise_band_importance(hs_model, hs_X_scaled, y_test_pool, cls_a, cls_b)
            else:
                importances = calculate_unet_pairwise_band_importance(hs_model, common_files, hyper_dir, hs_scaler_mean, hs_scaler_scale, cls_a, cls_b, device)
                
            top_2_idx = np.argsort(importances)[-2:][::-1]
            pair_bands = []
            pair_groups = set()
            
            for b in top_2_idx:
                wl = h2crop_band_mapping_no_water_absorption_bands.get(b, b)
                if isinstance(wl, (int, float)):
                    group = get_macro_group(wl)
                    pair_bands.append(f"{wl:.1f}nm ({group})")
                    pair_groups.add(group)
                else:
                    pair_bands.append(f"Idx:{b} (Unknown)")
                    pair_groups.add("Unknown")
                    
            if len(pair_groups) == 1:
                explanation = "Theoretical Overlap: Top bands isolated to a single macro-group. If MS fails, it is due to spatial averaging of this narrow region."
            else:
                explanation = "Theoretical Distinction: Top bands span multiple macro-groups. If MS fails, it is likely due to broad-band dilution rather than complete overlap."

            pairwise_records.append({
                "Pair": f"{class_names[cls_a]} vs {class_names[cls_b]}",
                "MS Confusion Rate": conf_multi,
                "HS Confusion Rate": conf_hyper,
                "HS Benefit Delta": empirical_delta,
                "Top HS Separating Bands": " | ".join(pair_bands),
                "Macro-Groups": ", ".join(list(pair_groups)),
                "Theoretical Hardware Explanation": explanation
            })
            
        df_results = pd.DataFrame(pairwise_records)
        
        # Split into Valid Hardware Justifications (Delta >= 0) and Anomalies (Delta < 0)
        df_valid = df_results[df_results["HS Benefit Delta"] >= 0].sort_values(by="HS Benefit Delta", ascending=False)
        df_anomalies = df_results[df_results["HS Benefit Delta"] < 0].sort_values(by="HS Benefit Delta", ascending=True) # Sort by worst negative delta
        
        # 4. Save Outputs
        df_results.to_csv(os.path.join(out_dir, f"{model_name}_empirical_theoretical_analysis.csv"), index=False)
        
        report_path = os.path.join(out_dir, f"{model_name}_empirical_theoretical_report.txt")
        with open(report_path, "w") as f:
            f.write(f"EMPIRICAL VS THEORETICAL SEPARABILITY: {model_name.upper()}\n")
            f.write("="*70 + "\n\n")
            
            f.write("--- 1. VALID HARDWARE JUSTIFICATIONS (Sorted by HS Benefit Delta) ---\n")
            f.write("Positive delta implies Hyperspectral performs better. Higher delta = stronger upgrade justification.\n\n")
            
            for _, row in df_valid.iterrows():
                f.write(f">> {row['Pair']}\n")
                f.write(f"   Empirical Delta:      {row['HS Benefit Delta']:.2%} (MS Conf: {row['MS Confusion Rate']:.2%} -> HS Conf: {row['HS Confusion Rate']:.2%})\n")
                f.write(f"   Top HS Features:      {row['Top HS Separating Bands']}\n")
                f.write(f"   Macro-Groups:         {row['Macro-Groups']}\n")
                f.write(f"   Hardware Explanation: {row['Theoretical Hardware Explanation']}\n\n")
                
            f.write("\n" + "="*70 + "\n")
            f.write("--- 2. ANOMALIES / HYPERSPECTRAL UNDERPERFORMANCE (Negative Delta) ---\n")
            f.write("Multispectral outperformed Hyperspectral here. This indicates the HS model overfit to narrow-band noise\n")
            f.write("(The Hughes Phenomenon), while the broad MS bands provided robust regularization.\n\n")
            
            if df_anomalies.empty:
                f.write("  No anomalies detected.\n")
            else:
                for _, row in df_anomalies.iterrows():
                    f.write(f">> {row['Pair']}\n")
                    f.write(f"   Empirical Delta:      {row['HS Benefit Delta']:.2%} (MS Conf: {row['MS Confusion Rate']:.2%} -> HS Conf: {row['HS Confusion Rate']:.2%})\n")
                    f.write(f"   Top HS Features:      {row['Top HS Separating Bands']}\n")
                    f.write(f"   Macro-Groups:         {row['Macro-Groups']}\n\n")

    print(f"\nAnalysis complete! Sorted empirical-theoretical reports saved to: {out_dir}")