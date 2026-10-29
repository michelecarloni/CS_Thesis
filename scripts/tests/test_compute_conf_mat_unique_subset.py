import os
import sys
import gc
import csv
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Setup project root path
project_root = os.path.abspath('../..')
if project_root not in sys.path:
    sys.path.append(project_root)

from H2Crop.data_structures import h2crop_taxonomy_dict
from H2Crop.H2CropTileDataset import H2CropTileDataset
from models.unet import UNet  

def find_best_model(base_dir, split_id, modality):
    """
    Recursively searches the base_dir to find the 'best_model.pth' 
    that matches the given split_id and modality.
    """
    search_pattern = os.path.join(base_dir, '**', 'best_model.pth')
    all_models = glob.glob(search_pattern, recursive=True)
    
    for model_path in all_models:
        if modality in model_path and f"split_{split_id}" in model_path:
            return model_path
            
    return None

if __name__ == "__main__":
    # ==========================================
    # 1. CONFIGURATION & PATHS
    # ==========================================
    # Insert your absolute path here
    checkpoints_base_dir = "/mnt/windows/backupUsed/School career/University - Master/CS_Thesis/checkpoints" 
    
    dataset_base_dir = os.path.join(project_root, "ds", "H2Crop_tiles_unique_subset")
    evaluation_out_base = os.path.join(project_root, "conf_mat_unique_subset")
    
    modalities = ["hyperspectral", "multispectral"]
    num_splits = 3
    patch_size = 64
    batch_size = 32
    taxonomy = 3
    encoder_name = "resnet18"
    encoder_depth = 4
    use_gpu = True
    
    subset_classes = [8, 11, 23, 30, 38, 1, 2, 10, 29, 50]
    num_classes = len(subset_classes) + 1
    sample_y = [0] + sorted(subset_classes)
    
    taxonomy_key = f'Taxonomy_{taxonomy}'
    target_names = [h2crop_taxonomy_dict.get(taxonomy_key, {}).get(c, f"Class {c}") if c != 0 else "Background (0)" for c in sample_y]
    
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # ==========================================
    # 2. EVALUATION LOOP
    # ==========================================
    for mod in modalities:
        print(f"\n{'='*70}")
        print(f"STARTING EVALUATION FOR MODALITY: {mod.upper()}")
        print(f"{'='*70}")
        
        in_channels = 218 if mod == "hyperspectral" else 10
        
        for split_id in range(1, num_splits + 1):
            print(f"\n[*] Processing Split {split_id}...")
            
            # Create the target directory: conf_mat_unique_subset/modality/split_X
            split_out_dir = os.path.join(evaluation_out_base, mod, f"split_{split_id}")
            os.makedirs(split_out_dir, exist_ok=True)
            
            best_model_path = find_best_model(checkpoints_base_dir, split_id, mod)
            if not best_model_path:
                print(f"    [Error] Could not find best_model.pth for Split {split_id}, {mod}. Skipping.")
                continue
            print(f"    -> Found model: {best_model_path}")
            
            test_dataset_dir = os.path.join(dataset_base_dir, f"split_{split_id}", f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}", "test")
            if not os.path.exists(test_dataset_dir):
                print(f"    [Error] Test dataset missing at {test_dataset_dir}. Skipping.")
                continue
                
            test_dataset = H2CropTileDataset(
                test_dataset_dir, 
                subset_classes=subset_classes, 
                minority_classes=[], 
                debug=False,
                train_aug_mode="no_aug"
            )
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=min(4, os.cpu_count() or 1), pin_memory=use_gpu)
            
            model = UNet(in_channels=in_channels, num_classes=num_classes, encoder_name=encoder_name, encoder_depth=encoder_depth).to(device)
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            model.eval()
            
            global_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
            
            eval_loop = tqdm(test_loader, desc=f"Evaluating Split {split_id}", leave=False)
            with torch.no_grad():
                for batch_X, batch_y in eval_loop:
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.long().to(device, non_blocking=True)
                    
                    batch_X = torch.nan_to_num(batch_X, nan=0.0, posinf=0.0, neginf=0.0)
                    outputs = model(batch_X)
                        
                    _, predicted = torch.max(outputs.data, 1)
                    
                    indices = num_classes * batch_y.view(-1) + predicted.view(-1)
                    batch_cm = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
                    global_cm += batch_cm
                    
            cm_numpy = global_cm.cpu().numpy()
            
            # Prepare CSV Metrics
            split_metrics = []
            macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
            
            for idx, target_name in enumerate(target_names):
                tp = cm_numpy[idx, idx]
                fp = cm_numpy[:, idx].sum() - tp
                fn = cm_numpy[idx, :].sum() - tp
                support = cm_numpy[idx, :].sum()
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                
                split_metrics.append([target_name, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}", support])
                
                if idx > 0: 
                    macro_p += p; macro_r += r; macro_f1 += f1
                    
            macro_p /= (num_classes - 1)
            macro_r /= (num_classes - 1)
            macro_f1 /= (num_classes - 1)
            
            # Save CSV file in the split folder without modality in the name
            csv_path = os.path.join(split_out_dir, "metrics.csv")
            with open(csv_path, mode='w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support_Pixels"])
                writer.writerows(split_metrics)
                writer.writerow(["MACRO_AVG_CROPS", f"{macro_p:.4f}", f"{macro_r:.4f}", f"{macro_f1:.4f}", ""])
            
            # Plot Absolute Confusion Matrix
            fig_abs, ax_abs = plt.subplots(figsize=(12, 10))
            ConfusionMatrixDisplay(confusion_matrix=cm_numpy, display_labels=target_names).plot(ax=ax_abs, cmap='Blues', colorbar=True, values_format='d')
            plt.title(f"Absolute Confusion Matrix (Pixels)\n{mod.upper()} | Split {split_id}")
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(split_out_dir, "cm_absolute.png"), dpi=300)
            plt.close(fig_abs)
            
            # Plot Normalized Confusion Matrix
            cm_normalized = cm_numpy.astype('float') / cm_numpy.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized) 
            
            fig_norm, ax_norm = plt.subplots(figsize=(12, 10))
            ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=target_names).plot(ax=ax_norm, cmap='Blues', colorbar=True, values_format='.2f')
            plt.title(f"Normalized Confusion Matrix (Recall)\n{mod.upper()} | Split {split_id}")
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(split_out_dir, "cm_normalized.png"), dpi=300)
            plt.close(fig_norm)
            
            print(f"    -> Saved metrics and matrices to {split_out_dir}")
            
            del model, global_cm, test_dataset, test_loader
            if use_gpu: torch.cuda.empty_cache()
            gc.collect()

    print("\n--- Evaluation Complete! ---")