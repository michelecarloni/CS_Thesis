import os
import sys
import glob
import random
import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import uuid
import gc

# Setup project root path to import UNet and taxonomy dictionaries
project_root = os.path.abspath('../../')
if project_root not in sys.path:
    sys.path.append(project_root)

from models.unet import UNet  
from H2Crop.data_structures import h2crop_taxonomy_dict

# ==========================================
# EXTERNAL CONFIGURATION VARIABLES
# ==========================================
patch_size = 64
dataset_base = "../../ds"
ml_checkpoints_base = "/media/michele/T7/checkpoints/checkpoints_9_standard_ML_uniqueSub_split_same_tot_tiles"
unet_checkpoints_base = "/media/michele/T7/checkpoints/checkpoints_9_unet_pSize64_uniqueSub_NoGlobalSetTiles_ScalerYes_random_aug"
scalers_path = "/media/michele/T7/scalers/scalers_unique_subset" 

img_out_dir = "../../img_comp_unique_subset"


def find_unet_model(base_dir, split_id, modality, use_best_model=True):
    """
    Recursively searches the base_dir to find the U-Net model 
    that matches the given split_id and modality.
    """
    if use_best_model:
        search_pattern = os.path.join(base_dir, '**', 'best_model.pth')
    else:
        search_pattern = os.path.join(base_dir, '**', 'epoch_*.pth')
        
    all_models = glob.glob(search_pattern, recursive=True)
    
    if not use_best_model:
        all_models = sorted(all_models)
    
    for model_path in reversed(all_models): 
        if modality in model_path and f"split_{split_id}" in model_path:
            return model_path
            
    return None


def plot_inference_comparison(split_id, num_samples, pSize, taxonomy=3, unet_encoders=["resnet18"], use_best_model=True):
    """
    Generates comparative plots combining Hyperspectral and Multispectral rows in the SAME image.
    Applies unified Standard Scaling independently for ML (CPU) and U-Net (GPU).
    """
    print(f"\n{'='*70}")
    model_type_str = "BEST MODEL" if use_best_model else "FINAL EPOCH"
    print(f"GENERATING COMBINED INFERENCE PLOTS | Unique Subset - Split {split_id} | pSize {pSize} | {model_type_str}")
    print(f"{'='*70}")

    modalities = ["hyperspectral", "multispectral"]
    ml_models = ["decision_tree", "random_forest", "logistic_regression", "linear_svm"]
    
    titles = ["Ground Truth"] + [m.replace('_', ' ').title() for m in ml_models] + [f"UNET-{e.replace('resnet', 'ResNet')}" for e in unet_encoders]

    subset_classes = [8, 11, 23, 30, 38, 1, 2, 10, 29, 50]
    num_classes = len(subset_classes) + 1

    unique_classes = [0] + sorted(subset_classes)
    taxonomy_key = f'Taxonomy_{taxonomy}'
    class_names = [h2crop_taxonomy_dict.get(taxonomy_key, {}).get(c, f"Class {c}") if c != 0 else "Background" for c in unique_classes]
    
    base_cmap = plt.cm.tab20.colors
    cmap_colors = ['black'] + [base_cmap[i % len(base_cmap)] for i in range(len(subset_classes))]
    custom_cmap = ListedColormap(cmap_colors)
    legend_patches = [mpatches.Patch(color=cmap_colors[i], label=class_names[i]) for i in range(len(unique_classes))]

    inverse_mapping = {idx: raw_label for idx, raw_label in enumerate(unique_classes)}
    raw_to_plot_idx = {raw: idx for idx, raw in enumerate(unique_classes)}
    map_to_plot = np.vectorize(lambda x: raw_to_plot_idx.get(x, 0)) 

    current_dataset_base = os.path.join(dataset_base, "H2Crop_tiles_unique_subset", f"split_{split_id}")
    os.makedirs(img_out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. SYNCHRONIZED FILE SAMPLING
    print("\n--- Synchronizing Test Samples ---")
    hyper_test_dir = os.path.join(current_dataset_base, f"hyperspectral_taxonomy_{taxonomy}_pSize_{pSize}", "test")
    multi_test_dir = os.path.join(current_dataset_base, f"multispectral_taxonomy_{taxonomy}_pSize_{pSize}", "test")
    
    hyper_files = set(os.path.basename(f) for f in glob.glob(os.path.join(hyper_test_dir, "*.npz")))
    multi_files = set(os.path.basename(f) for f in glob.glob(os.path.join(multi_test_dir, "*.npz")))
    
    common_files = list(hyper_files.intersection(multi_files))
    
    if not common_files:
        print(f"[Error] No common test files found for Split {split_id}.")
        return
        
    sampled_basenames = random.sample(common_files, min(num_samples, len(common_files)))
    
    hyper_paths = [os.path.join(hyper_test_dir, f) for f in sampled_basenames]
    multi_paths = [os.path.join(multi_test_dir, f) for f in sampled_basenames]
    
    print(f"    Successfully matched and sampled {len(sampled_basenames)} identical geographic tiles.")

    # 2. LOAD MODELS & UNIFIED SCALERS
    print("\n--- Loading Models & Scalers ---")
    scalers_cpu = {}
    scalers_gpu = {"mean": {}, "scale": {}}
    ml_networks = {"hyperspectral": {}, "multispectral": {}}
    unet_networks = {"hyperspectral": {}, "multispectral": {}}
    
    for mod in modalities:
        in_channels = 218 if mod == "hyperspectral" else 10
            
        # Unified Scaler Load
        scaler_path = os.path.join(scalers_path, "unique_subset", mod, f"scaler_split_{split_id}_tax_{taxonomy}_pSize_{pSize}.joblib")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            scalers_cpu[mod] = scaler # Retained for Standard ML flat arrays
            
            # Convert to tensors for U-Net GPU Broadcasting
            scalers_gpu["mean"][mod] = torch.tensor(scaler.mean_, dtype=torch.float32, device=device).view(1, in_channels, 1, 1)
            scalers_gpu["scale"][mod] = torch.tensor(scaler.scale_, dtype=torch.float32, device=device).view(1, in_channels, 1, 1)
        else:
            print(f"    [Warning] Scaler not found at: {scaler_path}")
            
        # Standard ML Algorithms
        for ml_algo in ml_models:
            ml_model_path = os.path.join(ml_checkpoints_base, "unique_subset", ml_algo, mod, f"{ml_algo}_split_{split_id}_tax_{taxonomy}_pSize_{pSize}.joblib")
            if os.path.exists(ml_model_path):
                ml_networks[mod][ml_algo] = joblib.load(ml_model_path)
            else:
                print(f"    [Warning] ML Model not found at: {ml_model_path}")
                
        # Deep Learning U-Net
        for encoder in unet_encoders:
            checkpoint_path = find_unet_model(unet_checkpoints_base, split_id, mod, use_best_model)
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"    -> Loading U-Net weights from: {os.path.basename(checkpoint_path)}")
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
                
                block_indices = []
                for key in state_dict.keys():
                    if "model.decoder.blocks." in key:
                        try:
                            idx = int(key.split("model.decoder.blocks.")[1].split(".")[0])
                            block_indices.append(idx)
                        except ValueError:
                            pass
                
                inferred_depth = max(block_indices) + 1 if block_indices else 3
                
                model = UNet(in_channels=in_channels, num_classes=num_classes, encoder_name=encoder, encoder_depth=inferred_depth).to(device)
                model.load_state_dict(state_dict)
                model.eval()
                unet_networks[mod][encoder] = model
            else:
                print(f"    [Warning] U-Net Model not found for Split {split_id}, {mod}")

    # 3. STRICT 1-BY-1 INFERENCE & PLOT
    num_plots_row = 5 + len(unet_encoders)
    mul_k = 4 if num_plots_row == 7 else 5 if num_plots_row == 6 else 4
    nrows = 2 
    
    for idx, (hyper_file, multi_file) in enumerate(zip(hyper_paths, multi_paths)):
        print(f"\n    Generating plot {idx+1}/{len(hyper_paths)}...")
        
        fig, axes = plt.subplots(nrows, num_plots_row, figsize=(24, mul_k * nrows), gridspec_kw={'hspace': 0.45, 'wspace': 0.05})
        axes = np.atleast_2d(axes) 
        
        for row_idx, (mod, file_path) in enumerate([("hyperspectral", hyper_file), ("multispectral", multi_file)]):
            in_channels = 218 if mod == "hyperspectral" else 10
            
            with np.load(file_path) as data:
                X_raw = data['X'] 
                y_true = data['y']
                
            H, W = y_true.shape
            total_px = y_true.size
            predictions = [y_true]
            
            # A) Standard ML Inference (CPU Scaling)
            X_flat = X_raw.reshape(in_channels, -1).T
            
            # Neutralize NaNs before scaling
            X_flat = np.nan_to_num(X_flat, nan=0.0, posinf=0.0, neginf=0.0)
            if mod in scalers_cpu:
                X_flat = scalers_cpu[mod].transform(X_flat)
                
            for ml_algo in ml_models:
                if ml_algo in ml_networks[mod]:
                    y_pred_flat = ml_networks[mod][ml_algo].predict(X_flat)
                    predictions.append(y_pred_flat.reshape(H, W))
                    del y_pred_flat
                else:
                    predictions.append(np.zeros((H, W)))
                    
            # B) U-Net Inference (GPU Scaling)
            with torch.no_grad():
                X_tensor = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(0).to(device)
                X_tensor = torch.nan_to_num(X_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                
                if mod in scalers_gpu["mean"] and mod in scalers_gpu["scale"]:
                    X_tensor = (X_tensor - scalers_gpu["mean"][mod]) / scalers_gpu["scale"][mod]
                
                for encoder in unet_encoders:
                    if encoder in unet_networks[mod]:
                        output = unet_networks[mod][encoder](X_tensor)
                        pred_mapped = torch.max(output, 1)[1].squeeze(0).cpu().numpy()
                        pred_raw = np.vectorize(inverse_mapping.get)(pred_mapped)
                        predictions.append(pred_raw)
                        del output, pred_mapped, pred_raw
                    else:
                        predictions.append(np.zeros((H, W)))

            # C) Plot Row
            for col_idx, (pred_raw_grid, title) in enumerate(zip(predictions, titles)):
                ax = axes[row_idx, col_idx]
                pred_plot_grid = map_to_plot(pred_raw_grid)
                
                ax.imshow(pred_plot_grid, cmap=custom_cmap, vmin=0, vmax=len(unique_classes)-1, interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                
                if row_idx == 0:
                    ax.set_title(title, fontsize=16, pad=12)
                    
                if col_idx == 0:
                    ax.set_ylabel(f"{mod.title()}\nSample {idx + 1}", fontsize=14, labelpad=15, rotation=90)
                else:
                    correct_px = np.sum(pred_raw_grid == y_true)
                    acc = correct_px / total_px if total_px > 0 else 0.0

                    present_classes = np.unique(y_true)
                    present_classes = present_classes[present_classes != 0] 
                    
                    if len(present_classes) > 0:
                        iou_list = []
                        for c in present_classes:
                            intersection = np.sum((pred_raw_grid == c) & (y_true == c))
                            union = np.sum((pred_raw_grid == c) | (y_true == c))
                            iou_list.append(intersection / union if union > 0 else 0.0)
                        mIoU = np.mean(iou_list)
                        miou_text = f"mIoU: {mIoU:.1%}"
                    else:
                        miou_text = "mIoU: N/A"
                        
                    metrics_text = f"{miou_text} | OA: {acc:.1%}"
                    ax.set_xlabel(metrics_text, fontsize=12, fontweight='bold', color='black', labelpad=8)
                    
                del pred_plot_grid
                
            del X_raw, y_true, X_flat, X_tensor, predictions
            
        # D) Add Legend & Finalize
        fig.subplots_adjust(bottom=0.15)
        fig.legend(handles=legend_patches, loc='lower center', ncol=min(6, len(unique_classes)), 
                   bbox_to_anchor=(0.5, 0.02), fontsize=14, frameon=False)
        
        random_id = uuid.uuid4().hex[:8]
        type_str = "best" if use_best_model else "epoch"
        out_file = os.path.join(img_out_dir, f"inference_unique_split_{split_id}_{type_str}_sample_{idx+1}_{random_id}.png")
        
        plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
        
        fig.clf()
        plt.close(fig)
        plt.close('all')
        del fig, axes
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    print(f"\nAll plots saved successfully to {img_out_dir}!")


if __name__ == "__main__":
    split_list = [1, 2, 3]
    num_iter = 20
 
    for split in split_list:
        for i in range(0, num_iter):
            plot_inference_comparison(
                split_id=split, 
                num_samples=1, 
                pSize=patch_size, 
                taxonomy=3,
                use_best_model=True
            )