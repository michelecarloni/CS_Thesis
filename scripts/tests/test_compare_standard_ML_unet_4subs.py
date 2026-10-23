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

# Setup project root path to import UNet and taxonomy dictionaries
project_root = os.path.abspath('../../')
if project_root not in sys.path:
    sys.path.append(project_root)

from models.unet import UNet  
from H2Crop.data_structures import h2crop_taxonomy_dict


# EXTERNAL CONFIGURATION VARIABLES
patch_size = 64
dataset_base = "../../ds"
ml_checkpoints_base = "/media/michele/T7/checkpoints/checkpoints_8_standard_ML_tiles_4_subs_no_hyp_pSize_64_balance_split"
unet_checkpoints_base = "/media/michele/T7/checkpoints/checkpoints_8_unet_pSize_64_balanced_split_min_class_aug"
img_out_dir = "../../img"


def plot_inference_comparison(subset_id, num_samples, pSize, taxonomy=3, unet_encoders=["resnet18"]):
    """
    Generates comparative plots combining Hyperspectral and Multispectral rows in the SAME image.
    Interleaves the rows: HSI Sample 1, MSI Sample 1, HSI Sample 2, MSI Sample 2, etc.
    Utilizes synchronized sampling to ensure modalities display the exact same geographic tile.
    Plots Mean Intersection over Union (mIoU) AND Overall Accuracy (OA) below each prediction image.
    """
    print(f"\n{'='*70}")
    print(f"GENERATING COMBINED INFERENCE PLOTS | Subset {subset_id} | pSize {pSize}")
    print(f"{'='*70}")

    # CONFIGURATION & PATHS
    modalities = ["hyperspectral", "multispectral"]
    ml_models = ["decision_tree", "random_forest", "logistic_regression", "linear_svm"]
    
    titles = ["Ground Truth"] + [m.replace('_', ' ').title() for m in ml_models] + [f"UNET-{e.replace('resnet', 'ResNet')}" for e in unet_encoders]

    subsets = {
        1: [8, 11, 23, 56],
        2: [13, 30, 38, 53, 61],
        3: [1, 2, 10, 17],
        4: [29, 50, 64, 76]
    }
    subset_classes = subsets[subset_id]
    num_classes = len(subset_classes) + 1

    # COLORMAP & LEGEND SETUP
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

    # Construct the specific dataset path dynamically using the external dataset_base
    current_dataset_base = os.path.join(dataset_base, f"H2Crop_tiles_ds_subset_{subset_id}")
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
        print("[Error] No common test files found between hyperspectral and multispectral directories.")
        return
        
    sampled_basenames = random.sample(common_files, min(num_samples, len(common_files)))
    
    sampled_files = {
        "hyperspectral": [os.path.join(hyper_test_dir, f) for f in sampled_basenames],
        "multispectral": [os.path.join(multi_test_dir, f) for f in sampled_basenames]
    }
    print(f"    Successfully matched and sampled {len(sampled_basenames)} identical geographic tiles.")

    # 2. LOAD MODELS & SCALERS
    print("\n--- Loading Models & Scalers ---")
    scalers = {}
    ml_networks = {"hyperspectral": {}, "multispectral": {}}
    unet_networks = {"hyperspectral": {}, "multispectral": {}}
    
    use_optuna = "optuna" in ml_checkpoints_base.lower()
    suffix = "_optuna" if use_optuna else ""
    
    for mod in modalities:
        in_channels = 218 if mod == "hyperspectral" else 10
            
        scaler_filename = f"scaler_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{pSize}{suffix}.joblib"
        scaler_path = os.path.join(ml_checkpoints_base, "scalers", mod, scaler_filename)
        if os.path.exists(scaler_path):
            scalers[mod] = joblib.load(scaler_path)
            
        for ml_algo in ml_models:
            model_filename = f"{ml_algo}_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{pSize}{suffix}.joblib"
            ml_model_path = os.path.join(ml_checkpoints_base, ml_algo, mod, model_filename)
            if os.path.exists(ml_model_path):
                ml_networks[mod][ml_algo] = joblib.load(ml_model_path)
                
        for encoder in unet_encoders:
            checkpoint_path = os.path.join(unet_checkpoints_base, f"unet_enc_{encoder}", mod, f"subset_{subset_id}_pSize_{pSize}", f"best_model.pth")
            if os.path.exists(checkpoint_path):
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

    # 3. COMBINED BATCH INFERENCE & PLOT
    chunk_size = 1
    num_chunks = int(np.ceil(num_samples / chunk_size))

    num_plots_row = 5 + len(unet_encoders)
    
    # Safely assign mul_k based on the number of columns, with a fallback
    if num_plots_row == 7:
        mul_k = 4
    elif num_plots_row == 6:
        mul_k = 5
    else:
        mul_k = 4
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size
        
        hyper_chunk = sampled_files["hyperspectral"][start_idx:end_idx]
        multi_chunk = sampled_files["multispectral"][start_idx:end_idx]
        
        nrows = len(hyper_chunk) + len(multi_chunk)
        if nrows == 0:
            break
            
        print(f"\n    Generating plot batch {chunk_idx+1}/{num_chunks} ({nrows} rows total)...")
        
        # hspace increased to 0.45 to ensure room for the metrics printed below the image
        fig, axes = plt.subplots(nrows,
                                num_plots_row,
                                figsize=(24, mul_k * nrows),
                                gridspec_kw={'hspace': 0.45, 'wspace': 0.05}
                                )
        axes = np.atleast_2d(axes) 
        
        current_row = 0
        max_chunk_len = max(len(hyper_chunk), len(multi_chunk))
        
        for file_idx in range(max_chunk_len):
            for mod in modalities:
                chunk = hyper_chunk if mod == "hyperspectral" else multi_chunk
                
                if file_idx >= len(chunk):
                    continue
                    
                file_path = chunk[file_idx]
                in_channels = 218 if mod == "hyperspectral" else 10
                
                with np.load(file_path) as data:
                    X_raw = data['X'] 
                    y_true = data['y']
                    
                H, W = y_true.shape
                total_px = y_true.size
                predictions = [y_true]
                
                # A) ML Inference
                X_flat = X_raw.reshape(in_channels, -1).T
                if mod in scalers:
                    X_flat = scalers[mod].transform(X_flat)
                    
                for ml_algo in ml_models:
                    if ml_algo in ml_networks[mod]:
                        y_pred_flat = ml_networks[mod][ml_algo].predict(X_flat)
                        predictions.append(y_pred_flat.reshape(H, W))
                    else:
                        predictions.append(np.zeros((H, W)))
                        
                # B) U-Net Inference
                with torch.no_grad():
                    X_tensor = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(0).to(device)
                    for encoder in unet_encoders:
                        if encoder in unet_networks[mod]:
                            output = unet_networks[mod][encoder](X_tensor)
                            pred_mapped = torch.max(output, 1)[1].squeeze(0).cpu().numpy()
                            pred_raw = np.vectorize(inverse_mapping.get)(pred_mapped)
                            predictions.append(pred_raw)
                        else:
                            predictions.append(np.zeros((H, W)))

                # C) Plot Row
                for col_idx, (pred_raw_grid, title) in enumerate(zip(predictions, titles)):
                    ax = axes[current_row, col_idx]
                    pred_plot_grid = map_to_plot(pred_raw_grid)
                    
                    ax.imshow(pred_plot_grid, cmap=custom_cmap, vmin=0, vmax=len(unique_classes)-1, interpolation='nearest')
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    if current_row == 0:
                        ax.set_title(title, fontsize=16, pad=12)
                        
                    if col_idx == 0:
                        ax.set_ylabel(f"{mod.title()}\nSample {start_idx + file_idx + 1}", fontsize=14, labelpad=15, rotation=90)
                    else:
                        # 1. Overall Accuracy (including background)
                        correct_px = np.sum(pred_raw_grid == y_true)
                        acc = correct_px / total_px if total_px > 0 else 0.0

                        # 2. Mean IoU (excluding background)
                        present_classes = np.unique(y_true)
                        present_classes = present_classes[present_classes != 0] # exclude background class 0
                        
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
                            
                        # Combine both metrics to be displayed below the image
                        metrics_text = f"{miou_text} | OA: {acc:.1%}"
                        
                        # Plot metrics text directly under the x-axis of each image plot
                        ax.set_xlabel(metrics_text, fontsize=12, fontweight='bold', color='black', labelpad=8)
                
                current_row += 1
                
        # D) Add Legend & Finalize (tight_layout removed to prevent warnings)
        fig.subplots_adjust(bottom=0.15)
        fig.legend(handles=legend_patches, loc='lower center', ncol=len(unique_classes), 
                   bbox_to_anchor=(0.5, 0.02), fontsize=14, frameon=False)
        
        random_id = uuid.uuid4().hex[:8]
        out_file = os.path.join(img_out_dir, f"inference_sub{subset_id}_combined_batch_{chunk_idx+1}_{random_id}.png")
        
        plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
            
    print(f"\nAll batched plots saved successfully to {img_out_dir}!")


def plot_all_inference_comparisons(subset_id, pSize, taxonomy=3, unet_encoders=["resnet18"]):
    """
    Generates comparative plots combining Hyperspectral and Multispectral rows in the SAME image
    for ALL tiles present in the test set. Overlays both mIoU and Overall Accuracy universally below each plot.
    """
    print(f"\n{'='*70}")
    print(f"GENERATING FULL TEST SET INFERENCE PLOTS | Subset {subset_id} | pSize {pSize}")
    print(f"{'='*70}")

    # CONFIGURATION & PATHS
    modalities = ["hyperspectral", "multispectral"]
    ml_models = ["decision_tree", "random_forest", "logistic_regression", "linear_svm"]
    
    titles = ["Ground Truth"] + [m.replace('_', ' ').title() for m in ml_models] + [f"UNET-{e.replace('resnet', 'ResNet')}" for e in unet_encoders]

    subsets = {
        1: [8, 11, 23, 56],
        2: [13, 30, 38, 53, 61],
        3: [1, 2, 10, 17],
        4: [29, 50, 64, 76]
    }
    subset_classes = subsets[subset_id]
    num_classes = len(subset_classes) + 1
    
    # COLORMAP & LEGEND SETUP
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

    current_dataset_base = os.path.join(dataset_base, f"H2Crop_tiles_ds_subset_{subset_id}")
    os.makedirs(img_out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. SYNCHRONIZED FILE SAMPLING (ALL TEST FILES)
    print("\n--- Synchronizing Test Samples ---")
    hyper_test_dir = os.path.join(current_dataset_base, f"hyperspectral_taxonomy_{taxonomy}_pSize_{pSize}", "test")
    multi_test_dir = os.path.join(current_dataset_base, f"multispectral_taxonomy_{taxonomy}_pSize_{pSize}", "test")
    
    hyper_files = set(os.path.basename(f) for f in glob.glob(os.path.join(hyper_test_dir, "*.npz")))
    multi_files = set(os.path.basename(f) for f in glob.glob(os.path.join(multi_test_dir, "*.npz")))
    
    common_files = list(hyper_files.intersection(multi_files))
    
    if not common_files:
        print("[Error] No common test files found between hyperspectral and multispectral directories.")
        return
        
    sampled_basenames = sorted(common_files)
    num_samples = len(sampled_basenames)
    
    sampled_files = {
        "hyperspectral": [os.path.join(hyper_test_dir, f) for f in sampled_basenames],
        "multispectral": [os.path.join(multi_test_dir, f) for f in sampled_basenames]
    }
    print(f"    Successfully matched {num_samples} identical geographic tiles across modalities.")

    # 2. LOAD MODELS & SCALERS
    print("\n--- Loading Models & Scalers ---")
    scalers = {}
    ml_networks = {"hyperspectral": {}, "multispectral": {}}
    unet_networks = {"hyperspectral": {}, "multispectral": {}}
    
    use_optuna = "optuna" in ml_checkpoints_base.lower()
    suffix = "_optuna" if use_optuna else ""
    
    for mod in modalities:
        in_channels = 218 if mod == "hyperspectral" else 10
            
        scaler_filename = f"scaler_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{pSize}{suffix}.joblib"
        scaler_path = os.path.join(ml_checkpoints_base, "scalers", mod, scaler_filename)
        if os.path.exists(scaler_path):
            scalers[mod] = joblib.load(scaler_path)
            
        for ml_algo in ml_models:
            model_filename = f"{ml_algo}_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{pSize}{suffix}.joblib"
            ml_model_path = os.path.join(ml_checkpoints_base, ml_algo, mod, model_filename)
            if os.path.exists(ml_model_path):
                ml_networks[mod][ml_algo] = joblib.load(ml_model_path)
                
        for encoder in unet_encoders:
            checkpoint_path = os.path.join(unet_checkpoints_base, f"unet_enc_{encoder}", mod, f"subset_{subset_id}_pSize_{pSize}", f"best_model.pth")
            if os.path.exists(checkpoint_path):
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

    # 3. COMBINED BATCH INFERENCE & PLOT
    chunk_size = 1
    num_chunks = int(np.ceil(num_samples / chunk_size))

    num_plots_row = 5 + len(unet_encoders)
    
    # Safely assign mul_k based on the number of columns, with a fallback
    if num_plots_row == 7:
        mul_k = 4
    elif num_plots_row == 6:
        mul_k = 5
    else:
        mul_k = 4
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size
        
        hyper_chunk = sampled_files["hyperspectral"][start_idx:end_idx]
        multi_chunk = sampled_files["multispectral"][start_idx:end_idx]
        
        nrows = len(hyper_chunk) + len(multi_chunk)
        if nrows == 0:
            break
            
        print(f"\n    Generating plot batch {chunk_idx+1}/{num_chunks} ({nrows} rows total)...")
        
        # hspace increased to 0.45 to ensure room for the metrics printed below the image
        fig, axes = plt.subplots(nrows,
                                num_plots_row,
                                figsize=(24, mul_k * nrows),
                                gridspec_kw={'hspace': 0.45, 'wspace': 0.05}
                                )
        axes = np.atleast_2d(axes) 
        
        current_row = 0
        max_chunk_len = max(len(hyper_chunk), len(multi_chunk))
        
        for file_idx in range(max_chunk_len):
            for mod in modalities:
                chunk = hyper_chunk if mod == "hyperspectral" else multi_chunk
                
                if file_idx >= len(chunk):
                    continue
                    
                file_path = chunk[file_idx]
                in_channels = 218 if mod == "hyperspectral" else 10
                
                with np.load(file_path) as data:
                    X_raw = data['X'] 
                    y_true = data['y']
                    
                H, W = y_true.shape
                total_px = y_true.size
                predictions = [y_true]
                
                # A) ML Inference
                X_flat = X_raw.reshape(in_channels, -1).T
                if mod in scalers:
                    X_flat = scalers[mod].transform(X_flat)
                    
                for ml_algo in ml_models:
                    if ml_algo in ml_networks[mod]:
                        y_pred_flat = ml_networks[mod][ml_algo].predict(X_flat)
                        predictions.append(y_pred_flat.reshape(H, W))
                    else:
                        predictions.append(np.zeros((H, W)))
                        
                # B) U-Net Inference
                with torch.no_grad():
                    X_tensor = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(0).to(device)
                    for encoder in unet_encoders:
                        if encoder in unet_networks[mod]:
                            output = unet_networks[mod][encoder](X_tensor)
                            pred_mapped = torch.max(output, 1)[1].squeeze(0).cpu().numpy()
                            pred_raw = np.vectorize(inverse_mapping.get)(pred_mapped)
                            predictions.append(pred_raw)
                        else:
                            predictions.append(np.zeros((H, W)))

                # C) Plot Row
                for col_idx, (pred_raw_grid, title) in enumerate(zip(predictions, titles)):
                    ax = axes[current_row, col_idx]
                    pred_plot_grid = map_to_plot(pred_raw_grid)
                    
                    ax.imshow(pred_plot_grid, cmap=custom_cmap, vmin=0, vmax=len(unique_classes)-1, interpolation='nearest')
                    
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    if current_row == 0:
                        ax.set_title(title, fontsize=16, pad=12)
                        
                    if col_idx == 0:
                        ax.set_ylabel(f"{mod.title()}\nSample {start_idx + file_idx + 1}", fontsize=14, labelpad=15, rotation=90)
                    else:
                        # 1. Overall Accuracy (including background)
                        correct_px = np.sum(pred_raw_grid == y_true)
                        acc = correct_px / total_px if total_px > 0 else 0.0

                        # 2. Mean IoU (excluding background)
                        present_classes = np.unique(y_true)
                        present_classes = present_classes[present_classes != 0] # exclude background class 0
                        
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
                            
                        # Combine both metrics to be displayed below the image
                        metrics_text = f"{miou_text} | OA: {acc:.1%}"
                        
                        # Plot metrics text directly under the x-axis of each image plot
                        ax.set_xlabel(metrics_text, fontsize=12, fontweight='bold', color='black', labelpad=8)
                
                current_row += 1
                
        # D) Add Legend & Finalize (tight_layout removed to prevent warnings)
        fig.subplots_adjust(bottom=0.15)
        fig.legend(handles=legend_patches, loc='lower center', ncol=len(unique_classes), 
                   bbox_to_anchor=(0.5, 0.02), fontsize=14, frameon=False)
        
        out_file = os.path.join(img_out_dir, f"inference_sub{subset_id}_all_batch_{chunk_idx+1}.png")
        
        plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
            
    print(f"\nAll full-test-set batched plots saved successfully to {img_out_dir}!")



"""
if __name__ == "__main__":
    sub_list = [1,2,3,4]
    num_iter = 20
 
    for sub in sub_list:
        for i in range (0, num_iter):
            plot_inference_comparison(subset_id=sub, num_samples=1, pSize=patch_size, taxonomy=3)
"""

            
if __name__ == "__main__":
    sub_list = [1, 2, 3, 4]

    for sub in sub_list:
        plot_all_inference_comparisons(subset_id=sub, pSize=patch_size, taxonomy=3)