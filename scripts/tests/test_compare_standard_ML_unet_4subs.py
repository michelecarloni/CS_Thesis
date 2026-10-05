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

# Setup project root path to import UNet and taxonomy dictionaries
project_root = os.path.abspath('../../')
if project_root not in sys.path:
    sys.path.append(project_root)

from models.unet import UNet  
from H2Crop.data_structures import h2crop_taxonomy_dict


# EXTERNAL CONFIGURATION VARIABLES
patch_size = 64
dataset_base = "../../ds"
ml_checkpoints_base = "/media/michele/T7/checkpoints/checkpoints_6_standard_ML_tiles_4_subs_optuna_target_MacroF1score_patch_size_64_train_balanced"
unet_checkpoints_base = "/media/michele/T7/checkpoints/checkpoints_6_unet_4subs_optuna_focal_dice_patch_size_64_train_balanced"
img_out_dir = "../../img"


def plot_inference_comparison(subset_id, num_samples, pSize, taxonomy=3):
    """
    Generates comparative plots combining Hyperspectral and Multispectral rows in the SAME image.
    Interleaves the rows: HSI Sample 1, MSI Sample 1, HSI Sample 2, MSI Sample 2, etc.
    Utilizes external global variables for paths and configurations.
    """
    print(f"\n{'='*70}")
    print(f"GENERATING COMBINED INFERENCE PLOTS | Subset {subset_id} | pSize {pSize}")
    print(f"{'='*70}")

    # CONFIGURATION & PATHS
    modalities = ["hyperspectral", "multispectral"]
    ml_models = ["decision_tree", "random_forest", "logistic_regression", "linear_svm"]
    unet_encoders = ["resnet18", "resnet50"]
    
    titles = ["Ground Truth"] + [m.replace('_', ' ').title() for m in ml_models] + [f"UNET-{e.replace('resnet', 'ResNet')}" for e in unet_encoders]

    subsets = {
        1: [8, 11, 23, 56],
        2: [13, 30, 38, 53, 61],
        3: [1, 2, 10, 17],
        4: [29, 50, 64, 76]
    }
    subset_classes = subsets[subset_id]
    num_classes = len(subset_classes) + 1

    unet_epoch = 30
    
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

    # LOAD EVERYTHING UPFRONT
    print("\n--- Loading Models & Sampling Files ---")
    scalers = {}
    ml_networks = {"hyperspectral": {}, "multispectral": {}}
    unet_networks = {"hyperspectral": {}, "multispectral": {}}
    sampled_files = {"hyperspectral": [], "multispectral": []}
    
    for mod in modalities:
        in_channels = 218 if mod == "hyperspectral" else 10
        
        # Files (using the dynamically constructed current_dataset_base)
        test_dir = os.path.join(current_dataset_base, f"{mod}_taxonomy_{taxonomy}_pSize_{pSize}", "test")
        test_files = glob.glob(os.path.join(test_dir, "*.npz"))
        if test_files:
            sampled_files[mod] = random.sample(test_files, min(num_samples, len(test_files)))
            
        # Scaler (using external ml_checkpoints_base)
        scaler_path = os.path.join(ml_checkpoints_base, "scalers", mod, f"scaler_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{pSize}.joblib")
        if os.path.exists(scaler_path):
            scalers[mod] = joblib.load(scaler_path)
            
        # ML Models (using external ml_checkpoints_base)
        for ml_algo in ml_models:
            ml_model_path = os.path.join(ml_checkpoints_base, ml_algo, mod, f"{ml_algo}_tiles_subset_{subset_id}_optuna.joblib")
            if os.path.exists(ml_model_path):
                ml_networks[mod][ml_algo] = joblib.load(ml_model_path)
                
        # U-Net Models (using external unet_checkpoints_base)
        for encoder in unet_encoders:
            checkpoint_path = os.path.join(unet_checkpoints_base, f"unet_enc_{encoder}", mod, f"subset_{subset_id}_pSize_{pSize}", f"epoch_{unet_epoch}.pth")
            if os.path.exists(checkpoint_path):
                # 1. Load the raw dictionary of weights first
                state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
                
                # 2. Inspect the dictionary keys to find the maximum decoder block number
                block_indices = []
                for key in state_dict.keys():
                    if "model.decoder.blocks." in key:
                        try:
                            idx = int(key.split("model.decoder.blocks.")[1].split(".")[0])
                            block_indices.append(idx)
                        except ValueError:
                            pass
                
                # 3. Calculate inferred depth (max index + 1, fallback to 3 if missing)
                inferred_depth = max(block_indices) + 1 if block_indices else 3
                print(f"      [Auto-Detect] {encoder} ({mod}) was trained with encoder_depth={inferred_depth}")
                
                # 4. Instantiate the model dynamically and load weights
                model = UNet(in_channels=in_channels, num_classes=num_classes, encoder_name=encoder, encoder_depth=inferred_depth).to(device)
                model.load_state_dict(state_dict)
                model.eval()
                unet_networks[mod][encoder] = model

    # COMBINED BATCH INFERENCE & PLOT
    chunk_size = 3
    num_chunks = int(np.ceil(num_samples / chunk_size))
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = start_idx + chunk_size
        
        hyper_chunk = sampled_files["hyperspectral"][start_idx:end_idx]
        multi_chunk = sampled_files["multispectral"][start_idx:end_idx]
        
        nrows = len(hyper_chunk) + len(multi_chunk)
        if nrows == 0:
            break
            
        print(f"\n    Generating plot batch {chunk_idx+1}/{num_chunks} ({nrows} rows total)...")
        
        fig, axes = plt.subplots(nrows, 7, figsize=(24, 4 * nrows))
        axes = np.atleast_2d(axes) 
        
        current_row = 0
        
        # Loop through file indices first, THEN modalities to interleave them
        max_chunk_len = max(len(hyper_chunk), len(multi_chunk))
        
        for file_idx in range(max_chunk_len):
            for mod in modalities:
                chunk = hyper_chunk if mod == "hyperspectral" else multi_chunk
                
                # Skip if this modality doesn't have a file at this index
                if file_idx >= len(chunk):
                    continue
                    
                file_path = chunk[file_idx]
                in_channels = 218 if mod == "hyperspectral" else 10
                
                with np.load(file_path) as data:
                    X_raw = data['X'] 
                    y_true = data['y']
                    
                H, W = y_true.shape
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
                    
                    # Remove x/y ticks but keep the box
                    ax.set_xticks([])
                    ax.set_yticks([])
                    
                    # Add titles only to the very top row
                    if current_row == 0:
                        ax.set_title(title, fontsize=16, pad=12)
                        
                    # Add Modality label to the Ground Truth column
                    if col_idx == 0:
                        ax.set_ylabel(f"{mod.title()}\nSample {start_idx + file_idx + 1}", fontsize=14, labelpad=15, rotation=90)
                
                current_row += 1
                
        # D) Add Legend & Finalize
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        fig.legend(handles=legend_patches, loc='lower center', ncol=len(unique_classes), 
                   bbox_to_anchor=(0.5, 0.02), fontsize=14, frameon=False)
        
        out_file = os.path.join(img_out_dir, f"inference_sub{subset_id}_combined_batch_{chunk_idx+1}.png")
        plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
            
    print(f"\nAll batched plots saved successfully to {img_out_dir}!")

if __name__ == "__main__":
    # Ensure we use the global patch_size variable for the script execution
    plot_inference_comparison(subset_id=1, num_samples=30, pSize=patch_size, taxonomy=3)
    plot_inference_comparison(subset_id=2, num_samples=30, pSize=patch_size, taxonomy=3)
    plot_inference_comparison(subset_id=3, num_samples=30, pSize=patch_size, taxonomy=3)
    plot_inference_comparison(subset_id=4, num_samples=30, pSize=patch_size, taxonomy=3)