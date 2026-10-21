import os
import sys

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

from H2Crop.H2Crop import H2Crop
from scripts.extract_tiles_4_subs import extract_4_subs_tiles

# Import pipeline
from pipelines.pipeline_H2Crop_unet import pipeline_H2Crop_unet

if __name__ == "__main__":
    # ==========================================
    # HYPERPARAMETERS & CONFIGURATION
    # ==========================================
    modalities = ["hyperspectral", "multispectral"]
    taxonomy = 3
    patch_sizes = [64]
    valid_threshold = 0.05  # Increased to ensure good crop density per tile
    save_results_dir = "../results_8"
    
    # Exclusively use ResNet-18 encoder
    encoders = ["resnet18"]
    
    use_gpu = True
    use_optuna = False
    
    n_trials = 10
    epochs_per_trial = 15
    final_epochs = 100       # Increased to 40 for CosineAnnealingLR convergence
    batch_size = 32         # Optimized batch size for 64x64 tiles
    encoder_depth = 4       # Matched with 64x64 tiles and ResNet-18
    debug = False
    
    subsets = {
        1: [8, 11, 23, 56],
        2: [13, 30, 38, 53, 61],
        3: [1, 2, 10, 17],
        4: [29, 50, 64, 76]
    }
    
    print("Initializing H2Crop Loader...")
    loader = H2Crop()

    # ==========================================
    # 1. SUBSET EXTRACTION PHASE
    # ==========================================
    extract_4_subs_tiles(patch_sizes, subsets, modalities, loader, taxonomy, valid_threshold)

    # ==========================================
    # 2. TRAINING PHASE (Deep Learning U-Net)
    # ==========================================
    print("\n--- Starting Deep Learning Phase (U-Net) ---")
    
    for encoder in encoders:
        for patch_size in patch_sizes:
            
            current_results_dir = os.path.join(save_results_dir, f"unet_enc_{encoder}_{patch_size}")
            
            for subset_id, subset_classes in subsets.items():
                save_base_dir = f"../ds/H2Crop_tiles_ds_subset_{subset_id}"
                
                for mod in modalities:
                    dataset_dir = os.path.join(save_base_dir, f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
                    
                    if os.path.exists(dataset_dir):
                        in_channels = 218 if mod == "hyperspectral" else 10
                        num_classes = len(subset_classes) + 1
                        model_name = f"unet_enc_{encoder}"
                        
                        print(f"\nInitializing {model_name} for {mod.upper()} Subset {subset_id}...")
                        
                        # Standard Training (No Optuna)
                        pipeline_H2Crop_unet(
                            in_channels=in_channels,
                            encoder_name=encoder,
                            model_name=model_name,
                            save_results_dir=current_results_dir,
                            dataset_dir=dataset_dir,
                            subset_id=subset_id,
                            subset_classes=subset_classes,
                            modality=mod,
                            taxonomy=taxonomy,
                            patch_size=patch_size,
                            use_gpu=use_gpu,
                            epochs=final_epochs,
                            batch_size=batch_size,
                            encoder_depth=encoder_depth,
                            lr=1e-3,
                            weight_decay=1e-4,
                            debug=debug
                        )
                    else:
                        print(f"[Error] Dataset directory {dataset_dir} missing. Skipping training for this config.")