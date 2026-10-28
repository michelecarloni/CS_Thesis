import os
import sys

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

from H2Crop.H2Crop import H2Crop
from scripts.extract_tiles_unique_sub import extract_unique_sub_tiles

# Import pipeline
from pipelines.pipeline_H2Crop_unet_unique_subset import pipeline_H2Crop_unet_unique

if __name__ == "__main__":
    # ==========================================
    # HYPERPARAMETERS & CONFIGURATION
    # ==========================================
    modalities = ["hyperspectral", "multispectral"]
    taxonomy = 3
    patch_sizes = [64]
    valid_threshold = 0.10  
    save_results_dir = "../results_9"
    
    encoders = ["resnet18"]
    use_gpu = True
    debug = False
    
    # Unified subset configuration (Classes common to previous subsets 1, 2, 3, and 4)
    unique_classes = [8, 11, 23, 30, 38, 1, 2, 10, 29, 50]
    
    # Cross Validation Settings
    num_splits = 3
    apply_mild_balance = True # Toggle to False for completely random train set extraction
    
    # U-Net Training parameters
    final_epochs = 100
    batch_size = 32
    encoder_depth = 4
    train_aug_mode = "random_aug" # Must be "random_aug" or "no_aug" for this pipeline
    
    print("Initializing H2Crop Loader...")
    loader = H2Crop()

    # ==========================================
    # 1. UNIQUE SUBSET EXTRACTION PHASE
    # ==========================================
    extract_unique_sub_tiles(
        patch_sizes=patch_sizes, 
        unique_classes=unique_classes, 
        modalities=modalities, 
        loader=loader, 
        taxonomy=taxonomy, 
        valid_threshold=valid_threshold,
        num_splits=num_splits,
        apply_mild_balance=apply_mild_balance
    )

    # ==========================================
    # 2. TRAINING PHASE (Deep Learning U-Net CV)
    # ==========================================
    print("\n--- Starting Deep Learning Phase (CV across Splits) ---")
    
    # Use ../ to step out of the scripts folder
    save_base_dir = "../ds/H2Crop_tiles_unique_subset"
    
    for encoder in encoders:
        for patch_size in patch_sizes:
            
            for split_id in range(1, num_splits + 1):
                # We define the specific results folder for this split and model
                split_results_dir = os.path.join(save_results_dir, f"split_{split_id}", f"unet_enc_{encoder}_{patch_size}")
                
                for mod in modalities:
                    dataset_dir = os.path.join(save_base_dir, f"split_{split_id}", f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
                    
                    if os.path.exists(dataset_dir):
                        in_channels = 218 if mod == "hyperspectral" else 10
                        model_name = f"unet_enc_{encoder}"
                        
                        print(f"\n[*] Preparing {model_name} for {mod.upper()} | SPLIT {split_id}")
                        
                        # Call the new pipeline
                        pipeline_H2Crop_unet_unique(
                            in_channels=in_channels,
                            encoder_name=encoder,
                            model_name=model_name,
                            save_results_dir=split_results_dir,
                            dataset_dir=dataset_dir,
                            split_id=split_id,
                            subset_classes=unique_classes,
                            modality=mod,
                            taxonomy=taxonomy,
                            patch_size=patch_size,
                            use_gpu=use_gpu,
                            epochs=final_epochs,
                            batch_size=batch_size,
                            encoder_depth=encoder_depth,
                            lr=1e-3,
                            weight_decay=1e-4,
                            debug=debug,
                            train_aug_mode=train_aug_mode
                        )
                    else:
                        print(f"[Error] Dataset directory {dataset_dir} missing. Skipping training for this config.")
                        
    print("\n--- Cross-Validation Phase Complete! ---")