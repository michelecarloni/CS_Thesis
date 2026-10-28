import os
import sys

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

from H2Crop.H2Crop import H2Crop
from scripts.extract_tiles_unique_sub import extract_unique_sub_tiles

# (Placeholder for the pipeline import)
# from pipelines.pipeline_H2Crop_unet_unique import pipeline_H2Crop_unet_unique

if __name__ == "__main__":
    # ==========================================
    # HYPERPARAMETERS & CONFIGURATION
    # ==========================================
    modalities = ["hyperspectral", "multispectral"]
    taxonomy = 3
    patch_sizes = [64]
    valid_threshold = 0.10  
    save_results_dir = "../results_8_unique_subset"
    
    encoders = ["resnet18"]
    use_gpu = True
    
    # Unified subset configuration
    unique_classes = [8, 11, 23, 30, 38, 1, 2, 10, 29, 50]
    
    # Cross Validation Settings
    num_splits = 3
    apply_mild_balance = True # Toggle to False for completely random extraction
    
    # U-Net Training parameters (for later)
    final_epochs = 100
    batch_size = 32
    encoder_depth = 4
    train_aug_mode = "random_aug"
    
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

    sys.exit(0)

    # ==========================================
    # 2. TRAINING PHASE (Deep Learning U-Net)
    # ==========================================
    print("\n--- Starting Deep Learning Phase (CV across Splits) ---")
    
    save_base_dir = "../ds/H2Crop_tiles_unique_subset"
    
    for encoder in encoders:
        for patch_size in patch_sizes:
            
            for split_id in range(1, num_splits + 1):
                split_results_dir = os.path.join(save_results_dir, f"split_{split_id}", f"unet_enc_{encoder}_{patch_size}")
                
                for mod in modalities:
                    dataset_dir = os.path.join(save_base_dir, f"split_{split_id}", f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
                    
                    if os.path.exists(dataset_dir):
                        in_channels = 218 if mod == "hyperspectral" else 10
                        num_classes = len(unique_classes) + 1 # +1 for background
                        model_name = f"unet_enc_{encoder}"
                        
                        print(f"\n[*] Preparing {model_name} for {mod.upper()} | SPLIT {split_id}")
                        
                        # PIPELINE CALL GOES HERE IN THE NEXT STEPS
                        """
                        pipeline_H2Crop_unet_unique(...)
                        """
                    else:
                        print(f"[Error] Dataset directory {dataset_dir} missing. Skipping.")