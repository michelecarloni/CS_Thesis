import os
import sys

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

from H2Crop.H2Crop import H2Crop
from scripts.extract_tiles_unique_sub import extract_unique_sub_tiles
from pipelines.pipeline_H2Crop_standard_ML_unique_subset import pipeline_H2Crop_standard_ML_algo_unique

if __name__ == "__main__":
    # ==========================================
    # HYPERPARAMETERS & CONFIGURATION
    # ==========================================
    modalities = ["hyperspectral", "multispectral"]
    taxonomy = 3
    patch_sizes = [64]
    valid_threshold = 0.10  
    
    # Unified path variables as requested
    save_results_dir = "../results_9"
    save_base_dir = "../ds/H2Crop_tiles_unique_subset"

    # Unified subset configuration: Dictionary with tuple key (common classes) and list value (minority classes)
    unique_classes_dict = {(8, 11, 23, 30, 38, 1, 2): [10, 29, 50]}
    
    common_classes = list(list(unique_classes_dict.keys())[0])
    minority_classes = list(unique_classes_dict.values())[0]
    all_subset_classes = common_classes + minority_classes
    
    # Cross Validation Settings
    num_splits = 3
    apply_mild_balance = True
    
    # Standard ML Training parameters
    train_samples_per_class = 100000
    debug = False 
    
    print("Initializing H2Crop Loader...")
    loader = H2Crop()

    # ==========================================
    # 1. UNIQUE SUBSET EXTRACTION PHASE
    # ==========================================
    # The extraction phase safely skips if the data is already extracted
    extract_unique_sub_tiles(
        patch_sizes=patch_sizes, 
        unique_classes=all_subset_classes, 
        modalities=modalities, 
        loader=loader, 
        taxonomy=taxonomy, 
        valid_threshold=valid_threshold,
        num_splits=num_splits,
        apply_mild_balance=apply_mild_balance
    )

    # ==========================================
    # 2. TRAINING PHASE (ML Baselines)
    # ==========================================
    print("\n--- Starting Training Phase (Standard ML Segmentations - Unique Subset) ---")
    
    for patch_size in patch_sizes:
        for split_id in range(1, num_splits + 1):
            for mod in modalities:
                dataset_dir = os.path.join(save_base_dir, f"split_{split_id}", f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
                
                if os.path.exists(dataset_dir):
                    pipeline_H2Crop_standard_ML_algo_unique(
                        save_results_dir=save_results_dir,
                        dataset_dir=dataset_dir,
                        split_id=split_id,
                        subset_classes=all_subset_classes, 
                        modality=mod,
                        taxonomy=taxonomy,
                        patch_size=patch_size,
                        train_samples_per_class=train_samples_per_class,
                        debug=debug
                    )
                else:
                    print(f"[Error] Dataset directory {dataset_dir} missing. Verify extraction finished successfully.")