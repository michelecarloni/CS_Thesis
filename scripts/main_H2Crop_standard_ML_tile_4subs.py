import os
import sys

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

from H2Crop.H2Crop import H2Crop

from scripts.extract_tiles_4_subs import extract_4_subs_tiles

# Import both the standard baseline and the Optuna-optimized pipeline
from pipelines.pipeline_H2Crop_standard_ML_algo_tiles import pipeline_H2Crop_standard_ML_algo_tiles
from pipelines.pipeline_H2Crop_standard_ML_algo_tiles_optuna import pipeline_H2Crop_standard_ML_algo_tiles_optuna

if __name__ == "__main__":
    # ==========================================
    # HYPERPARAMETERS & CONFIGURATION
    # ==========================================
    modalities = ["hyperspectral", "multispectral"]
    taxonomy = 3
    patch_sizes = [32]
    save_results_dir = "../results_6"
    use_gpu = True
    
    use_optuna = True            
    n_trials = 8                # Number of Optuna trials to run per algorithm
    max_train_pixels = 500000    # Memory safety cap for training data
    debug = False                # Toggle to True for a rapid plumbing test (processes only 10 files)

    train_samples_per_class=100000,
    test_samples_per_class=100000,
    
    subsets = {
        1: [8, 11, 23, 56],
        2: [13, 30, 38, 53, 61],
        3: [1, 2, 10, 17],
        4: [29, 50, 64, 76]
    }
    
    # Initialize the dataset loader
    print("Initializing H2Crop Loader...")
    loader = H2Crop()

    # ==========================================
    # 1. SUBSET EXTRACTION PHASE
    # ==========================================
    extract_4_subs_tiles(patch_sizes, subsets, modalities, loader, taxonomy)

    

    # ==========================================
    # 2. TRAINING PHASE (ML Baselines)
    # ==========================================
    print("\n--- Starting Training Phase (Standard ML Segmentations) ---")
    
    for patch_size in patch_sizes:
        for subset_id in subsets.keys():
            save_base_dir = f"../ds/H2Crop_tiles_ds_subset_{subset_id}"
            
            for mod in modalities:
                dataset_dir = os.path.join(save_base_dir, f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
                
                if os.path.exists(dataset_dir):
                    
                    # Route to the appropriate pipeline based on the toggle
                    if use_optuna:
                        pipeline_H2Crop_standard_ML_algo_tiles_optuna(
                            save_results_dir=save_results_dir,
                            dataset_dir=dataset_dir,
                            subset_id=subset_id,
                            modality=mod,
                            taxonomy=taxonomy,
                            patch_size=patch_size,
                            use_gpu=use_gpu,
                            max_train_pixels=max_train_pixels,
                            n_trials=n_trials,
                            debug=debug
                        )
                    else:
                        pipeline_H2Crop_standard_ML_algo_tiles(
                            save_results_dir=save_results_dir,
                            dataset_dir=dataset_dir,
                            subset_id=subset_id,
                            modality=mod,
                            taxonomy=taxonomy,
                            patch_size=patch_size,
                            use_gpu=use_gpu,
                            train_samples_per_class=train_samples_per_class,
                            test_samples_per_class=test_samples_per_class,
                            test_batch_size=50, 
                            debug=debug
                        )
                else:
                    print(f"[Error] Dataset directory {dataset_dir} missing. Skipping training for this config.")