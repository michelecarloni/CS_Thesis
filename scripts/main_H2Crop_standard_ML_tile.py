import os
import sys

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

from H2Crop.H2Crop import H2Crop
from pipelines import pipeline_H2Crop_standard_ML_algo_tiles

if __name__ == "__main__":
    # Define the parameters and subsets
    modalities = ["hyperspectral", "multispectral"]
    taxonomy = 3
    patch_sizes = [32]
    save_results_dir = "../results_5"
    use_gpu = True
    
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
    print("\n--- Starting Subset Extraction Phase ---")
    
    # Outer loop for iterating over different patch sizes
    for patch_size in patch_sizes:
        print(f"\n{'='*60}")
        print(f"STARTING PROCESSING FOR PATCH SIZE: {patch_size}x{patch_size}")
        print(f"{'='*60}")
        
        for subset_id, subset_classes in subsets.items():
            print(f"\n{'='*50}")
            print(f"PROCESSING SUBSET {subset_id}: {subset_classes} (Patch Size: {patch_size})")
            print(f"{'='*50}")
            
            # Define the base directory for this specific subset
            save_base_dir = f"../ds/H2Crop_tiles_ds_subset_{subset_id}"
            
            for mod in modalities:
                # Reconstruct the exact final directory path that the function will create
                final_save_dir = os.path.join(save_base_dir, f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
                summary_file_path = os.path.join(final_save_dir, "split_summary.txt")
                
                # Check if this specific extraction already completed successfully
                if os.path.exists(final_save_dir) and os.path.exists(summary_file_path):
                    print(f"[*] {mod.upper()} data for Subset {subset_id} (pSize: {patch_size}) already exists. Skipping extraction.")
                else:
                    print(f"[*] No existing splits found for {mod.upper()} Subset {subset_id} (pSize: {patch_size}). Starting extraction...")
                    
                    # Call the new extraction function
                    loader.extract_and_save_tiles_subset(
                        save_base_dir=save_base_dir, 
                        subset_classes=subset_classes, 
                        modality=mod, 
                        taxonomy=taxonomy, 
                        patch_size=patch_size
                    )

    # ==========================================
    # 3. TRAINING PHASE (ML Baselines)
    # ==========================================
    print("\n--- Starting Training Phase (Standard ML Segmentations) ---")
    
    for patch_size in patch_sizes:
        for subset_id in subsets.keys():
            save_base_dir = f"../ds/H2Crop_tiles_ds_subset_{subset_id}"
            
            for mod in modalities:
                dataset_dir = os.path.join(save_base_dir, f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
                
                if os.path.exists(dataset_dir):
                    pipeline_H2Crop_standard_ML_algo_tiles(
                        save_results_dir=save_results_dir,
                        dataset_dir=dataset_dir,
                        subset_id=subset_id,
                        modality=mod,
                        taxonomy=taxonomy,
                        patch_size=patch_size,
                        use_gpu=use_gpu
                    )
                else:
                    print(f"[Error] Dataset directory {dataset_dir} missing. Skipping training for this config.")