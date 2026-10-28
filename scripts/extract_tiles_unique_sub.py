import os

def extract_unique_sub_tiles(patch_sizes, unique_classes, modalities, loader, taxonomy, valid_threshold, num_splits, apply_mild_balance):
    print("\n--- Starting Unique Subset Extraction Phase ---")
    save_base_dir = "../ds/H2Crop_tiles_unique_subset"
    
    # Check if the base directory already exists and contains files/folders to skip the entire extraction
    if os.path.exists(save_base_dir) and len(os.listdir(save_base_dir)) > 0:
        print(f"[*] The directory '{save_base_dir}' already exists and is not empty.")
        print("[*] Skipping the entire extraction phase.")
        return
        
    for patch_size in patch_sizes:
        print(f"\n{'='*60}")
        print(f"STARTING PROCESSING FOR PATCH SIZE: {patch_size}x{patch_size}")
        print(f"{'='*60}")
        
        loader.extract_unique_subset_splits(
            save_base_dir=save_base_dir,
            subset_classes=unique_classes,
            modalities=modalities,
            taxonomy=taxonomy,
            patch_size=patch_size,
            num_splits=num_splits,
            apply_mild_balance=apply_mild_balance,
            valid_threshold=valid_threshold
        )