import os

def extract_unique_sub_tiles(patch_sizes, unique_classes, modalities, loader, taxonomy, valid_threshold, num_splits, apply_mild_balance):
    print("\n--- Starting Golden Unique Subset Extraction Phase ---")
    save_base_dir = "../ds/H2Crop_tiles_unique_subset"
    
    for patch_size in patch_sizes:
        print(f"\n{'='*60}")
        print(f"STARTING PROCESSING FOR PATCH SIZE: {patch_size}x{patch_size}")
        print(f"Classes: {unique_classes}")
        print(f"Mild Balance (Global): {apply_mild_balance}")
        print(f"{'='*60}")
        
        # We now use a unified extraction method that builds the golden dataset 
        # and distributes it across the splits in one go.
        loader.extract_unique_subset_and_split(
            save_base_dir=save_base_dir,
            subset_classes=unique_classes,
            modalities=modalities,
            taxonomy=taxonomy,
            patch_size=patch_size,
            num_splits=num_splits,
            apply_mild_balance=apply_mild_balance,
            valid_threshold=valid_threshold
        )