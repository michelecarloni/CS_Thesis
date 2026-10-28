import os

def extract_unique_sub_tiles(patch_sizes, unique_classes, modalities, loader, taxonomy, valid_threshold, num_splits, apply_mild_balance):
    print("\n--- Starting Unique Subset Extraction Phase ---")
    save_base_dir = "../ds/H2Crop_tiles_unique_subset"
    
    for patch_size in patch_sizes:
        print(f"\n{'='*60}")
        print(f"STARTING PROCESSING FOR PATCH SIZE: {patch_size}x{patch_size}")
        print(f"{'='*60}")
        
        for split_id in range(1, num_splits + 1):
            print(f"\n{'='*50}")
            print(f"PROCESSING SPLIT {split_id}/{num_splits} (Patch Size: {patch_size})")
            print(f"{'='*50}")
            
            split_base_dir = os.path.join(save_base_dir, f"split_{split_id}")
            
            for mod in modalities:
                final_save_dir = os.path.join(split_base_dir, f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
                summary_file_path = os.path.join(final_save_dir, "split_summary.txt")
                
                # Granular check: Skip only if this exact split and modality is already completed
                if os.path.exists(final_save_dir) and os.path.exists(summary_file_path):
                    print(f"[*] {mod.upper()} data for Split {split_id} (pSize: {patch_size}) already exists. Skipping extraction.")
                else:
                    print(f"[*] No existing data found for {mod.upper()} Split {split_id}. Starting extraction...")
                    
                    loader.extract_unique_subset_single_split(
                        split_base_dir=split_base_dir,
                        split_id=split_id,
                        subset_classes=unique_classes,
                        modality=mod,
                        taxonomy=taxonomy,
                        patch_size=patch_size,
                        apply_mild_balance=apply_mild_balance,
                        valid_threshold=valid_threshold
                    )