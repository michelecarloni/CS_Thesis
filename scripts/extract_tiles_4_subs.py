import os

def extract_4_subs_tiles(patch_sizes, subsets, modalities, loader, taxonomy):
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