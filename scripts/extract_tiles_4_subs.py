import os

def extract_4_subs_tiles(patch_sizes, subsets, modalities, loader, taxonomy, valid_threshold):
    print("\n--- Starting Subset Extraction Phase ---")
    for patch_size in patch_sizes:
        print(f"\n{'='*60}")
        print(f"STARTING PROCESSING FOR PATCH SIZE: {patch_size}x{patch_size}")
        print(f"{'='*60}")
        
        # UNPACK THE TUPLE HERE
        for subset_id, (subset_classes, minority_classes) in subsets.items():
            print(f"\n{'='*50}")
            print(f"PROCESSING SUBSET {subset_id}: {subset_classes} (Patch Size: {patch_size})")
            print(f"{'='*50}")
            
            save_base_dir = f"../ds/H2Crop_tiles_ds_subset_{subset_id}"
            
            for mod in modalities:
                final_save_dir = os.path.join(save_base_dir, f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
                summary_file_path = os.path.join(final_save_dir, "split_summary.txt")
                
                train_dir = os.path.join(final_save_dir, "train")
                val_dir = os.path.join(final_save_dir, "validation")
                test_dir = os.path.join(final_save_dir, "test")
                
                # If tiles were already extracted into train/val/test, backfill split_summary.txt and skip!
                if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0 and not os.path.exists(summary_file_path):
                    n_train = len([f for f in os.listdir(train_dir) if f.endswith('.npz')])
                    n_val = len([f for f in os.listdir(val_dir) if f.endswith('.npz')]) if os.path.exists(val_dir) else 0
                    n_test = len([f for f in os.listdir(test_dir) if f.endswith('.npz')]) if os.path.exists(test_dir) else 0
                    n_total = n_train + n_val + n_test
                    
                    with open(summary_file_path, "w") as f:
                        f.write("--- Shared 70/20/10 Extraction Summary ---\n")
                        f.write(f"Modality: {mod}\n")
                        f.write(f"Subset Classes: {subset_classes}\n")
                        f.write(f"Total Tiles Extracted: {n_total}\n\n")
                        f.write("--- Tile Distribution ---\n")
                        f.write(f"Train Tiles: {n_train}\n")
                        f.write(f"Validation Tiles: {n_val}\n")
                        f.write(f"Test Tiles: {n_test}\n")
                    print(f"[*] Generated missing split_summary.txt for existing {mod.upper()} Subset {subset_id} ({n_total} tiles).")

                # Standard check
                if os.path.exists(final_save_dir) and os.path.exists(summary_file_path):
                    print(f"[*] {mod.upper()} data for Subset {subset_id} (pSize: {patch_size}) already exists. Skipping extraction.")
                else:
                    print(f"[*] No existing splits found for {mod.upper()} Subset {subset_id} (pSize: {patch_size}). Starting extraction...")
                    
                    loader.extract_and_save_tiles_subset(
                        save_base_dir=save_base_dir, 
                        subset_classes=subset_classes, 
                        modality=mod, 
                        taxonomy=taxonomy, 
                        patch_size=patch_size,
                        valid_threshold=valid_threshold
                    )