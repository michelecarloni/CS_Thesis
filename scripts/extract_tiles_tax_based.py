import os

def extract_tiles_tax_based(modality, save_tile_ds_dir, taxonomy, loader):
    print("\n--- Starting Tile Extraction ---")
    for mod in modality:
        target_dir = os.path.join(save_tile_ds_dir, f"{mod}_taxonomy_{taxonomy}")
        train_subfolder = os.path.join(target_dir, "train")
        
        # Check if the 'train' subfolder exists and has files inside it
        if os.path.exists(train_subfolder) and len(os.listdir(train_subfolder)) > 0:
            print(f"[*] Pre-split data already exists for {mod.upper()}. Skipping extraction. (Found at {target_dir})")
            continue
            
        print(f"[*] No existing splits found for {mod.upper()}. Starting extraction & splitting...")
        
        loader.extract_and_save_tiles(
            save_base_dir=save_tile_ds_dir, 
            modality=mod, 
            taxonomy=taxonomy, 
            patch_size=32,
            min_train_tiles_threshold=10000
        )