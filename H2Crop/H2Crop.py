import os
import h5py
import numpy as np
from tqdm import tqdm
import random
import gc
from scipy.stats import mode
from contextlib import redirect_stdout
import shutil
from sklearn.model_selection import train_test_split


class H2Crop:
    def __init__(self, dataset_path='/media/michele/T7/datasets/H2Crop'):
        """
        Initializes the H2Crop dataset loader.
        """
        self.dataset_path = dataset_path
        self.h5_dir = os.path.join(self.dataset_path, 'h5_data')
        
        # Verify the directory exists upon initialization
        if not os.path.exists(self.h5_dir):
            print(f"Warning: Directory not found at {self.h5_dir}")

    def extract_H2Crop_pixel_sample(self, save_dir, file_chunks_list, modality, class_action="drop", class_list=None, detail_layer=0, static=True, keep_prior=False, samples_per_class=25000):
        """
        Extracts, balances, and saves pixel-wise samples to disk to prevent redundant processing.
        
        Parameters:
        - save_dir: Directory where the extracted .npz file will be saved.
        - file_chunks_list: List of file chunks to process.
        - modality: "hyperspectral" or "multispectral".
        - class_action: "drop" or "keep".
        - class_list: List of class IDs to filter.
        - detail_layer: Taxonomy detail level (e.g., 3).
        - static: Boolean, whether to extract static snapshot or temporal data.
        - keep_prior: Boolean, whether to append prior data.
        - samples_per_class: Target number of samples per class.
        
        Returns:
        - str: Path to the saved .npz file.
        """
        
        
        if modality.lower() not in ["hyperspectral", "multispectral"]:
            print('Extraction aborted: modality is neither "Hyperspectral" nor "Multispectral"')
            return None

        class_action = class_action.lower()
        if class_action not in ["drop", "keep"]:
            print('Extraction aborted: class_action must be either "drop" or "keep"')
            return None

        os.makedirs(save_dir, exist_ok=True)
        save_file_path = os.path.join(save_dir, f"{modality}_extracted_pixels.npz")
        
        # Check if it already exists to avoid accidental overwrites
        if os.path.exists(save_file_path):
            print(f"\n[Info] Data already extracted for {modality} at {save_file_path}.")
            print("To re-extract, delete the existing file first.")
            return save_file_path

        print(f"\n{'='*60}")
        print(f"STARTING PIXEL EXTRACTION: {modality.upper()}")
        print(f"Target: {samples_per_class} samples per class")
        print(f"{'='*60}")

        collected_X = {}
        class_counts = {}
        target_classes = set(class_list) if (class_action == "keep" and class_list is not None) else None
        files_processed = 0

        # Mute inner prints to keep tqdm clean
        with open(os.devnull, 'w') as devnull:
            for current_chunk in tqdm(file_chunks_list, desc=f"Extracting ({modality})", unit="chunk", position=0, leave=True):
                files_processed += len(current_chunk)
                
                with redirect_stdout(devnull):
                    batch = self.load_h5_data(
                        file_list=current_chunk, 
                        detail_layer=detail_layer, 
                        static=static, 
                        data_type=modality, 
                        keep_prior=keep_prior
                    )
                
                if not batch:
                    continue
                    
                for sample in batch:
                    X_img = sample[modality]
                    y_img = sample['labels']
                    
                    if modality == "hyperspectral":
                        X_img = self.upsample_hyperspectral(X_img)
                        
                    X_img = np.transpose(X_img, (1, 2, 0))
                    X_flat = X_img.reshape(-1, X_img.shape[-1])
                    y_flat = y_img.reshape(-1)
                    
                    if keep_prior and 'prior' in sample:
                        prior_flat = sample['prior'].reshape(-1, 1)
                        X_flat = np.hstack((X_flat, prior_flat))
                        
                    if class_list is not None:
                        if class_action == "keep":
                            valid_mask = np.isin(y_flat, class_list)
                        elif class_action == "drop":
                            valid_mask = ~np.isin(y_flat, class_list)
                            
                        X_flat = X_flat[valid_mask]
                        y_flat = y_flat[valid_mask]

                    if len(y_flat) == 0:
                        continue
                        
                    unique_classes_in_patch = np.unique(y_flat)
                    for class_id in unique_classes_in_patch:
                        if class_counts.get(class_id, 0) >= samples_per_class:
                            continue
                            
                        mask = (y_flat == class_id)
                        pixels_to_add = X_flat[mask]
                        
                        if class_id not in collected_X:
                            collected_X[class_id] = []
                            class_counts[class_id] = 0
                            
                        collected_X[class_id].append(pixels_to_add)
                        class_counts[class_id] += len(pixels_to_add)
                        
                del batch
                gc.collect()

                # Termination check
                if target_classes:
                    if all(class_counts.get(c, 0) >= samples_per_class for c in target_classes):
                        print(f"\n--> Success! Target of {samples_per_class} samples reached for target classes.")
                        break
                else:
                    if len(class_counts) > 0 and all(count >= samples_per_class for count in class_counts.values()):
                        print(f"\n--> Success! Target of {samples_per_class} samples reached for all discovered classes.")
                        break

        print(f"\nExtraction complete. Processed {files_processed} files.")

        # Balancing Phase
        X_final_list, y_final_list = [], []
        print(f"\n--- Final Class Distribution (Target: {samples_per_class}) ---")
        
        for class_id, arrs in collected_X.items():
            X_c = np.vstack(arrs)
            total_found = len(X_c)
            
            if total_found > samples_per_class:
                idx = np.random.choice(total_found, samples_per_class, replace=False)
                X_c = X_c[idx]
                print(f" - Class {class_id}: Found {total_found} -> Downsampled to {samples_per_class}")
            else:
                print(f" - Class {class_id}: Found {total_found} -> Kept {total_found} (WARNING: Under target!)")
                
            X_final_list.append(X_c)
            y_final_list.append(np.full(len(X_c), class_id, dtype=np.int32))
            
        del collected_X
        gc.collect()
            
        if not X_final_list:
            print("Extraction aborted: No valid pixels found.")
            return None
            
        X = np.vstack(X_final_list)
        y = np.concatenate(y_final_list)

        print(f"\nSaving arrays to disk: {X.shape[0]} total pixels...")
        
        # Save compressed numpy arrays to save disk space and read time
        np.savez_compressed(save_file_path, X=X, y=y)
        print(f"Successfully saved to: {save_file_path}")
        
        return save_file_path


    def extract_global_subset_and_split(self, save_base_dir, subset_classes, modalities, taxonomy=3, patch_size=64, num_splits=3, apply_mild_balance=True, valid_threshold=0.10, max_files=None, max_workers=None):
        """
        Creates a 'Golden Dataset' pool of tiles (globally mild-balanced if requested),
        syncs it across modalities, and then performs a strict tile-level cross-validation 
        split so every fold uses the exact same global pool of tiles.
        """
        import os
        import json
        import shutil
        import numpy as np
        from tqdm import tqdm
        import concurrent.futures
        from functools import partial
        import h5py
        from scipy.stats import mode
        from sklearn.model_selection import KFold

        # Setup Golden Pool Directories
        golden_dir = os.path.join(save_base_dir, "golden_pool")
        os.makedirs(golden_dir, exist_ok=True)
        
        h5_files = [f for f in os.listdir(self.h5_dir) if f.endswith('.h5')]
        if not h5_files:
            print(f"[Warning] No .h5 files found in {self.h5_dir}")
            return
            
        if max_files is not None:
            h5_files = h5_files[:max_files]

        # Ensure hyperspectral is processed first to drive the balancing
        if modalities[0] != "hyperspectral" and "hyperspectral" in modalities:
            modalities.remove("hyperspectral")
            modalities.insert(0, "hyperspectral")

        shared_balance_path = os.path.join(golden_dir, f"golden_balanced_tiles_pSize_{patch_size}.json")

        for mod in modalities:
            mod_golden_dir = os.path.join(golden_dir, f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
            
            if os.path.exists(mod_golden_dir) and len(os.listdir(mod_golden_dir)) > 0:
                print(f"[*] Golden pool for {mod.upper()} already extracted. Skipping.")
            else:
                os.makedirs(mod_golden_dir, exist_ok=True)
                
                # We reuse the shared extraction logic, but without routing to train/val/test yet.
                # We map all valid files to "golden" and pass the mod_golden_dir as the target.
                dummy_file_to_split = {f: "golden" for f in h5_files}
                
                worker_func = partial(
                    self._process_single_h5_file_shared, 
                    taxonomy=taxonomy, 
                    modality=mod, 
                    patch_size=patch_size, 
                    subset_classes=subset_classes, 
                    train_dir=mod_golden_dir, # All routed here
                    val_dir=mod_golden_dir,
                    test_dir=mod_golden_dir,
                    file_to_split=dummy_file_to_split,
                    valid_threshold=valid_threshold
                )

                with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                    list(tqdm(
                        executor.map(worker_func, h5_files), 
                        total=len(h5_files), 
                        desc=f"Extracting Global Pool ({mod.upper()})"
                    ))
                    
                # 2. Global Mild Balancing & Modality Syncing
                if apply_mild_balance:
                    if os.path.exists(shared_balance_path):
                        print(f"[*] Syncing {mod.upper()} golden pool to primary modality balance...")
                        with open(shared_balance_path, 'r') as f:
                            kept_basenames = set(json.load(f))
                            
                        deleted_sync = 0
                        for f_name in os.listdir(mod_golden_dir):
                            if f_name.endswith('.npz') and f_name not in kept_basenames:
                                os.remove(os.path.join(mod_golden_dir, f_name))
                                deleted_sync += 1
                        print(f"[*] Sync complete. Removed {deleted_sync} tiles.")
                    else:
                        print(f"[*] Applying Global Mild Balancing to {mod.upper()} pool...")
                        self._mild_balance_golden_pool(mod_golden_dir, subset_classes)
                        
                        # Save the balanced list so subsequent modalities replicate this exact drop
                        kept_basenames = [f for f in os.listdir(mod_golden_dir) if f.endswith('.npz')]
                        with open(shared_balance_path, 'w') as f:
                            json.dump(kept_basenames, f, indent=4)

        # 3. Cross-Validation Tile-Level Routing
        print("\n--- Initiating Strict Tile-Level Cross-Validation Routing ---")
        
        with open(shared_balance_path, 'r') as f:
            golden_tiles = np.array(json.load(f))
            
        kf = KFold(n_splits=num_splits, shuffle=True, random_state=42)
        splits = list(kf.split(golden_tiles))
        
        for split_id, (train_val_idx, test_idx) in enumerate(splits, 1):
            split_base = os.path.join(save_base_dir, f"split_{split_id}")
            
            # Further split train_val into train (80% of remaining) and val (20% of remaining)
            from sklearn.model_selection import train_test_split
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.20, random_state=42 + split_id)
            
            split_map = {
                "train": set(golden_tiles[train_idx]),
                "validation": set(golden_tiles[val_idx]),
                "test": set(golden_tiles[test_idx])
            }
            
            for mod in modalities:
                mod_golden_dir = os.path.join(golden_dir, f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
                final_save_dir = os.path.join(split_base, f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
                
                if os.path.exists(os.path.join(final_save_dir, "split_summary.txt")):
                    continue
                    
                print(f"[*] Populating {mod.upper()} Split {split_id}...")
                
                for subset_name in ["train", "validation", "test"]:
                    target_dir = os.path.join(final_save_dir, subset_name)
                    os.makedirs(target_dir, exist_ok=True)
                    
                    # Copy files into place (symlinks can break depending on OS/File System)
                    for tile_name in split_map[subset_name]:
                        src = os.path.join(mod_golden_dir, tile_name)
                        dst = os.path.join(target_dir, tile_name)
                        shutil.copy2(src, dst)
                        
                # Summary logic...
                with open(os.path.join(final_save_dir, "split_summary.txt"), "w") as f:
                    f.write(f"--- Golden Pool CV Summary (Split {split_id}) ---\n")
                    f.write(f"Total Tiles: {len(golden_tiles)}\n")
                    f.write(f"Train Tiles: {len(train_idx)}\n")
                    f.write(f"Validation Tiles: {len(val_idx)}\n")
                    f.write(f"Test Tiles: {len(test_idx)}\n")


    def _mild_balance_golden_pool(self, golden_dir, subset_classes):
        """
        Balances the entire global pool.
        """
        import os
        import numpy as np
        import random
        from scipy.stats import mode
        from tqdm import tqdm
        
        class_to_files = {c: [] for c in subset_classes}
        files = [os.path.join(golden_dir, f) for f in os.listdir(golden_dir) if f.endswith('.npz')]
        
        for fpath in tqdm(files, desc="Scanning Global Pool"):
            with np.load(fpath) as data:
                y = data['y']
                crop_pixels = y[np.isin(y, subset_classes)]
                if len(crop_pixels) > 0:
                    predominant = int(mode(crop_pixels, keepdims=False)[0])
                    if predominant in class_to_files:
                        class_to_files[predominant].append(fpath)
                        
        counts = {c: len(class_to_files[c]) for c in subset_classes}
        valid_counts = [c for c in counts.values() if c > 0]
        mean_count = int(np.mean(valid_counts)) if valid_counts else 0
        
        deleted = 0
        random.seed(42)
        
        for c in subset_classes:
            if counts[c] > mean_count:
                drop_count = counts[c] - mean_count
                random.shuffle(class_to_files[c])
                
                for fpath in class_to_files[c][:drop_count]:
                    os.remove(fpath)
                deleted += drop_count
                
        print(f"Global balance applied. Capped at {mean_count}. Removed {deleted} tiles.")

    
    def extract_and_save_tiles(self, save_base_dir, modality="hyperspectral", taxonomy=3, patch_size=32, max_files=None, min_train_tiles_threshold=20):
        """
        Iterates through the h5_data directory, extracts spatial patches,
        calculates the predominant class for each, and saves them temporarily.
        Triggers per-class splitting, filtering, and organization.
        
        Parameters:
        - save_base_dir: string, base directory to save the extracted tiles
        - modality: string, "hyperspectral" or "multispectral"
        - taxonomy: int, defines the taxonomy level for labels
        - patch_size: int, size of square tile
        - max_files: int or None, limit files for quick testing
        - min_train_tiles_threshold: int, minimum number of training tiles required to keep a class
        
        Returns:
        - list of int: Sorted list of class IDs that met the threshold and were kept.
        """
        final_save_dir = os.path.join(save_base_dir, f"{modality}_taxonomy_{taxonomy}")
        os.makedirs(final_save_dir, exist_ok=True)
        
        h5_files = [f for f in os.listdir(self.h5_dir) if f.endswith('.h5')]
        if not h5_files:
            print(f"[Warning] No .h5 files found in {self.h5_dir}")
            return []
        
        print(f"Found {len(h5_files)} images in {self.h5_dir}.")
        print(f"Extracting {patch_size}x{patch_size} tiles to: {final_save_dir}")
        if max_files is not None:
            print(f"[Testing Mode] Execution limited to the first {max_files} files.")
            
        class_to_files = {}
        files_processed = 0
        
        for filename in tqdm(h5_files, desc=f"Extracting Tiles ({modality})"):
            if max_files is not None and files_processed >= max_files:
                print(f"\nReached testing limit of {max_files} files. Halting extraction.")
                break
                
            file_path = os.path.join(self.h5_dir, filename)
            sample_id = filename.replace('.h5', '')
            
            try:
                with h5py.File(file_path, 'r') as h5f:
                    labels_full = np.array(h5f['label'])
                    mask_array = labels_full[taxonomy]
                    
                    if modality.lower() == "hyperspectral":
                        image_array = np.array(h5f['EnMAP_data'])
                        image_array = self.upsample_hyperspectral(image_array)
                    elif modality.lower() == "multispectral":
                        s2_full = np.array(h5f['S2_data'])
                        month_str = filename[4:6]
                        s2_time_index = int(month_str) - 1
                        image_array = s2_full[s2_time_index]
                    else:
                        raise ValueError("Modality must be 'hyperspectral' or 'multispectral'")
                    
                    _, h, w = image_array.shape
                    
                    tile_idx = 0
                    for i in range(0, h - patch_size + 1, patch_size):
                        for j in range(0, w - patch_size + 1, patch_size):
                            img_patch = image_array[:, i:i+patch_size, j:j+patch_size]
                            mask_patch = mask_array[i:i+patch_size, j:j+patch_size]
                            
                            predominant_class = int(mode(mask_patch.flatten(), keepdims=False)[0])
                            
                            # Save temporarily in the base directory
                            tile_filename = os.path.join(final_save_dir, f"{sample_id}_tile_{tile_idx}.npz")
                            np.savez_compressed(tile_filename, X=img_patch, y=predominant_class)
                            
                            if predominant_class not in class_to_files:
                                class_to_files[predominant_class] = []
                            class_to_files[predominant_class].append(tile_filename)
                            
                            tile_idx += 1
                            
                files_processed += 1
                            
            except Exception as e:
                print(f"\n[Error] Failed to process {filename}: {str(e)}")
                continue
                    
        print(f"\nExtraction complete! Grouped tiles across {len(class_to_files)} classes.")
        
        # Trigger per-class split, threshold filtering, and organization
        kept_classes = self._split_and_move_tiles(class_to_files, final_save_dir, min_train_tiles_threshold)
        return kept_classes


    def _split_and_move_tiles(self, class_to_files_map, base_dir, min_train_tiles_threshold):
        """
        Performs a per-class train/val/test split (70/20/10), filters out classes 
        whose training set size falls below the threshold, deletes discarded files, 
        and moves valid files into structured subfolders.
        """
        print(f"\n--- Processing per-class splits and applying threshold (min_train={min_train_tiles_threshold}) ---")
        
        train_dir = os.path.join(base_dir, "train")
        val_dir = os.path.join(base_dir, "validation")
        test_dir = os.path.join(base_dir, "test")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        kept_classes = []
        dropped_classes = []
        
        for class_id, file_paths in class_to_files_map.items():
            total_tiles = len(file_paths)
            
            # If a class has fewer than 10 tiles overall, it cannot form a valid 70/20/10 split
            if total_tiles < 10:
                dropped_classes.append((class_id, total_tiles, "Too few total tiles"))
                for fp in file_paths:
                    if os.path.exists(fp):
                        os.remove(fp)
                continue
                
            # Per-class split: 70% train, 20% val, 10% test
            try:
                train_files, test_files = train_test_split(file_paths, test_size=0.10, random_state=42)
                val_ratio = 0.20 / 0.90
                train_files, val_files = train_test_split(train_files, test_size=val_ratio, random_state=42)
            except Exception as e:
                dropped_classes.append((class_id, total_tiles, f"Split error: {str(e)}"))
                for fp in file_paths:
                    if os.path.exists(fp):
                        os.remove(fp)
                continue
                
            # Apply training set threshold filter
            if len(train_files) < min_train_tiles_threshold:
                dropped_classes.append((class_id, len(train_files), f"Below train threshold ({min_train_tiles_threshold})"))
                for fp in file_paths:
                    if os.path.exists(fp):
                        os.remove(fp)
                continue
                
            kept_classes.append(class_id)
            
            # Move files to their respective split directories
            def move_files(files, target_subfolder):
                for fp in files:
                    filename = os.path.basename(fp)
                    dest = os.path.join(target_subfolder, filename)
                    shutil.move(fp, dest)
                    
            move_files(train_files, train_dir)
            move_files(val_files, val_dir)
            move_files(test_files, test_dir)
            
        print(f" -> Kept {len(kept_classes)} classes meeting the threshold.")
        print(f" -> Dropped {len(dropped_classes)} rare classes.")
        for dc in dropped_classes[:5]:
            print(f"    - Class {dc[0]}: {dc[1]} training tiles ({dc[2]})+")
        if len(dropped_classes) > 5:
            print(f"    - ... and {len(dropped_classes) - 5} more.")
            
        return sorted(kept_classes)


    def extract_and_save_tiles_subset(self, save_base_dir, subset_classes, modality="hyperspectral", taxonomy=3, patch_size=64, max_files=None, max_workers=None, valid_threshold=0.10):
        """
        Extracts segmentation tiles ensuring a shared train/val/test split across modalities
        by splitting at the .h5 sample ID level first. Then, it applies a mild balancing 
        to the train set, forcing secondary modalities (e.g., multispectral) to perfectly 
        mirror the exact tile drops of the primary modality (hyperspectral).
        """
        import concurrent.futures
        from functools import partial
        import json
        import os
        from tqdm import tqdm
        
        final_save_dir = os.path.join(save_base_dir, f"{modality}_taxonomy_{taxonomy}_pSize_{patch_size}")
        train_dir = os.path.join(final_save_dir, "train")
        val_dir = os.path.join(final_save_dir, "validation")
        test_dir = os.path.join(final_save_dir, "test")
        
        for d in [train_dir, val_dir, test_dir]:
            os.makedirs(d, exist_ok=True)
            
        # Path for the shared split map at the subset base level so both modalities share it
        shared_split_path = os.path.join(save_base_dir, "shared_split_mapping.json")
        # Path to sync the exact train tiles kept after mild balancing
        shared_balance_path = os.path.join(save_base_dir, f"shared_balanced_train_pSize_{patch_size}.json")
        
        h5_files = [f for f in os.listdir(self.h5_dir) if f.endswith('.h5')]
        if not h5_files:
            print(f"[Warning] No .h5 files found in {self.h5_dir}")
            return
            
        if max_files is not None:
            h5_files = h5_files[:max_files]
            
        # 1. Create or load the shared file-level split (70/20/10) across modalities
        if os.path.exists(shared_split_path):
            print(f"[*] Found existing shared split mapping at {shared_split_path}. Loading splits...")
            with open(shared_split_path, 'r') as f:
                split_mapping = json.load(f)
        else:
            print(f"[*] Creating new 70/20/10 shared split mapping for all .h5 files...")
            import random
            random.seed(42)
            shuffled_files = sorted([f for f in h5_files])
            random.shuffle(shuffled_files)
            
            n_total = len(shuffled_files)
            train_end = int(n_total * 0.70)
            val_end = int(n_total * 0.90)
            
            split_mapping = {
                "train": shuffled_files[:train_end],
                "validation": shuffled_files[train_end:val_end],
                "test": shuffled_files[val_end:]
            }
            os.makedirs(save_base_dir, exist_ok=True)
            with open(shared_split_path, 'w') as f:
                json.dump(split_mapping, f, indent=4)
            print(f"[*] Saved shared split mapping to {shared_split_path}")

        file_to_split = {}
        for split_name, filenames in split_mapping.items():
            for fname in filenames:
                file_to_split[fname] = split_name

        print(f"\n--- Extracting {patch_size}x{patch_size} Segmentation Tiles ({modality.upper()}) ---")
        print(f"Target Subset Classes: {subset_classes}")
        print(f"Saving directly into shared split folders under: {final_save_dir}")
        
        worker_func = partial(
            self._process_single_h5_file_shared, 
            taxonomy=taxonomy, 
            modality=modality, 
            patch_size=patch_size, 
            subset_classes=subset_classes, 
            train_dir=train_dir,
            val_dir=val_dir,
            test_dir=test_dir,
            file_to_split=file_to_split,
            valid_threshold=valid_threshold
        )
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(worker_func, h5_files), 
                total=len(h5_files), 
                desc=f"Extracting & Routing ({modality})"
            ))
            
        # CROSS-MODALITY SYNC FOR BALANCED TRAIN SET
        if os.path.exists(shared_balance_path):
            print(f"[*] Found shared balanced train list. Syncing {modality} train set to match exactly...")
            with open(shared_balance_path, 'r') as f:
                kept_train_basenames = set(json.load(f))
                
            deleted_sync = 0
            for f_name in os.listdir(train_dir):
                if f_name.endswith('.npz') and f_name not in kept_train_basenames:
                    os.remove(os.path.join(train_dir, f_name))
                    deleted_sync += 1
            print(f"[*] Sync complete. Removed {deleted_sync} tiles to mirror the primary modality.")
        else:
            # Post-process the train directory to apply mild balancing (first modality only)
            self._mild_balance_train_dir(train_dir, val_dir, subset_classes)
            
            # Save the resulting valid basenames to force the next modality to match
            kept_train_basenames = [f for f in os.listdir(train_dir) if f.endswith('.npz')]
            with open(shared_balance_path, 'w') as f:
                json.dump(kept_train_basenames, f, indent=4)
            print(f"[*] Saved balanced train list to {shared_balance_path} for cross-modality syncing.")
            
        # 2. Count extracted tiles per split and write split_summary.txt
        n_train = len([f for f in os.listdir(train_dir) if f.endswith('.npz')])
        n_val = len([f for f in os.listdir(val_dir) if f.endswith('.npz')])
        n_test = len([f for f in os.listdir(test_dir) if f.endswith('.npz')])
        total_extracted = n_train + n_val + n_test
        
        print(f"\nExtraction complete for {modality}! Total Valid Tiles: {total_extracted}")
        print(f" -> Train: {n_train} tiles")
        print(f" -> Validation: {n_val} tiles")
        print(f" -> Test: {n_test} tiles")

        summary_path = os.path.join(final_save_dir, "split_summary.txt")
        with open(summary_path, "w") as f:
            f.write("--- Shared 70/20/10 Extraction Summary ---\n")
            f.write(f"Modality: {modality}\n")
            f.write(f"Subset Classes: {subset_classes}\n")
            f.write(f"Total Tiles Extracted: {total_extracted}\n\n")
            f.write("--- Tile Distribution ---\n")
            f.write(f"Train Tiles: {n_train}\n")
            f.write(f"Validation Tiles: {n_val}\n")
            f.write(f"Test Tiles: {n_test}\n")

        print(f"Summary saved to: {summary_path}")


    def _mild_balance_train_dir(self, train_dir, val_dir, subset_classes):
        """
        Mildly balances the train set by capping majority classes to the mean class count.
        Enforces a 'validation floor' so train samples are always > val samples.
        """
        import os
        import numpy as np
        import random
        from scipy.stats import mode
        from tqdm import tqdm
        
        print("\n--- Applying Mild Balancing (Soft Cap) to Train Set ---")
        
        def get_predominant_class_counts(directory):
            class_to_files = {c: [] for c in subset_classes}
            files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npz')]
            
            for fpath in tqdm(files, desc=f"Scanning {os.path.basename(directory)}"):
                with np.load(fpath) as data:
                    y = data['y']
                    # Isolate only the pixels belonging to the target crops (ignore background)
                    crop_pixels = y[np.isin(y, subset_classes)]
                    if len(crop_pixels) > 0:
                        predominant = int(mode(crop_pixels, keepdims=False)[0])
                        # Only track it if the predominant class is actually one of our targets
                        if predominant in class_to_files:
                            class_to_files[predominant].append(fpath)
            return class_to_files

        # 1. Count validation tiles to establish the floor
        val_class_to_files = get_predominant_class_counts(val_dir)
        val_counts = {c: len(val_class_to_files[c]) for c in subset_classes}
        
        # 2. Count train tiles 
        train_class_to_files = get_predominant_class_counts(train_dir)
        train_counts = {c: len(train_class_to_files[c]) for c in subset_classes}
        
        # 3. Calculate the mild cap (Mean of train counts)
        valid_train_counts = [count for count in train_counts.values() if count > 0]
        mean_train_count = int(np.mean(valid_train_counts)) if valid_train_counts else 0
        
        deleted_total = 0
        
        for c in subset_classes:
            t_count = train_counts[c]
            v_count = val_counts.get(c, 0)
            
            # The cap is the mean, but strictly >= 2x the validation size
            soft_cap = max(mean_train_count, int(v_count * 2.0))
            
            if t_count > soft_cap:
                files_to_drop = t_count - soft_cap
                print(f" -> Class {c}: Downsampling from {t_count} to {soft_cap} (Val size: {v_count})")
                
                # Shuffle deterministically and delete the excess files
                random.seed(42)
                random.shuffle(train_class_to_files[c])
                drop_list = train_class_to_files[c][:files_to_drop]
                
                for fpath in drop_list:
                    os.remove(fpath)
                    
                deleted_total += files_to_drop
            else:
                print(f" -> Class {c}: Kept all {t_count} tiles (Below cap {soft_cap}. Val size: {v_count})")
                
        print(f"\nMild balancing complete. Removed {deleted_total} excessive majority tiles from the train set.\n")


    def _process_single_h5_file_shared(self, filename, taxonomy, modality, patch_size, subset_classes, train_dir, val_dir, test_dir, file_to_split, valid_threshold=0.10):
        """
        Processes a single .h5 file, downscales multispectral/hyperspectral correctly, 
        and saves tiles directly into the shared train/val/test split folders.
        """
        import h5py
        import numpy as np
        from scipy.stats import mode
        
        target_split = file_to_split.get(filename, None)
        if target_split is None:
            return 0  # File not in split mapping
            
        if target_split == "train":
            dest_dir = train_dir
        elif target_split == "validation":
            dest_dir = val_dir
        else:
            dest_dir = test_dir
            
        file_path = os.path.join(self.h5_dir, filename)
        sample_id = filename.replace('.h5', '')
        extracted_count = 0
        
        try:
            with h5py.File(file_path, 'r') as h5f:
                labels_full = np.array(h5f['label'])
                mask_192 = labels_full[taxonomy]  # Shape: (192, 192)
                
                if modality.lower() == "hyperspectral":
                    image_array = np.array(h5f['EnMAP_data'], dtype=np.float32) # (218, 64, 64)
                elif modality.lower() == "multispectral":
                    s2_full = np.array(h5f['S2_data'], dtype=np.float32)
                    month_str = filename[4:6]
                    s2_time_index = int(month_str) - 1
                    s2_month = s2_full[s2_time_index] # (10, 192, 192)
                    
                    c = s2_month.shape[0]
                    image_array = s2_month.reshape(c, 64, 3, 64, 3).mean(axis=(2, 4))
                else:
                    raise ValueError("Modality must be 'hyperspectral' or 'multispectral'")
                
                # Downscale mask (192, 192) -> (64, 64) via 3x3 majority vote
                if mask_192.shape[0] == 192 and image_array.shape[1] == 64:
                    mask_blocks = mask_192.reshape(64, 3, 64, 3).transpose(0, 2, 1, 3).reshape(64, 64, 9)
                    mask_array = mode(mask_blocks, axis=-1, keepdims=False)[0].astype(np.int64)
                else:
                    mask_array = mask_192.astype(np.int64)
                
                _, h, w = image_array.shape
                
                tile_idx = 0
                for i in range(0, h - patch_size + 1, patch_size):
                    for j in range(0, w - patch_size + 1, patch_size):
                        mask_patch = mask_array[i:i+patch_size, j:j+patch_size].copy()
                        
                        effective_area = patch_size * patch_size
                        target_pixels = np.sum(np.isin(mask_patch, subset_classes))
                        pixel_ratio = target_pixels / effective_area
                        
                        if pixel_ratio < valid_threshold:
                            continue
                            
                        mask_patch[~np.isin(mask_patch, subset_classes)] = 0
                        
                        img_patch = image_array[:, i:i+patch_size, j:j+patch_size]
                        tile_filename = os.path.join(dest_dir, f"{sample_id}_tile_{tile_idx}.npz")
                        
                        np.savez_compressed(tile_filename, X=img_patch, y=mask_patch)
                        extracted_count += 1
                        tile_idx += 1
                        
        except Exception as e:
            print(f"[Error] Failed to process {filename}: {str(e)}")
            
        return extracted_count


    def _split_balanced_segmentation_tiles(self, tile_stats, base_dir, subset_classes, temp_dir):
        """
        Performs a simple random 70/20/10 split of the extracted tiles.
        """
        import random
        import shutil
        import os
        
        print("\n--- Simple Random Split (70/20/10) ---")
        train_dir = os.path.join(base_dir, "train")
        val_dir = os.path.join(base_dir, "validation")
        test_dir = os.path.join(base_dir, "test")

        for d in [train_dir, val_dir, test_dir]:
            os.makedirs(d, exist_ok=True)

        # 1. Shuffle the tiles randomly
        random.seed(42)  # Lock the seed so the splits are reproducible across runs
        random.shuffle(tile_stats)
        
        n_total = len(tile_stats)
        if n_total == 0:
            print("No valid tiles found to split!")
            return

        # 2. Calculate indices for a 70/20/10 split
        train_end = int(n_total * 0.70)
        val_end = int(n_total * 0.90)  # 70% + 20%
        
        train_split = tile_stats[:train_end]
        val_split = tile_stats[train_end:val_end]
        test_split = tile_stats[val_end:]
        
        print(f"Total Valid Tiles: {n_total}")
        print(f" -> Train: {len(train_split)} tiles (70%)")
        print(f" -> Validation: {len(val_split)} tiles (20%)")
        print(f" -> Test: {len(test_split)} tiles (10%)")

        # 3. Move the files from the temporary folder to their respective split directories
        for stat in train_split:
            filename = os.path.basename(stat['path'])
            shutil.move(stat['path'], os.path.join(train_dir, filename))
            
        for stat in val_split:
            filename = os.path.basename(stat['path'])
            shutil.move(stat['path'], os.path.join(val_dir, filename))
            
        for stat in test_split:
            filename = os.path.basename(stat['path'])
            shutil.move(stat['path'], os.path.join(test_dir, filename))

        # Cleanup temp directory
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)

        # 4. Write Summary Log
        summary_path = os.path.join(base_dir, "split_summary.txt")
        with open(summary_path, "w") as f:
            f.write("--- Simple Random 70/20/10 Extraction Summary ---\n")
            f.write(f"Subset Classes: {subset_classes}\n")
            f.write(f"Total Tiles Extracted: {n_total}\n\n")
            
            f.write("--- Tile Distribution ---\n")
            f.write(f"Train Tiles: {len(train_split)}\n")
            f.write(f"Validation Tiles: {len(val_split)}\n")
            f.write(f"Test Tiles: {len(test_split)}\n")

        print(f"\nExtraction and splitting complete. Summary saved to: {summary_path}")

    def get_file_list(self, from_train, limit, path=None):
        """
        Generates a fixed list of random sample IDs to ensure consistency 
        across multiple data loading runs.
        
        Parameters:
        - from_train (bool): If True, reads from the train.txt file. If False, reads directory contents.
        - path (str): The full path to either the train.txt file or the h5_data directory.
        - limit (int): The maximum number of filenames to sample.
        
        Returns:
        - list of strings: The randomly sampled filenames (without the .h5 extension).
        """
        file_names = []
        
        if from_train:
            # Read from the training text file
            if not os.path.isfile(path):
                print(f"Error: Training list file not found at {path}")
                return []
                
            with open(path, 'r') as f:
                file_names = [line.strip().replace('.h5', '') for line in f.readlines() if line.strip()]
        else:
            # Read directly from the h5_data directory
            if path is None:
                path = self.h5_dir
            if not os.path.isdir(path):
                print(f"Error: Directory not found at {path}")
                return []
                
            file_names = [f.replace('.h5', '') for f in os.listdir(path) if f.endswith('.h5')]
            
        if not file_names:
            print(f"Error: No valid files/entries found at {path}.")
            return []
            
        # Safely cap the limit if the user asks for more files than exist
        actual_limit = min(limit, len(file_names))
        
        # Randomly sampling without dusplicates
        sampled_ids = random.sample(file_names, actual_limit)
        
        print(f"Successfully sampled {len(sampled_ids)} file IDs.")
        return sampled_ids


    def get_chunked_file_list(self, chunk_size=300, path=None, random_state=42):
        """
        Retrieves all available .h5 files in the dataset, shuffles them to prevent 
        geographic bias, and splits them into a list of smaller file lists (chunks).
        
        Parameters:
        - chunk_size (int): The maximum number of filenames per sublist.
        - path (str): Optional path to the directory. Defaults to self.h5_dir.
        - random_state (int): Seed for reproducibility across different script runs.
        
        Returns:
        - list of lists of strings: A list where each element is a list of file IDs.
        """
        if path is None:
            path = self.h5_dir
            
        if not os.path.isdir(path):
            print(f"Error: Directory not found at {path}")
            return []
            
        # Get all valid files and strip the extension to match your pipeline's format
        file_names = [f.replace('.h5', '') for f in os.listdir(path) if f.endswith('.h5')]
        
        if not file_names:
            print(f"Error: No valid files found at {path}.")
            return []
            
        # Remove any potential duplicates (just as a safety net)
        file_names = list(set(file_names))
            
        # Shuffle the list to ensure a random geographic distribution in every chunk
        import random
        random.seed(random_state)
        random.shuffle(file_names)
        
        # Split the flattened list into a list of lists based on chunk_size
        chunked_list = [file_names[i:i + chunk_size] for i in range(0, len(file_names), chunk_size)]
        
        print(f"Successfully parsed {len(file_names)} total files.")
        print(f"-> Created {len(chunked_list)} chunks (max {chunk_size} files per chunk).")
        
        return chunked_list



    def _load_single_file(self, filename, detail_layer=None, static=False):
        """
        Internal helper method to load a single .h5 file and apply filtering logic.
        """
        file_path = os.path.join(self.h5_dir, filename)
        
        dataset_dict = {'sample_id': filename.replace('.h5', '')}
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Get Hyperspectral
                dataset_dict['hyperspectral'] = np.array(f['EnMAP_data'])
                
                # Get Multispectral (+ process for filtering temporal dimension if necessary)
                s2_full = np.array(f['S2_data'])
                if static:
                    month_str = filename[4:6]
                    s2_time_index = int(month_str) - 1
                    dataset_dict['multispectral'] = s2_full[s2_time_index] 
                else:
                    dataset_dict['multispectral'] = s2_full
                
                # Get labels
                labels_full = np.array(f['label'])

                # Get priors
                priors_full = np.array(f['priors'])
                
                if detail_layer is not None and isinstance(detail_layer, int) and 0 <= detail_layer <= 3:
                    dataset_dict['labels'] = labels_full[detail_layer]
                    dataset_dict['prior'] = priors_full[detail_layer]
                else:
                    dataset_dict['labels'] = labels_full
                    dataset_dict['prior'] = priors_full
                    
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
            
        return dataset_dict


    def get_random_h5_sample(self, detail_layer=None, static=False):
        """
        Picks a random file from the h5_data directory and returns its data.
        
        Parameters:
        - detail_layer (int): 0, 1, 2, or 3. If None, returns all layers.
        - static (bool): If True, returns only the S2 month matching the EnMAP data.
        """
        all_files = [f for f in os.listdir(self.h5_dir) if f.endswith('.h5')]
        if not all_files:
            print("No .h5 files found in directory.")
            return None
            
        random_file = random.choice(all_files)
        return self._load_single_file(random_file, detail_layer, static)


    def load_h5_data(self, file_list, detail_layer=None, static=False, data_type="hyperspectral", keep_prior=False):
        """
        Loads data based on a pre-determined list of sample IDs, 
        filtering the modalities on the fly to maximize RAM efficiency.
        
        Parameters:
        - file_list (list of str): The list of sample IDs generated by get_file_list().
        - ... (other parameters remain the same)
        """
        if data_type not in ["hyperspectral", "multispectral"]:
            raise ValueError("data_type must be either 'hyperspectral' or 'multispectral'")
            
        if not file_list:
            print("Error: Provided file_list is empty.")
            return []
            
        key_to_remove = "multispectral" if data_type == "hyperspectral" else "hyperspectral"
        
        loaded_data = []
        
        for sample_id in file_list:
            filename = f"{sample_id}.h5"
            data_dict = self._load_single_file(filename, detail_layer, static)
            
            if data_dict:
                # Filter on the fly
                if key_to_remove in data_dict:
                    data_dict.pop(key_to_remove)
                    
                if not keep_prior and 'prior' in data_dict:
                    data_dict.pop('prior')
                    
                loaded_data.append(data_dict)
                
        print(f"Successfully loaded {len(loaded_data)} files into memory.")
        print(f"-> Modality kept: '{data_type}'. Priors kept: {keep_prior}.")
        
        return loaded_data


    def upsample_hyperspectral(self, hyper_data):
        """
        Upsamples the 64x64 hyperspectral array to 192x192 to match Sentinel-2 and labels resolution.
        Uses nearest-neighbor interpolation to maintain exact spectral purity.
        
        Parameters:
        - hyper_data: numpy array of shape (Channels, 64, 64)
        
        Returns:
        - numpy array of shape (Channels, 192, 192)
        """
        # Repeat 3 times along the height (axis 1), then 3 times along the width (axis 2)
        upsampled = np.repeat(np.repeat(hyper_data, 3, axis=1), 3, axis=2)
        return upsampled
    

    def balance_pixels(self, X, y, total_samples=100000, random_state=42):
        """
        Balances the dataset evenly across all classes, aiming for a total number of samples.
        Automatically caps the size if the rarest class doesn't have enough pixels.
        
        Parameters:
        - X: Flattened feature matrix (Pixels, Channels)
        - y: Flattened label array (Pixels,)
        - total_samples: The target total size for the final dataset (e.g., 100000)
        - random_state: In this way when I pass the same X and y the picked samples will be the same
                        Help with running first with hyperspectral and then multispectral

        Returns:
        - X_final, y_final: Perfectly balanced arrays.
        """
        unique_classes, class_counts = np.unique(y, return_counts=True)
        num_classes = len(unique_classes)
        
        print("\n--- Pixel Balancing ---")
        print("Original Class Distribution:")
        for cls, count in zip(unique_classes, class_counts):
            print(f" - Class {cls}: {count} pixels")
            
        # Calculate how many pixels we need per class to hit the total_samples target
        if total_samples is not None:
            desired_per_class = total_samples // num_classes
            
            # Protect against asking for more pixels than the rarest class actually has
            rarest_class_count = np.min(class_counts)
            target_per_class = min(desired_per_class, rarest_class_count)
            
            if target_per_class < desired_per_class:
                print(f"\nWarning: Target was {desired_per_class} per class, but the rarest class only has {rarest_class_count}.")
                print(f"Adjusting to {rarest_class_count} per class to maintain a mathematically perfect balance.")
        else:
            # If no total is given, just balance to the rarest class
            target_per_class = np.min(class_counts)
            
        print(f"\nBalancing dataset to {target_per_class} pixels per class...")
        
        X_balanced_list = []
        y_balanced_list = []
        
        for cls in unique_classes:
            # Find all rows belonging to this class
            cls_indices = np.where(y == cls)[0]
            
            # Randomly sample the exact number needed
            sampled_indices = np.random.choice(cls_indices, size=target_per_class, replace=False)
            
            X_balanced_list.append(X[sampled_indices])
            y_balanced_list.append(y[sampled_indices])
            
        # Stack everything back together
        X_final = np.vstack(X_balanced_list)
        y_final = np.concatenate(y_balanced_list)
        
        actual_total = len(y_final)
        print(f"Final Balanced Shape: X={X_final.shape}, y={y_final.shape} (Total: {actual_total})")
        
        return X_final, y_final
    

    def drop_classes(self, X, y, classes_to_drop=None):
        """
        Removes specific classes from the flattened dataset.
        
        Parameters:
        - X: Flattened feature matrix (Pixels, Channels)
        - y: Flattened label array (Pixels,)
        - classes_to_drop: List of class integers to remove (e.g., [0, 5])
        
        Returns:
        - X_filtered, y_filtered: Arrays with the specified classes removed.
        """
        if not classes_to_drop:
            return X, y
            
        print(f"\n--- Dropping Classes: {classes_to_drop} ---")
        
        # creating mask
        valid_mask = ~np.isin(y, classes_to_drop)
        
        # Applying mask
        X_filtered = X[valid_mask]
        y_filtered = y[valid_mask]
        
        pixels_dropped = len(y) - len(y_filtered)
        print(f"Original shape: {y.shape[0]} pixels")
        print(f"New shape:      {y_filtered.shape[0]} pixels")
        print(f"Dropped {pixels_dropped} pixels belonging to classes {classes_to_drop}.")
        
        return X_filtered, y_filtered