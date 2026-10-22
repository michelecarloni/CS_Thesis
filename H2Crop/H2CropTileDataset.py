import os
import glob
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class H2CropTileDataset(Dataset):
    """
    Custom PyTorch Dataset that loads 64x64 tiles from disk on the fly.
    Supports four distinct training augmentation modes via train_aug_mode.
    """
    def __init__(
        self, 
        directory, 
        subset_classes, 
        minority_classes=None,
        debug=False, 
        augment=None, 
        train_aug_mode="minority_class_aug",
        min_rare_pixels_per_tile=10
    ):
        self.directory = directory
        self.subset_classes = sorted(subset_classes)
        self.minority_classes = set(minority_classes) if minority_classes is not None else set()
        self.debug = debug
        self.train_aug_mode = train_aug_mode.lower()
        self.min_rare_pixels_per_tile = min_rare_pixels_per_tile
        
        valid_modes = ["no_aug", "random_aug", "full_aug", "minority_class_aug"]
        if self.train_aug_mode not in valid_modes:
            raise ValueError(f"Invalid train_aug_mode '{train_aug_mode}'. Must be one of {valid_modes}")
            
        self.filepaths = sorted(glob.glob(os.path.join(directory, "*.npz")))
        if debug:
            self.filepaths = self.filepaths[:10]
            
        if len(self.filepaths) == 0:
            raise ValueError(f"No .npz files found in {directory}. Check your paths!")
            
        is_train_dir = os.path.basename(os.path.normpath(directory)).lower() == "train"
        self.is_train = is_train_dir if augment is None else augment
            
        self.unique_classes = np.array([0] + self.subset_classes)
        max_label = int(np.max(self.unique_classes))
        self.label_mapping = np.zeros(max_label + 1, dtype=np.int64)
        for new_idx, raw_label in enumerate(self.unique_classes):
            self.label_mapping[raw_label] = new_idx

        self.all_aug_modes = ["orig", "hflip", "vflip", "rot90", "rot180", "rot270"]
        
        self.rare_tile_paths = []
        self.common_tile_paths = []
        self.active_samples = []
        
        if self.is_train:
            self._prepare_dataset_partitioning()
            self.set_epoch(0)
        else:
            self.active_samples = [(fp, "orig") for fp in self.filepaths]

    def _prepare_dataset_partitioning(self):
        """
        Scans cached metadata to partition training tiles into rare vs common based on explicit classes.
        """
        cache_path = os.path.join(self.directory, "tile_class_pixel_counts.json")
        tile_to_class_counts = {}
        
        if not self.debug and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    tile_to_class_counts = json.load(f)
            except Exception:
                tile_to_class_counts = {}

        missing_files = [fp for fp in self.filepaths if os.path.basename(fp) not in tile_to_class_counts]
        if missing_files:
            print(f"[*] Scanning {len(missing_files)} train tiles to build class frequency index...")
            for fp in tqdm(missing_files, desc="Indexing Train Tiles", leave=False):
                with np.load(fp) as data:
                    y = data['y']
                    vals, counts = np.unique(y, return_counts=True)
                    crop_counts = {
                        str(int(c)): int(cnt) 
                        for c, cnt in zip(vals, counts) 
                        if int(c) in self.subset_classes
                    }
                    tile_to_class_counts[os.path.basename(fp)] = crop_counts
            
            if not self.debug:
                with open(cache_path, "w") as f:
                    json.dump(tile_to_class_counts, f)

        # REMOVED THRESHOLD CALCULATION: We now explicitly use self.minority_classes
        self.rare_classes = self.minority_classes

        # Partition files
        for fp in self.filepaths:
            crop_counts = tile_to_class_counts.get(os.path.basename(fp), {})
            has_rare = any(
                (int(c_str) in self.rare_classes and pix_cnt >= self.min_rare_pixels_per_tile)
                for c_str, pix_cnt in crop_counts.items()
            )
            if has_rare:
                self.rare_tile_paths.append(fp)
            else:
                self.common_tile_paths.append(fp)

        # Chunk size for common tiles (used in minority_class_aug to match expanded rare pool size)
        num_rare_expanded = len(self.rare_tile_paths) * len(self.all_aug_modes)
        self.chunk_size_common = max(1, num_rare_expanded) if len(self.rare_tile_paths) > 0 else len(self.filepaths)
        
        rng = random.Random(42)
        rng.shuffle(self.common_tile_paths)
        rng.shuffle(self.rare_tile_paths)

        print(f"    [Augmentation Mode: {self.train_aug_mode}]")
        print(f"    -> Explicit Minority Classes: {sorted(list(self.rare_classes))}")
        print(f"    -> Rare Tiles: {len(self.rare_tile_paths)} | Common Tiles: {len(self.common_tile_paths)}")

    def set_epoch(self, epoch):
        # (This method remains exactly identical to the previous version)
        if not self.is_train:
            return

        self.active_samples = []

        if self.train_aug_mode == "no_aug":
            for fp in self.filepaths:
                self.active_samples.append((fp, "orig"))

        elif self.train_aug_mode == "random_aug":
            for fp in self.filepaths:
                self.active_samples.append((fp, "random_choice"))

        elif self.train_aug_mode == "full_aug":
            for fp in self.filepaths:
                for mode in self.all_aug_modes:
                    self.active_samples.append((fp, mode))

        elif self.train_aug_mode == "minority_class_aug":
            for fp in self.rare_tile_paths:
                for mode in self.all_aug_modes:
                    self.active_samples.append((fp, mode))

            total_common = len(self.common_tile_paths)
            if total_common > 0:
                start_idx = (epoch * self.chunk_size_common) % total_common
                end_idx = start_idx + self.chunk_size_common
                
                if end_idx <= total_common:
                    epoch_common_paths = self.common_tile_paths[start_idx:end_idx]
                else:
                    epoch_common_paths = self.common_tile_paths[start_idx:] + self.common_tile_paths[:end_idx % total_common]
                
                for fp in epoch_common_paths:
                    self.active_samples.append((fp, "random_choice"))

        rng = random.Random(42 + epoch)
        rng.shuffle(self.active_samples)

    def __len__(self):
        return len(self.active_samples)
        
    def __getitem__(self, idx):
        # (This method remains exactly identical to the previous version)
        filepath, aug_mode = self.active_samples[idx]
        
        with np.load(filepath) as data:
            raw_X = np.nan_to_num(data['X'], nan=0.0, posinf=0.0, neginf=0.0)
            X = torch.tensor(raw_X, dtype=torch.float32)
            
            raw_y = data['y']
            mapped_y = self.label_mapping[raw_y]
            y = torch.tensor(mapped_y, dtype=torch.long)
            
        if aug_mode == "orig":
            pass
        elif aug_mode == "hflip":
            X = torch.flip(X, dims=[2])
            y = torch.flip(y, dims=[1])
        elif aug_mode == "vflip":
            X = torch.flip(X, dims=[1])
            y = torch.flip(y, dims=[0])
        elif aug_mode == "rot90":
            X = torch.rot90(X, k=1, dims=[1, 2])
            y = torch.rot90(y, k=1, dims=[0, 1])
        elif aug_mode == "rot180":
            X = torch.rot90(X, k=2, dims=[1, 2])
            y = torch.rot90(y, k=2, dims=[0, 1])
        elif aug_mode == "rot270":
            X = torch.rot90(X, k=3, dims=[1, 2])
            y = torch.rot90(y, k=3, dims=[0, 1])
        elif aug_mode == "random_choice":
            choice = random.choice(["none", "hflip", "vflip", "rot90", "rot180", "rot270"])
            if choice == "hflip":
                X = torch.flip(X, dims=[2])
                y = torch.flip(y, dims=[1])
            elif choice == "vflip":
                X = torch.flip(X, dims=[1])
                y = torch.flip(y, dims=[0])
            elif choice == "rot90":
                X = torch.rot90(X, k=1, dims=[1, 2])
                y = torch.rot90(y, k=1, dims=[0, 1])
            elif choice == "rot180":
                X = torch.rot90(X, k=2, dims=[1, 2])
                y = torch.rot90(y, k=2, dims=[0, 1])
            elif choice == "rot270":
                X = torch.rot90(X, k=3, dims=[1, 2])
                y = torch.rot90(y, k=3, dims=[0, 1])
                
        return X, y