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
    - On the 'train' split (IN-MEMORY ONLY, NO DISK DELETION):
        1. Rare tiles are expanded into all 6 deterministic views every epoch:
           [Original, Horizontal Flip, Vertical Flip, 90° Rot, 180° Rot, 270° Rot].
        2. Common tiles cycle through disjoint chunks across epochs, and for each 
           epoch, randomly decides whether to show the original tile or a randomly 
           augmented version.
    """
    def __init__(
        self, 
        directory, 
        subset_classes, 
        debug=False, 
        augment=None, 
        min_rare_pixels_per_tile=10
    ):
        self.directory = directory
        self.subset_classes = sorted(subset_classes)
        self.debug = debug
        self.min_rare_pixels_per_tile = min_rare_pixels_per_tile
        
        self.filepaths = sorted(glob.glob(os.path.join(directory, "*.npz")))
        if debug:
            self.filepaths = self.filepaths[:10]
            
        if len(self.filepaths) == 0:
            raise ValueError(f"No .npz files found in {directory}. Check your paths!")
            
        is_train_dir = os.path.basename(os.path.normpath(directory)).lower() == "train"
        self.is_train = is_train_dir if augment is None else augment
            
        # 1. Deterministically build classes array (Background 0 + sorted subset crops)
        self.unique_classes = np.array([0] + self.subset_classes)
        
        # 2. Fast NumPy lookup array for O(1) label mapping
        max_label = int(np.max(self.unique_classes))
        self.label_mapping = np.zeros(max_label + 1, dtype=np.int64)
        for new_idx, raw_label in enumerate(self.unique_classes):
            self.label_mapping[raw_label] = new_idx

        # The 6 exhaustive deterministic views for rare tiles
        self.rare_aug_modes = ["orig", "hflip", "vflip", "rot90", "rot180", "rot270"]
        
        self.rare_tile_paths = []
        self.common_tile_paths = []
        self.active_samples = []  # List of tuples: (filepath, aug_mode)
        
        if self.is_train:
            self._prepare_dynamic_balancing()
            self.set_epoch(0)
        else:
            self.active_samples = [(fp, "orig") for fp in self.filepaths]

    def _prepare_dynamic_balancing(self):
        """
        Scans or loads cached metadata to partition training tiles into rare vs common.
        Does NOT delete any files from disk.
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

        class_tile_counts = {c: 0 for c in self.subset_classes}
        for fp in self.filepaths:
            crop_counts = tile_to_class_counts.get(os.path.basename(fp), {})
            for c_str, pix_cnt in crop_counts.items():
                c_id = int(c_str)
                if c_id in class_tile_counts and pix_cnt >= self.min_rare_pixels_per_tile:
                    class_tile_counts[c_id] += 1

        # Identify rare classes (< 30% of max class tile count)
        max_tile_count = max(class_tile_counts.values()) if class_tile_counts else 0
        rare_threshold = max_tile_count * 0.30
        
        self.rare_classes = {
            c for c, cnt in class_tile_counts.items() 
            if 0 < cnt <= rare_threshold
        }
        
        if len(self.rare_classes) == 0:
            non_zero_classes = [c for c, cnt in class_tile_counts.items() if cnt > 0]
            if non_zero_classes:
                rarest_c = min(non_zero_classes, key=lambda c: class_tile_counts[c])
                self.rare_classes = {rarest_c}

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

        # Chunk size for common tiles to match the 6x expanded rare pool size
        num_rare_expanded = len(self.rare_tile_paths) * len(self.rare_aug_modes)
        self.chunk_size_common = max(1, num_rare_expanded)
        
        rng = random.Random(42)
        rng.shuffle(self.common_tile_paths)

        print(f"    [Dynamic Balancer] Class Tile Counts (>={self.min_rare_pixels_per_tile} px): {class_tile_counts}")
        print(f"    [Dynamic Balancer] Identified Rare Classes: {sorted(list(self.rare_classes))}")
        print(f"    [Dynamic Balancer] Rare Tiles: {len(self.rare_tile_paths)} (x6 deterministic views = {num_rare_expanded} samples/epoch)")
        print(f"    [Dynamic Balancer] Common Tiles Pool: {len(self.common_tile_paths)} (Slicing chunks of {self.chunk_size_common} per epoch)")

    def set_epoch(self, epoch):
        """
        Rebuilds active training samples for the epoch:
        - All rare tiles get all 6 views (orig, hflip, vflip, rot90, rot180, rot270).
        - A rotating chunk of common tiles is selected, with a random flag to show 
          either original or a randomly augmented version.
        """
        if not self.is_train:
            return

        # 1. Expand all rare tiles into 6 deterministic views
        rare_samples = []
        for fp in self.rare_tile_paths:
            for mode in self.rare_aug_modes:
                rare_samples.append((fp, mode))

        # 2. Extract a chunk of common tiles for this epoch
        total_common = len(self.common_tile_paths)
        if total_common > 0:
            start_idx = (epoch * self.chunk_size_common) % total_common
            end_idx = start_idx + self.chunk_size_common
            
            if end_idx <= total_common:
                epoch_common_paths = self.common_tile_paths[start_idx:end_idx]
            else:
                epoch_common_paths = self.common_tile_paths[start_idx:] + self.common_tile_paths[:end_idx % total_common]
            
            # For common tiles, tag them as "common_mix" so __getitem__ can randomly 
            # decide whether to keep them original or apply a random augmentation
            common_samples = [(fp, "common_mix") for fp in epoch_common_paths]
        else:
            common_samples = []

        rng = random.Random(42 + epoch)
        self.active_samples = rare_samples + common_samples
        rng.shuffle(self.active_samples)

    def __len__(self):
        return len(self.active_samples)
        
    def __getitem__(self, idx):
        filepath, aug_mode = self.active_samples[idx]
        
        with np.load(filepath) as data:
            raw_X = np.nan_to_num(data['X'], nan=0.0, posinf=0.0, neginf=0.0)
            X = torch.tensor(raw_X, dtype=torch.float32)
            
            raw_y = data['y']
            mapped_y = self.label_mapping[raw_y]
            y = torch.tensor(mapped_y, dtype=torch.long)
            
        # Apply augmentations based on mode
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
        elif aug_mode == "common_mix":
            # For common tiles: 50% chance to stay original, 50% chance for random augmentation
            if random.random() > 0.5:
                if random.random() > 0.5:
                    X = torch.flip(X, dims=[2])
                    y = torch.flip(y, dims=[1])
                if random.random() > 0.5:
                    X = torch.flip(X, dims=[1])
                    y = torch.flip(y, dims=[0])
                k = random.randint(1, 3)
                X = torch.rot90(X, k=k, dims=[1, 2])
                y = torch.rot90(y, k=k, dims=[0, 1])
                
        return X, y