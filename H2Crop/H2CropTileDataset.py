import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class H2CropTileDataset(Dataset):
    """
    Custom PyTorch Dataset that loads 64x64 tiles from disk on the fly.
    - Uses a deterministic subset_classes list to map labels to 0..K.
    - Automatically applies random flips and 90-degree rotations on the 'train' split.
    """
    def __init__(self, directory, subset_classes, debug=False, augment=None):
        self.filepaths = sorted(glob.glob(os.path.join(directory, "*.npz")))
        
        if debug:
            self.filepaths = self.filepaths[:10]
            
        if len(self.filepaths) == 0:
            raise ValueError(f"No .npz files found in {directory}. Check your paths!")
            
        # Automatically enable augmentation if loading from the 'train' folder (unless explicitly overridden)
        if augment is None:
            self.augment = os.path.basename(os.path.normpath(directory)).lower() == "train"
        else:
            self.augment = augment
            
        # 1. Deterministically build the classes array (Background 0 + sorted subset crops)
        self.unique_classes = np.array([0] + sorted(subset_classes))
        
        # 2. Build a fast NumPy lookup array for O(1) label mapping
        max_label = int(np.max(self.unique_classes))
        self.label_mapping = np.zeros(max_label + 1, dtype=np.int64)
        for new_idx, raw_label in enumerate(self.unique_classes):
            self.label_mapping[raw_label] = new_idx

    def __len__(self):
        return len(self.filepaths)
        
    def __getitem__(self, idx):
        with np.load(self.filepaths[idx]) as data:
            # Load spectral array and sanitize any isolated NaN/Inf sensor pixels
            raw_X = np.nan_to_num(data['X'], nan=0.0, posinf=0.0, neginf=0.0)
            X = torch.tensor(raw_X, dtype=torch.float32)
            
            # Extract raw spatial labels and map to contiguous 0..K indices
            raw_y = data['y']
            mapped_y = self.label_mapping[raw_y]
            y = torch.tensor(mapped_y, dtype=torch.long)
            
        # 3. Apply Geometric Data Augmentation (Flips & 90-deg Rotations) on Training Set
        if self.augment:
            # Random Horizontal Flip
            if random.random() > 0.5:
                X = torch.flip(X, dims=[2])
                y = torch.flip(y, dims=[1])
                
            # Random Vertical Flip
            if random.random() > 0.5:
                X = torch.flip(X, dims=[1])
                y = torch.flip(y, dims=[0])
                
            # Random 90, 180, or 270 degree Rotation
            k = random.randint(0, 3)
            if k > 0:
                X = torch.rot90(X, k, dims=[1, 2])
                y = torch.rot90(y, k, dims=[0, 1])
                
        return X, y