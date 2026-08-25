import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class H2CropTileDataset(Dataset):
    """
    Custom PyTorch Dataset that loads tiles from disk on the fly.
    Uses a deterministic subset_classes list to map labels, eliminating the need to scan files.
    """
    def __init__(self, directory, subset_classes, debug=False):
        self.filepaths = glob.glob(os.path.join(directory, "*.npz"))
        
        if debug:
            self.filepaths = self.filepaths[:10]
            
        if len(self.filepaths) == 0:
            raise ValueError(f"No .npz files found in {directory}. Check your paths!")
            
        # 1. Deterministically build the classes array (Background 0 + sorted subset crops)
        self.unique_classes = np.array([0] + sorted(subset_classes))
        
        # 2. Build a highly optimized NumPy lookup array for O(1) label mapping
        max_label = int(np.max(self.unique_classes))
        self.label_mapping = np.zeros(max_label + 1, dtype=np.int64)
        for new_idx, raw_label in enumerate(self.unique_classes):
            self.label_mapping[raw_label] = new_idx

    def __len__(self):
        return len(self.filepaths)
        
    def __getitem__(self, idx):
        with np.load(self.filepaths[idx]) as data:
            X = torch.tensor(data['X'], dtype=torch.float32)
            
            # Extract raw spatial labels
            raw_y = data['y']
            
            # Instantly map all pixels to contiguous 0-N indices using the lookup array
            mapped_y = self.label_mapping[raw_y]
            y = torch.tensor(mapped_y, dtype=torch.long)
            
        return X, y