import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset



class H2CropTileDataset(Dataset):
    """
    Custom PyTorch Dataset that loads tiles from disk on the fly.
    This guarantees we never run out of System RAM.
    """
    def __init__(self, tiles_dir):
        # Grab a list of all tile file paths
        self.filepaths = glob.glob(os.path.join(tiles_dir, "*.npz"))
        if len(self.filepaths) == 0:
            raise ValueError(f"No .npz tiles found in {tiles_dir}")
        
    def __len__(self):
        return len(self.filepaths)
        
    def __getitem__(self, idx):
        # Load a single tile into RAM only when the DataLoader requests it
        data = np.load(self.filepaths[idx])
        X = torch.tensor(data['X'], dtype=torch.float32)
        y = torch.tensor(data['y'], dtype=torch.long)
        return X, y