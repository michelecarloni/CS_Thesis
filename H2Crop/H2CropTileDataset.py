import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset



class H2CropTileDataset(Dataset):
    """
    Custom PyTorch Dataset that loads tiles from disk on the fly.
    Now accepts a specific list of filepaths (Train, Val, or Test split).
    """
    def __init__(self, filepaths):
        # We directly accept the list of files now; no os.path.join or glob needed!
        self.filepaths = filepaths
        if len(self.filepaths) == 0:
            raise ValueError("Provided filepaths list is empty.")
        
    def __len__(self):
        return len(self.filepaths)
        
    def __getitem__(self, idx):
        # Load a single tile into RAM only when the DataLoader requests it
        data = np.load(self.filepaths[idx])
        X = torch.tensor(data['X'], dtype=torch.float32)
        y = torch.tensor(data['y'], dtype=torch.long)
        return X, y