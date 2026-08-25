import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

class H2CropTileDataset(Dataset):
    """
    Custom PyTorch Dataset that loads tiles from disk on the fly.
    Accepts a directory path and a debug flag to dynamically build the filepath list.
    """
    def __init__(self, directory, debug=False):
        # Find all .npz files in the provided directory
        self.filepaths = glob.glob(os.path.join(directory, "*.npz"))
        
        # If debug mode is active, aggressively cut down the dataset to 10 files
        if debug:
            self.filepaths = self.filepaths[:10]
            
        if len(self.filepaths) == 0:
            raise ValueError(f"No .npz files found in {directory}. Check your paths!")
        
    def __len__(self):
        return len(self.filepaths)
        
    def __getitem__(self, idx):
        # Load a single tile into RAM only when the DataLoader requests it[cite: 4]
        # Using 'with' ensures the file is safely closed immediately after reading
        with np.load(self.filepaths[idx]) as data:
            X = torch.tensor(data['X'], dtype=torch.float32)[cite: 4]
            y = torch.tensor(data['y'], dtype=torch.long)[cite: 4]
            
        return X, y

    def get_unique_classes(self):
        """
        Scans the dataset to identify all unique classes present.
        Used by the pipeline to dynamically configure the U-Net's output channels.
        """
        unique_classes = set()
        
        print(f"      [Dataset] Scanning {len(self.filepaths)} files to determine unique classes...")
        for fp in self.filepaths:
            with np.load(fp) as data:
                unique_classes.update(np.unique(data['y']).tolist())
                
        return np.array(sorted(list(unique_classes)))