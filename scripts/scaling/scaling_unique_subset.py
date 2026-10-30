import os
import sys
import glob
import gc
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Setup project root path
project_root = os.path.abspath('../../')
if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================================
# CONFIGURATION
# ==========================================
patch_size = 64
taxonomy = 3
num_splits = 3

# Source data (Raw Unscaled)
dataset_base = "../../ds/H2Crop_tiles_unique_subset"

# Where to save the fitted StandardScaler .joblib files
scaler_checkpoints_dir = "../../checkpoints/scalers/unique_subset"

modalities = ["hyperspectral", "multispectral"]


def chunked_file_generator(files, chunk_size=200):
    """Yields batches of files to prevent memory spikes when fitting the scaler."""
    for i in range(0, len(files), chunk_size):
        yield files[i:i + chunk_size]


def fit_and_save_global_scaler(train_files, in_channels, mod, split_id, out_dir):
    """
    Fits a StandardScaler incrementally across the training set and saves it.
    """
    print(f"    -> Fitting StandardScaler for {mod.upper()} Split {split_id}...")
    scaler = StandardScaler()
    
    for chunk in tqdm(list(chunked_file_generator(train_files, chunk_size=200)), desc="       Fitting Batches"):
        X_list = []
        for f in chunk:
            with np.load(f) as data:
                # Reshape from (Channels, H, W) to (Pixels, Channels)
                X_flat = data['X'].reshape(in_channels, -1).T.astype(np.float32)
                X_list.append(X_flat)
                
        X_batch = np.concatenate(X_list)
        
        # Neutralize NaNs and Infs to prevent math errors during scaling
        X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Incrementally update the global mean and std deviation
        scaler.partial_fit(X_batch)
        
        # Aggressively clear memory before the next chunk
        del X_list, X_batch
        gc.collect()
        
    # Explicitly create the target directory if it does not exist
    os.makedirs(out_dir, exist_ok=True)
    scaler_path = os.path.join(out_dir, f"scaler_split_{split_id}_tax_{taxonomy}_pSize_{patch_size}.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"    -> Scaler saved to: {scaler_path}")


if __name__ == "__main__":
    print(f"{'='*70}")
    print("STARTING OUT-OF-CORE SCALER COMPUTATION")
    print(f"{'='*70}")
    
    # Ensure the base checkpoint directory exists
    os.makedirs(scaler_checkpoints_dir, exist_ok=True)
    
    for split_id in range(1, num_splits + 1):
        for mod in modalities:
            in_channels = 218 if mod == "hyperspectral" else 10
            
            print(f"\n[*] Processing Split {split_id} | {mod.upper()}")
            
            mod_folder_name = f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}"
            train_in = os.path.join(dataset_base, f"split_{split_id}", mod_folder_name, "train")
            
            if not os.path.exists(train_in):
                print(f"    [Warning] Training directory not found: {train_in}. Skipping.")
                continue
                
            train_files = glob.glob(os.path.join(train_in, "*.npz"))
            if not train_files:
                print(f"    [Warning] No .npz files found in {train_in}. Skipping.")
                continue
                
            scaler_out_dir = os.path.join(scaler_checkpoints_dir, mod)
            
            # Compute and save
            fit_and_save_global_scaler(train_files, in_channels, mod, split_id, scaler_out_dir)

    print(f"\n{'='*70}")
    print("ALL SCALERS SUCCESSFULLY COMPUTED AND SAVED")
    print(f"{'='*70}")