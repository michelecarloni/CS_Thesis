import os
import glob
import numpy as np

def check_npz_data_range(dataset_dir, num_files_to_check=5):
    """
    Loads a small sample of .npz files and prints the statistical 
    range of the raw feature data (X).
    """
    files = glob.glob(os.path.join(dataset_dir, "*.npz"))
    if not files:
        print(f"No .npz files found in {dataset_dir}")
        return
        
    print(f"Checking data range for {min(num_files_to_check, len(files))} files in:\n{dataset_dir}\n")
    
    for i, file_path in enumerate(files[:num_files_to_check]):
        with np.load(file_path) as data:
            X = data['X']
            print(f"File {i+1}: {os.path.basename(file_path)}")
            print(f"  Shape: {X.shape} | dtype: {X.dtype}")
            print(f"  Min:   {np.min(X):.6f}")
            print(f"  Max:   {np.max(X):.6f}")
            print(f"  Mean:  {np.mean(X):.6f}")
            print(f"  Std:   {np.std(X):.6f}\n")

if __name__ == "__main__":
    # Point this to one of your unscaled raw data folders
    target_dir = "ds/H2Crop_tiles_unique_subset/split_1/hyperspectral_taxonomy_3_pSize_64/train"
    
    check_npz_data_range(target_dir, num_files_to_check=5)