import os
import joblib
import numpy as np

def compare_all_scalers(base_path_1, base_path_2, num_splits=3, taxonomy=3, patch_size=64, tolerance=1e-5):
    """
    Iterates through all modalities and splits, loading scikit-learn StandardScaler 
    .joblib files from two base directories and comparing their mathematical equivalence.
    """
    modalities = ["hyperspectral", "multispectral"]
    
    print(f"{'='*70}")
    print("SCALER COMPARISON REPORT")
    print(f"Base 1: {base_path_1}")
    print(f"Base 2: {base_path_2}")
    print(f"{'='*70}")
    
    all_match = True
    
    for mod in modalities:
        print(f"\n--- MODALITY: {mod.upper()} ---")
        
        for split_id in range(1, num_splits + 1):
            file_name = f"scaler_split_{split_id}_tax_{taxonomy}_pSize_{patch_size}.joblib"
            
            path1 = os.path.join(base_path_1, mod, file_name)
            path2 = os.path.join(base_path_2, mod, file_name)
            
            if not os.path.exists(path1) or not os.path.exists(path2):
                print(f"  [Split {split_id}] ERROR: Missing files.")
                if not os.path.exists(path1): print(f"    -> Not found: {path1}")
                if not os.path.exists(path2): print(f"    -> Not found: {path2}")
                all_match = False
                continue
                
            scaler1 = joblib.load(path1)
            scaler2 = joblib.load(path2)
            
            if scaler1.mean_.shape != scaler2.mean_.shape:
                print(f"  [Split {split_id}] MISMATCH: Shapes differ ({scaler1.mean_.shape} vs {scaler2.mean_.shape})")
                all_match = False
                continue
                
            mean_diff = np.abs(scaler1.mean_ - scaler2.mean_)
            scale_diff = np.abs(scaler1.scale_ - scaler2.scale_)
            
            max_mean_diff = np.max(mean_diff)
            max_scale_diff = np.max(scale_diff)
            
            mean_match = np.allclose(scaler1.mean_, scaler2.mean_, atol=tolerance)
            scale_match = np.allclose(scaler1.scale_, scaler2.scale_, atol=tolerance)
            
            if mean_match and scale_match:
                print(f"  [Split {split_id}] MATCH | Max Diff \u03bc: {max_mean_diff:.4e} | Max Diff \u03c3: {max_scale_diff:.4e}")
            else:
                print(f"  [Split {split_id}] MISMATCH | Max Diff \u03bc: {max_mean_diff:.4e} | Max Diff \u03c3: {max_scale_diff:.4e}")
                all_match = False

    print(f"\n{'='*70}")
    if all_match:
        print("[VERDICT] All scalers are mathematically equivalent.")
    else:
        print("[VERDICT] Discrepancies found. Check the logs above.")
    print(f"{'='*70}")


if __name__ == "__main__":
    # Base paths pointing strictly to the 'unique_subset' directory
    preprocessing_base = "/media/michele/T7/scalers/scalers_unique_subset/unique_subset/"
    standard_ml_base = "/media/michele/T7/checkpoints/checkpoints_9_standard_ML_uniqueSub_split_same_tot_tiles/scalers/unique_subset"
    
    compare_all_scalers(
        base_path_1=preprocessing_base, 
        base_path_2=standard_ml_base, 
        num_splits=3, 
        taxonomy=3, 
        patch_size=64, 
        tolerance=1e-5
    )