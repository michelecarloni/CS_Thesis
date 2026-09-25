import os
import sys

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)
    
import json
import gc
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from H2Crop.data_structures import h2crop_taxonomy_dict
from hyperparams_tuning.optimize_hyperparameters import optimize_hyperparameters
from utils import load_and_flatten_segmentation_tiles
import cupy as cp





def pipeline_H2Crop_standard_ML_algo_tiles_optuna(
    save_results_dir, 
    dataset_dir, 
    subset_id, 
    modality, 
    taxonomy=3, 
    patch_size=32, 
    use_gpu=True, 
    max_train_pixels=500000,
    n_trials=20,
    test_batch_size=50, 
    debug=False
):
    """
    Optuna-powered ML segmentation pipeline utilizing a memory-safe Two-Pass Architecture.
    Includes explicit Validation set loading for rigorous hyperparameter optimization.

    Arguments:
    - save_results_dir (str): Base directory for metrics, reports, and confusion matrices.
    - dataset_dir (str): Directory containing 'train', 'validation', and 'test' subfolders of .npz tiles.
    - subset_id (int): Identifier for the current crop subset (1, 2, 3, or 4).
    - modality (str): Modality type ('hyperspectral' or 'multispectral').
    - taxonomy (int): Taxonomic hierarchical level for mapping class labels.
    - patch_size (int): Height and width of the square image tiles.
    - use_gpu (bool): If True, uses native cuML multiclass estimators on the GPU.
    - max_train_pixels (int): Maximum pixels to load for training. Validation is capped proportionally.
    - n_trials (int): Number of Optuna hyperparameter exploration trials per model.
    - test_batch_size (int): Number of .npz files loaded simultaneously during test evaluation.
    - debug (bool): If True, artificially restricts data and trials for rapid plumbing tests.
    """
    print(f"\n{'='*70}")
    mode = "DEBUG MODE" if debug else "PRODUCTION MODE (OPTUNA)"
    print(f"STARTING ML SEGMENTATION PIPELINE FOR: {modality.upper()} | Subset {subset_id} | {mode}")
    print(f"{'='*70}")

    results_out_dir = os.path.join(save_results_dir, modality)
    os.makedirs(results_out_dir, exist_ok=True)
    
    # LOAD & SCALE TRAIN/VAL SETS
    if debug:
        print("Loading Train & Val tiles (DEBUG MODE: Reading only 10 files)...")
    else:
        print(f"Loading Train & Val tiles (Train Cap: {max_train_pixels} px)...")
    
    X_train, y_train = load_and_flatten_segmentation_tiles(os.path.join(dataset_dir, "train"), debug=debug)
    X_val, y_val = load_and_flatten_segmentation_tiles(os.path.join(dataset_dir, "validation"), debug=debug)
    
    # Stratified downsampling for RAM protection (Train)
    if len(y_train) > max_train_pixels:
        print(f"      [Memory Manager] Stratified downsampling Train set to {max_train_pixels} pixels...")
        _, X_train, _, y_train = train_test_split(
            X_train, y_train, test_size=max_train_pixels, stratify=y_train, random_state=42
        )
        
    # Stratified downsampling for RAM protection (Validation - proportionally capped)
    val_cap = max_train_pixels // 4  
    if len(y_val) > val_cap:
        print(f"      [Memory Manager] Stratified downsampling Validation set to {val_cap} pixels...")
        _, X_val, _, y_val = train_test_split(
            X_val, y_val, test_size=val_cap, stratify=y_val, random_state=42
        )

    print("Fitting Scaler and projecting features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    
    y_train, y_val = y_train.astype(np.int32), y_val.astype(np.int32)
    
    # Save the scaler
    scaler_dir = os.path.join("..", "checkpoints", "scalers", modality)
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, f"scaler_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{patch_size}.joblib"))
    
    del X_train, X_val
    gc.collect()

    subset_classes = np.unique(y_train)
    taxonomy_key = f'Taxonomy_{taxonomy}'
    target_names = [h2crop_taxonomy_dict.get(taxonomy_key, {}).get(c, f"Class {c}") if c != 0 else "Background (0)" for c in subset_classes]

    # OPTUNA TUNING & TRAINING
    models_to_tune = ["decision_tree", "random_forest", "logistic_regression", "linear_svm"]
    active_trials = 2 if debug else n_trials
    
    print("\n--- INITIATING OPTUNA TUNING PASS ---")
    for algo_name in models_to_tune:
        gc.collect()
        if use_gpu:
            try:
                cp.get_default_memory_pool().free_all_blocks() 
            except Exception: pass
        
        # Run Optuna to find the best model configuration
        best_model, best_params = optimize_hyperparameters(
            model_name=algo_name,
            X_train=X_train_scaled, y_train=y_train,
            X_val=X_val_scaled, y_val=y_val,
            n_trials=active_trials,
            random_state=42,
            use_gpu=use_gpu
        )
        
        # Save Best Model and its Parameters
        checkpoint_dir = os.path.join("..", "checkpoints", algo_name.lower(), modality)
        os.makedirs(checkpoint_dir, exist_ok=True)
        joblib.dump(best_model, os.path.join(checkpoint_dir, f"{algo_name}_tiles_subset_{subset_id}_optuna.joblib"))
        
        algo_dir = os.path.join(results_out_dir, algo_name)
        os.makedirs(algo_dir, exist_ok=True)
        with open(os.path.join(algo_dir, f"best_params_subset_{subset_id}.json"), "w") as f:
            json.dump(best_params, f, indent=4)
            
        print(f"    Saved optimal checkpoint and parameters to disk.")
        
        del best_model
        gc.collect()

    # THE MEMORY PURGE
    print("\n[Memory Manager] Purging Train/Val data from RAM to prepare for evaluation...")
    del X_train_scaled, y_train, X_val_scaled, y_val
    gc.collect()

    # EVALUATION ONLY
    print("\n--- INITIATING EVALUATION PASS ---")
    test_files = glob.glob(os.path.join(dataset_dir, "test", "*.npz"))
    if debug: 
        test_files = test_files[:10]
        
    total_batches = (len(test_files) // test_batch_size) + 1

    for algo_name in models_to_tune:
        print(f"\n--> Evaluating {algo_name} on Test Set ({len(test_files)} total tiles)...")
        
        # Load the newly tuned optimal model
        model_filepath = os.path.join("..", "checkpoints", algo_name.lower(), modality, f"{algo_name}_tiles_subset_{subset_id}_optuna.joblib")
        model = joblib.load(model_filepath)
        global_cm = np.zeros((len(subset_classes), len(subset_classes)), dtype=np.int64)
        
        for batch_idx, i in enumerate(range(0, len(test_files), test_batch_size)):
            if batch_idx % 10 == 0: 
                print(f"      [Progress] Processing batch {batch_idx}/{total_batches}...")

            batch_paths = test_files[i:i+test_batch_size]
            X_batch_list, y_batch_list = [], []
            
            for f in batch_paths:
                with np.load(f, allow_pickle=False) as data:
                    X_img = data['X'].transpose(1, 2, 0).astype(np.float32)
                    X_batch_list.append(X_img.reshape(-1, X_img.shape[-1]))
                    y_batch_list.append(data['y'].flatten().astype(np.int32))
                
            X_batch = np.vstack(X_batch_list)
            y_batch = np.concatenate(y_batch_list)
            
            X_batch_scaled = scaler.transform(X_batch).astype(np.float32)
            y_pred = model.predict(X_batch_scaled)
            global_cm += confusion_matrix(y_batch, y_pred, labels=subset_classes)
            
            del X_batch_list, y_batch_list, X_batch, y_batch, X_batch_scaled, y_pred
            gc.collect()

        # RAM-SAFE CLASSIFICATION REPORT GENERATION
        print("      [Metrics] Calculating performance metrics directly from Confusion Matrix...")
        report_lines = [f"{'':<25} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>15}\n"]
        macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
        weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0
        total_support = np.sum(global_cm)
        
        for idx, target_name in enumerate(target_names):
            tp = global_cm[idx, idx]
            fp = global_cm[:, idx].sum() - tp
            fn = global_cm[idx, :].sum() - tp
            support = global_cm[idx, :].sum()
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            report_lines.append(f"{target_name:<25} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {support:>15}")
            macro_p += p; macro_r += r; macro_f1 += f1
            weighted_p += p * support; weighted_r += r * support; weighted_f1 += f1 * support
            
        num_classes = len(target_names)
        macro_p /= num_classes; macro_r /= num_classes; macro_f1 /= num_classes
        weighted_p /= total_support if total_support > 0 else 1
        weighted_r /= total_support if total_support > 0 else 1
        weighted_f1 /= total_support if total_support > 0 else 1
        accuracy = np.trace(global_cm) / total_support if total_support > 0 else 0.0
        
        report_lines.append(f"\n{'accuracy':<25} {'':>10} {'':>10} {accuracy:>10.4f} {total_support:>15}")
        report_lines.append(f"{'macro avg':<25} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {total_support:>15}")
        report_lines.append(f"{'weighted avg':<25} {weighted_p:>10.4f} {weighted_r:>10.4f} {weighted_f1:>10.4f} {total_support:>15}")
        
        algo_dir = os.path.join(results_out_dir, algo_name)
        with open(os.path.join(algo_dir, f"performance_subset_{subset_id}_optuna.txt"), "w") as f:
            f.write(f"--- Optuna Optimized Inference ---\nAlgorithm: {algo_name}\n\n" + "\n".join(report_lines))
            
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(confusion_matrix=global_cm, display_labels=target_names).plot(ax=ax, cmap='Blues', colorbar=False)
        plt.title(f"Confusion Matrix: {algo_name} (Optuna)\n({modality} | Subset {subset_id} | pSize {patch_size})")
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(algo_dir, f"confusion_matrix_subset_{subset_id}_optuna.png"), dpi=300)
        plt.close(fig)

        del model
        gc.collect()

    print(f"\nOptuna Pipeline completed successfully for {modality.upper()} Subset {subset_id}!")
