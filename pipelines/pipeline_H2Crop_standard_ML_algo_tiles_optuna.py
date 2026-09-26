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
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from H2Crop.data_structures import h2crop_taxonomy_dict
from hyperparams_tuning.optimize_hyperparameters import optimize_hyperparameters
from utils import load_and_flatten_segmentation_tiles
import cupy as cp

def balance_1d_pixels(X, y, target_samples_per_class, phase_name):
    """Helper function to apply strict bottleneck-driven pixel balancing."""
    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_pixels_available = np.min(class_counts)
    actual_samples = min(target_samples_per_class, min_pixels_available)
    
    print(f"      [{phase_name} Balancing] Target: {target_samples_per_class} px/class")
    print(f"      [{phase_name} Balancing] Bottleneck: {min_pixels_available} px")
    print(f"      [{phase_name} Balancing] Extracting EXACTLY {actual_samples} px per class.")
    
    balanced_indices = []
    np.random.seed(42)
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        sampled_indices = np.random.choice(cls_indices, size=actual_samples, replace=False)
        balanced_indices.append(sampled_indices)
        
    balanced_indices = np.concatenate(balanced_indices)
    np.random.shuffle(balanced_indices)
    
    X_balanced = np.ascontiguousarray(X[balanced_indices])
    y_balanced = np.ascontiguousarray(y[balanced_indices])
    
    return X_balanced, y_balanced

def pipeline_H2Crop_standard_ML_algo_tiles_optuna(
    save_results_dir, 
    dataset_dir, 
    subset_id, 
    modality, 
    taxonomy=3, 
    patch_size=32, 
    use_gpu=True, 
    train_samples_per_class=100000,
    val_samples_per_class=25000,
    test_samples_per_class=None,
    n_trials=20,
    test_batch_size=50, 
    debug=False
):
    print(f"\n{'='*70}")
    mode = "DEBUG MODE" if debug else "PRODUCTION MODE (OPTUNA - 1D BALANCED)"
    print(f"STARTING ML SEGMENTATION PIPELINE FOR: {modality.upper()} | Subset {subset_id} | {mode}")
    print(f"{'='*70}")

    results_out_dir = os.path.join(save_results_dir, modality)
    os.makedirs(results_out_dir, exist_ok=True)
    
    # ==========================================
    # 1. LOAD & BALANCE TRAIN & VAL SETS
    # ==========================================
    if debug:
        print("Loading Train & Val tiles (DEBUG MODE: Reading only 10 files)...")
    else:
        print(f"Loading Train & Val tiles for strict pixel extraction...")
    
    X_train_raw, y_train_raw = load_and_flatten_segmentation_tiles(os.path.join(dataset_dir, "train"), debug=debug)
    X_val_raw, y_val_raw = load_and_flatten_segmentation_tiles(os.path.join(dataset_dir, "validation"), debug=debug)
    
    print("\n--- Physical Pixel Balancing ---")
    X_train, y_train = balance_1d_pixels(X_train_raw, y_train_raw, train_samples_per_class, "Train")
    del X_train_raw, y_train_raw
    
    X_val, y_val = balance_1d_pixels(X_val_raw, y_val_raw, val_samples_per_class, "Validation")
    del X_val_raw, y_val_raw
    gc.collect()

    # ==========================================
    # 2. SCALE FEATURES
    # ==========================================
    print("\nFitting Scaler and scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    
    scaler_dir = os.path.join("..", "checkpoints", "scalers", modality)
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, f"scaler_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{patch_size}.joblib"))
    
    del X_train, X_val
    gc.collect()

    subset_classes = np.unique(y_train)
    taxonomy_key = f'Taxonomy_{taxonomy}'
    target_names = [h2crop_taxonomy_dict.get(taxonomy_key, {}).get(c, f"Class {c}") if c != 0 else "Background (0)" for c in subset_classes]

    # ==========================================
    # 3. OPTUNA TUNING & TRAINING
    # ==========================================
    models_to_tune = ["decision_tree", "random_forest", "logistic_regression", "linear_svm"]
    active_trials = 2 if debug else n_trials
    
    print("\n--- INITIATING OPTUNA TUNING PASS ---")
    best_models = {}  # Keep optimal models alive in RAM to bypass cuBLAS bugs
    
    for algo_name in models_to_tune:
        gc.collect()
        if use_gpu:
            try:
                cp.get_default_memory_pool().free_all_blocks() 
            except Exception: pass
        
        best_model, best_params = optimize_hyperparameters(
            model_name=algo_name,
            X_train=X_train_scaled, y_train=y_train,
            X_val=X_val_scaled, y_val=y_val,
            n_trials=active_trials,
            random_state=42,
            use_gpu=use_gpu
        )
        
        # Store in RAM for immediate inference
        best_models[algo_name] = best_model
        
        algo_dir = os.path.join(results_out_dir, algo_name)
        os.makedirs(algo_dir, exist_ok=True)
        with open(os.path.join(algo_dir, f"best_params_subset_{subset_id}.json"), "w") as f:
            json.dump(best_params, f, indent=4)
            
        print(f"    Tuning complete for {algo_name}. Parameters saved.")

    print("\n[Memory Manager] Purging Train/Val data from RAM to prepare for evaluation...")
    del X_train_scaled, y_train, X_val_scaled, y_val
    gc.collect()

    # ==========================================
    # 4. PRE-PROCESS TEST DATA
    # ==========================================
    print("\n--- PREPARING EVALUATION DATA ---")
    X_test_balanced, y_test_balanced = None, None
    test_files = []
    
    if test_samples_per_class is not None:
        print(f"[Memory Manager] Loading ALL test tiles for strict pixel balancing...")
        X_test_raw, y_test_raw = load_and_flatten_segmentation_tiles(os.path.join(dataset_dir, "test"), debug=debug)
        X_test_balanced, y_test_balanced = balance_1d_pixels(X_test_raw, y_test_raw, test_samples_per_class, "Test")
        del X_test_raw, y_test_raw
        gc.collect()
    else:
        print(f"[Memory Manager] Preparing imbalanced spatial data for streaming batches...")
        test_files = glob.glob(os.path.join(dataset_dir, "test", "*.npz"))
        random.seed(42)
        random.shuffle(test_files)
        if debug: test_files = test_files[:10]

    # ==========================================
    # 5. EVALUATION EXECUTION
    # ==========================================
    total_batches = (len(test_files) // test_batch_size) + 1 if not test_samples_per_class else 1

    for algo_name in models_to_tune:
        mode_str = "Balanced" if test_samples_per_class else "Imbalanced Streaming"
        print(f"\n--> Evaluating {algo_name} on Test Set ({mode_str})...")
        
        # Pull model directly from RAM
        model = best_models[algo_name]
        global_cm = np.zeros((len(subset_classes), len(subset_classes)), dtype=np.int64)
        
        if test_samples_per_class is not None:
            # 5A. Evaluate strictly balanced in-memory Test Set
            for i in range(0, len(X_test_balanced), test_batch_size * 1000):
                X_batch = X_test_balanced[i:i + test_batch_size * 1000]
                y_batch = y_test_balanced[i:i + test_batch_size * 1000]
                
                X_batch_scaled = np.ascontiguousarray(scaler.transform(X_batch).astype(np.float32))
                
                # Check if it's a cuML model (has a 'predict' method that might prefer CuPy)
                # But since Optuna's returned model handles types dynamically, standard predict is generally safe.
                # If you pinned LR/SVM to CPU in optimize_hyperparameters.py, they handle NumPy natively.
                if use_gpu and algo_name not in ["decision_tree", "logistic_regression", "linear_svm"]:
                    X_batch_gpu = cp.asarray(X_batch_scaled)
                    y_pred = model.predict(X_batch_gpu)
                    y_pred = cp.asnumpy(y_pred)
                    del X_batch_gpu
                else:
                    y_pred = model.predict(X_batch_scaled)
                    
                global_cm += confusion_matrix(y_batch, y_pred, labels=subset_classes)
                
        else:
            # 5B. Evaluate imbalanced streaming Test Set directly from disk
            for batch_idx, i in enumerate(range(0, len(test_files), test_batch_size)):
                if batch_idx % 10 == 0:
                    print(f"      [Progress] Processing batch {batch_idx}/{total_batches}...")

                batch_paths = test_files[i:i+test_batch_size]
                X_batch_list, y_batch_list = [], []
                
                for f in batch_paths:
                    with np.load(f) as data:
                        X_img = data['X'].transpose(1, 2, 0).astype(np.float32)
                        X_batch_list.append(X_img.reshape(-1, X_img.shape[-1]))
                        y_batch_list.append(data['y'].flatten().astype(np.int32))
                    
                X_batch = np.vstack(X_batch_list)
                y_batch = np.concatenate(y_batch_list)
                
                X_batch_scaled = np.ascontiguousarray(scaler.transform(X_batch).astype(np.float32))
                
                if use_gpu and algo_name not in ["decision_tree", "logistic_regression", "linear_svm"]:
                    X_batch_gpu = cp.asarray(X_batch_scaled)
                    y_pred = model.predict(X_batch_gpu)
                    y_pred = cp.asnumpy(y_pred)
                    del X_batch_gpu
                else:
                    y_pred = model.predict(X_batch_scaled)
                
                global_cm += confusion_matrix(y_batch, y_pred, labels=subset_classes)

        # ==========================================
        # 6. METRICS & REPORTS
        # ==========================================
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
        
        report = "\n".join(report_lines)
                    
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

        # 7. SAVE TO DISK AND PURGE
        checkpoint_dir = os.path.join("..", "checkpoints", algo_name.lower(), modality)
        os.makedirs(checkpoint_dir, exist_ok=True)
        joblib.dump(model, os.path.join(checkpoint_dir, f"{algo_name}_tiles_subset_{subset_id}_optuna.joblib"))
        
        del best_models[algo_name]
        del model
        gc.collect()

    print(f"\nOptuna Pipeline completed successfully for {modality.upper()} Subset {subset_id}!")