import os
import sys
import gc
import json
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from H2Crop.data_structures import h2crop_taxonomy_dict
from hyperparams_tuning.optimize_hyperparameters import optimize_hyperparameters
import cupy as cp

def extract_balanced_pixels_memory_safe(dataset_dir, target_samples_per_class, phase_name, debug=False):
    """Two-Pass Memory-Safe Extraction Algorithm for Train, Val, and Test sets."""
    files = glob.glob(os.path.join(dataset_dir, "*.npz"))
    if debug:
        files = files[:10]
        
    print(f"\n--- {phase_name}: Memory-Safe 1D Pixel Extraction ---")
    print(f"      [Pass 1] Scanning {len(files)} files to compute global class distribution...")
    
    y_arrays = []
    pixels_per_file = []
    for f in files:
        with np.load(f) as data:
            y_flat = data['y'].flatten().astype(np.int32)
            y_arrays.append(y_flat)
            pixels_per_file.append(len(y_flat))
            
    y_global = np.concatenate(y_arrays)
    unique_classes, class_counts = np.unique(y_global, return_counts=True)
    
    min_pixels_available = np.min(class_counts)
    actual_samples = min(target_samples_per_class, min_pixels_available)
    
    print(f"      [Balancing] Target: {target_samples_per_class} px/class | Bottleneck: {min_pixels_available} px")
    print(f"      [Balancing] Selectively targeting EXACTLY {actual_samples} completely random pixels per class.")
    
    np.random.seed(42)
    selected_global_indices = []
    for cls in unique_classes:
        cls_indices = np.where(y_global == cls)[0]
        sampled = np.random.choice(cls_indices, size=actual_samples, replace=False)
        selected_global_indices.append(sampled)
        
    selected_global_indices = np.concatenate(selected_global_indices)
    selected_global_indices.sort() 
    
    del y_global, y_arrays
    gc.collect()
    
    print(f"      [Pass 2] Selectively extracting only the {len(selected_global_indices)} targeted pixels from disk...")
    
    with np.load(files[0]) as data:
        num_features = data['X'].shape[0] 
        
    total_selected = len(selected_global_indices)
    X_balanced = np.zeros((total_selected, num_features), dtype=np.float32)
    y_balanced = np.zeros(total_selected, dtype=np.int32)
    
    current_global_offset = 0
    extracted_count = 0
    sg_idx = 0
    
    for f, num_pixels in zip(files, pixels_per_file):
        file_start = current_global_offset
        file_end = current_global_offset + num_pixels
        
        indices_in_file = []
        while sg_idx < total_selected and selected_global_indices[sg_idx] < file_end:
            indices_in_file.append(selected_global_indices[sg_idx] - file_start)
            sg_idx += 1
            
        if indices_in_file:
            with np.load(f) as data:
                X_img = data['X'].transpose(1, 2, 0).astype(np.float32)
                X_flat = X_img.reshape(-1, X_img.shape[-1])
                y_flat = data['y'].flatten().astype(np.int32)
                
                X_chunk = X_flat[indices_in_file]
                y_chunk = y_flat[indices_in_file]
                
                X_balanced[extracted_count : extracted_count + len(X_chunk)] = X_chunk
                y_balanced[extracted_count : extracted_count + len(y_chunk)] = y_chunk
                extracted_count += len(X_chunk)
                
        current_global_offset += num_pixels
        if sg_idx >= total_selected:
            break
            
    shuffle_mask = np.random.permutation(total_selected)
    X_balanced = np.ascontiguousarray(X_balanced[shuffle_mask])
    y_balanced = np.ascontiguousarray(y_balanced[shuffle_mask])
    
    print(f"      -> Extraction Complete! Final Shape: X={X_balanced.shape}, y={y_balanced.shape}")
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
    test_samples_per_class=1000,
    n_trials=20,
    debug=False
):
    if test_samples_per_class is None or val_samples_per_class is None:
        raise ValueError("Samples per class must be integers to ensure in-memory tuning safety.")

    print(f"\n{'='*70}")
    mode = "DEBUG MODE" if debug else "PRODUCTION MODE (OPTUNA IN-MEMORY)"
    print(f"STARTING ML SEGMENTATION PIPELINE FOR: {modality.upper()} | Subset {subset_id} | {mode}")
    print(f"{'='*70}")

    results_out_dir = os.path.join(save_results_dir, modality)
    os.makedirs(results_out_dir, exist_ok=True)
    
    # 1. LOAD & BALANCE TRAIN, VAL, AND TEST SETS
    X_train, y_train = extract_balanced_pixels_memory_safe(
        os.path.join(dataset_dir, "train"), train_samples_per_class, "Train", debug=debug
    )
    X_val, y_val = extract_balanced_pixels_memory_safe(
        os.path.join(dataset_dir, "validation"), val_samples_per_class, "Validation", debug=debug
    )
    X_test, y_test = extract_balanced_pixels_memory_safe(
        os.path.join(dataset_dir, "test"), test_samples_per_class, "Test", debug=debug
    )

    # 2. SCALE FEATURES
    print("\nFitting Scaler and scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)
    
    scaler_dir = os.path.join("..", "checkpoints", "scalers", modality)
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, f"scaler_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{patch_size}_optuna.joblib"))
    
    del X_train, X_val, X_test
    gc.collect()

    subset_classes = np.unique(y_train)
    taxonomy_key = f'Taxonomy_{taxonomy}'
    current_taxonomy = h2crop_taxonomy_dict.get(taxonomy_key, {})
    target_names = [current_taxonomy.get(c, f"Class {c}") if c != 0 else "Background (0)" for c in subset_classes]

    # 3. TUNING AND EVALUATION
    models_to_tune = ["decision_tree", "random_forest", "logistic_regression", "linear_svm"]
    cpu_only_models = ["logistic_regression", "linear_svm"]
    active_trials = 2 if debug else n_trials

    for algo_name in models_to_tune:
        print(f"\n{'-'*50}")
        print(f"PROCESSING ALGORITHM: {algo_name.upper()}")
        print(f"{'-'*50}")
        
        # Optimize Hyperparameters
        best_model, best_params = optimize_hyperparameters(
            model_name=algo_name,
            X_train=X_train_scaled, y_train=y_train,
            X_val=X_val_scaled, y_val=y_val,
            n_trials=active_trials,
            random_state=42,
            use_gpu=use_gpu
        )
        
        algo_dir = os.path.join(results_out_dir, algo_name)
        os.makedirs(algo_dir, exist_ok=True)
        with open(os.path.join(algo_dir, f"best_params_subset_{subset_id}.json"), "w") as f:
            json.dump(best_params, f, indent=4)

        # Evaluate strictly balanced Test Set
        print(f"--> Evaluating {algo_name} on balanced Test Set...")
        if use_gpu and algo_name not in cpu_only_models:
            X_test_gpu = cp.asarray(X_test_scaled)
            y_pred = best_model.predict(X_test_gpu)
            y_pred = cp.asnumpy(y_pred)
            del X_test_gpu
        else:
            y_pred = best_model.predict(X_test_scaled)
            
        global_cm = confusion_matrix(y_test, y_pred, labels=subset_classes)

        # Metrics & Reports
        print("      [Metrics] Calculating performance metrics...")
        report_lines = [f"{'':<25} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>15}\n"]
        macro_p, macro_r, macro_f1, total_support = 0.0, 0.0, 0.0, np.sum(global_cm)
        
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
            
        num_classes = len(target_names)
        macro_p /= num_classes; macro_r /= num_classes; macro_f1 /= num_classes
        accuracy = np.trace(global_cm) / total_support if total_support > 0 else 0.0
        
        report_lines.append(f"\n{'accuracy':<25} {'':>10} {'':>10} {accuracy:>10.4f} {total_support:>15}")
        report_lines.append(f"{'macro avg':<25} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {total_support:>15}")
        
        report = "\n".join(report_lines)
        with open(os.path.join(algo_dir, f"performance_subset_{subset_id}_optuna.txt"), "w") as f:
            f.write(f"--- Optuna Inference Complete ---\nAlgorithm: {algo_name}\n\n{report}")
            
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(confusion_matrix=global_cm, display_labels=target_names).plot(ax=ax, cmap='Blues', colorbar=False)
        plt.title(f"Confusion Matrix: {algo_name} (Optuna)\n({modality} | Subset {subset_id} | pSize {patch_size})")
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(algo_dir, f"confusion_matrix_subset_{subset_id}_optuna.png"), dpi=300)
        plt.close(fig)

        # Save model
        checkpoint_dir = os.path.join("..", "checkpoints", algo_name.lower(), modality)
        os.makedirs(checkpoint_dir, exist_ok=True)
        joblib.dump(best_model, os.path.join(checkpoint_dir, f"{algo_name}_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{patch_size}_optuna.joblib"))
        
        del best_model
        gc.collect()

    print(f"\nOptuna Pipeline completed successfully for {modality.upper()} Subset {subset_id}!")