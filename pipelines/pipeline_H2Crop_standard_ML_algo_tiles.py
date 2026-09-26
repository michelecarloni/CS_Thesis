import os
import gc
import glob
import joblib
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from H2Crop.data_structures import h2crop_taxonomy_dict
from utils import load_and_flatten_segmentation_tiles
import cupy as cp

# NVIDIA RAPIDS cuML (GPU Models - Logistic Regression and Linear SVM omitted)
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
except ImportError:
    pass

def pipeline_H2Crop_standard_ML_algo_tiles(
    save_results_dir, 
    dataset_dir, 
    subset_id, 
    modality, 
    taxonomy=3, 
    patch_size=32, 
    use_gpu=True, 
    train_samples_per_class=100000,
    test_samples_per_class=None,
    test_batch_size=50, 
    debug=False
):
    """
    Trains standard Machine Learning algorithms on pixel-wise tile data.
    
    Implements a strict bottleneck-driven undersampling strategy to extract perfectly 
    balanced 1D pixel arrays, preventing models from collapsing on the Background class.
    Forces Logistic Regression, Linear SVM, and Decision Tree to run on CPU to bypass 
    cuBLAS crashes, while optionally accelerating Random Forest on the GPU.
    """
    print(f"\n{'='*70}")
    mode = "DEBUG MODE" if debug else "PRODUCTION MODE (1D BALANCED)"
    print(f"STARTING ML SEGMENTATION PIPELINE FOR: {modality.upper()} | Subset {subset_id} | {mode}")
    print(f"{'='*70}")

    results_out_dir = os.path.join(save_results_dir, modality)
    os.makedirs(results_out_dir, exist_ok=True)
    
    # ==========================================
    # 1. LOAD & BALANCE TRAIN TILES
    # ==========================================
    if debug:
        print("Loading Train tiles (DEBUG MODE: Reading only 10 files)...")
    else:
        print(f"Loading Train tiles for strict pixel extraction...")
        
    X_train, y_train = load_and_flatten_segmentation_tiles(os.path.join(dataset_dir, "train"), debug=debug)
    
    print("\n--- Physical Pixel Balancing (Training) ---")
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    min_pixels_available = np.min(class_counts)
    
    actual_train_samples = min(train_samples_per_class, min_pixels_available)
    
    print(f"      [Balancing Manager] Target: {train_samples_per_class} pixels/class")
    print(f"      [Balancing Manager] Bottleneck (rarest class) has: {min_pixels_available} pixels")
    print(f"      [Balancing Manager] Extracting EXACTLY {actual_train_samples} random pixels per class.")
    
    balanced_indices = []
    np.random.seed(42)
    
    for cls in unique_classes:
        cls_indices = np.where(y_train == cls)[0]
        sampled_indices = np.random.choice(cls_indices, size=actual_train_samples, replace=False)
        balanced_indices.append(sampled_indices)
        
    balanced_indices = np.concatenate(balanced_indices)
    np.random.shuffle(balanced_indices) 
    
    X_train = np.ascontiguousarray(X_train[balanced_indices])
    y_train = np.ascontiguousarray(y_train[balanced_indices])
    
    print(f"Final perfectly balanced Train shape: X={X_train.shape}, y={y_train.shape}")

    # ==========================================
    # 2. SCALE FEATURES
    # ==========================================
    print("\nFitting Scaler and scaling Train features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    y_train = y_train.astype(np.int32)
    
    scaler_dir = os.path.join("..", "checkpoints", "scalers", modality)
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_filename = f"scaler_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{patch_size}.joblib"
    joblib.dump(scaler, os.path.join(scaler_dir, scaler_filename))
    
    del X_train
    gc.collect()

    # ==========================================
    # 3. PRE-PROCESS TEST DATA
    # ==========================================
    print("\n--- PREPARING EVALUATION DATA ---")
    X_test_balanced, y_test_balanced = None, None
    test_files = []
    
    if test_samples_per_class is not None:
        print(f"[Memory Manager] Loading ALL test tiles for strict pixel balancing (Target: {test_samples_per_class}/class)...")
        X_test_raw, y_test_raw = load_and_flatten_segmentation_tiles(os.path.join(dataset_dir, "test"), debug=debug)
        
        unique_classes_test, class_counts_test = np.unique(y_test_raw, return_counts=True)
        min_pixels_test = np.min(class_counts_test)
        actual_test_samples = min(test_samples_per_class, min_pixels_test)
        
        print(f"      [Balancing Manager] Test Bottleneck: {min_pixels_test} pixels.")
        print(f"      [Balancing Manager] Extracting EXACTLY {actual_test_samples} random pixels per class.")
        
        balanced_test_indices = []
        for cls in unique_classes_test:
            cls_indices = np.where(y_test_raw == cls)[0]
            sampled_indices = np.random.choice(cls_indices, size=actual_test_samples, replace=False)
            balanced_test_indices.append(sampled_indices)
            
        balanced_test_indices = np.concatenate(balanced_test_indices)
        np.random.shuffle(balanced_test_indices)
        
        X_test_balanced = np.ascontiguousarray(X_test_raw[balanced_test_indices])
        y_test_balanced = np.ascontiguousarray(y_test_raw[balanced_test_indices])
        
        del X_test_raw, y_test_raw
        gc.collect()
        
        print(f"Final perfectly balanced Test shape: X={X_test_balanced.shape}, y={y_test_balanced.shape}")
    else:
        print(f"[Memory Manager] Preparing imbalanced spatial data for streaming batches...")
        test_files = glob.glob(os.path.join(dataset_dir, "test", "*.npz"))
        random.seed(42)
        random.shuffle(test_files)
        
        if debug:
            test_files = test_files[:10]

    # ==========================================
    # 4. LAZY MODEL INITIALIZATION
    # ==========================================
    # Force Decision Tree, Logistic Regression, and Linear SVM to always use CPU
    model_configs = {
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=15, random_state=42),
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
        "linear_svm": lambda: LinearSVC(max_iter=1000, dual=False, random_state=42)
    }
    
    if use_gpu:
        model_configs["random_forest"] = lambda: cuRF(n_estimators=150, max_depth=15, max_features='sqrt', random_state=42)
    else:
        model_configs["random_forest"] = lambda: RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=-1, random_state=42)

    subset_classes = np.unique(y_train)
    taxonomy_key = f'Taxonomy_{taxonomy}'
    current_taxonomy = h2crop_taxonomy_dict.get(taxonomy_key, {})
    target_names = [current_taxonomy.get(c, f"Class {c}") if c != 0 else "Background (0)" for c in subset_classes]
    
    cpu_only_models = ["decision_tree", "logistic_regression", "linear_svm"]

    # ==========================================
    # 5. LINEAR ALGORITHM PASS (Train -> Eval -> Save)
    # ==========================================
    total_batches = (len(test_files) // test_batch_size) + 1 if not test_samples_per_class else 1

    for algo_name, model_fn in model_configs.items():
        print(f"\n{'-'*50}")
        print(f"PROCESSING ALGORITHM: {algo_name.upper()}")
        print(f"{'-'*50}")
        
        # 5A. TRAIN
        print(f"--> Training {algo_name}...")
        model = model_fn()
        
        # Manually push data to GPU for cuML models only
        if use_gpu and algo_name not in cpu_only_models:
            X_train_gpu = cp.asarray(X_train_scaled)
            model.fit(X_train_gpu, y_train)
            del X_train_gpu
        else:
            model.fit(X_train_scaled, y_train)

        # 5B. EVALUATE
        mode_str = "Balanced" if test_samples_per_class else "Imbalanced Streaming"
        print(f"--> Evaluating {algo_name} on Test Set ({mode_str})...")
        
        global_cm = np.zeros((len(subset_classes), len(subset_classes)), dtype=np.int64)
        
        if test_samples_per_class is not None:
            # Evaluate strictly balanced in-memory Test Set
            for i in range(0, len(X_test_balanced), test_batch_size * 1000):
                X_batch = X_test_balanced[i:i + test_batch_size * 1000]
                y_batch = y_test_balanced[i:i + test_batch_size * 1000]
                
                X_batch_scaled = np.ascontiguousarray(scaler.transform(X_batch).astype(np.float32))
                
                if use_gpu and algo_name not in cpu_only_models:
                    X_batch_gpu = cp.asarray(X_batch_scaled)
                    y_pred = model.predict(X_batch_gpu)
                    y_pred = cp.asnumpy(y_pred)
                    del X_batch_gpu
                else:
                    y_pred = model.predict(X_batch_scaled)
                    
                global_cm += confusion_matrix(y_batch, y_pred, labels=subset_classes)
                
        else:
            # Evaluate imbalanced streaming Test Set directly from disk
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
                
                if use_gpu and algo_name not in cpu_only_models:
                    X_batch_gpu = cp.asarray(X_batch_scaled)
                    y_pred = model.predict(X_batch_gpu)
                    y_pred = cp.asnumpy(y_pred)
                    del X_batch_gpu
                else:
                    y_pred = model.predict(X_batch_scaled)
                
                global_cm += confusion_matrix(y_batch, y_pred, labels=subset_classes)

        # 5C. METRICS & REPORTS
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
        os.makedirs(algo_dir, exist_ok=True)
        
        report_path = os.path.join(algo_dir, f"performance_subset_{subset_id}.txt")
        with open(report_path, "w") as f:
            f.write(f"--- Inference Complete ---\nAlgorithm: {algo_name}\n\n")
            f.write(f"--- Test Set Classification Report ---\n{report}")
            
        matrix_path = os.path.join(algo_dir, f"confusion_matrix_subset_{subset_id}.png")
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(confusion_matrix=global_cm, display_labels=target_names).plot(ax=ax, cmap='Blues', colorbar=False)
        plt.title(f"Confusion Matrix: {algo_name}\n({modality} | Subset {subset_id} | pSize {patch_size})")
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.tight_layout()
        plt.savefig(matrix_path, dpi=300)
        plt.close(fig)

        # 5D. SAVE TO DISK AND PURGE
        checkpoint_dir = os.path.join("..", "checkpoints", algo_name.lower(), modality)
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_filepath = os.path.join(checkpoint_dir, f"{algo_name}_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{patch_size}.joblib")
        
        joblib.dump(model, model_filepath)
        print(f"    Saved checkpoint to: {model_filepath}")
        
        del model
        gc.collect()

    print(f"\nPipeline completed successfully for {modality.upper()} Subset {subset_id}!")