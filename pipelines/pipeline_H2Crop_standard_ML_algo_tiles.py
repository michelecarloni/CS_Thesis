import os
import gc
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from H2Crop.data_structures import h2crop_taxonomy_dict
from tqdm import tqdm

def chunked_file_generator(files, chunk_size=500):
    # Using a Python generator (yield) instead of returning a massive list of lists.
    # This ensures we only hold one chunk of file paths in RAM at any given time,
    # which is crucial when iterating over massive 87GB test and train datasets.
    for i in range(0, len(files), chunk_size):
        yield files[i:i + chunk_size]

def load_chunk_into_memory(file_chunk):
    # This reads a subset of .npz files, transposes the spatial patches, 
    # and flattens them into a 2D matrix of individual pixels for standard ML.
    X_list, y_list = [], []
    for f in file_chunk:
        with np.load(f) as data:
            X_img = data['X'].transpose(1, 2, 0).astype(np.float32)
            X_flat = X_img.reshape(-1, X_img.shape[-1])
            y_flat = data['y'].flatten().astype(np.int32)
            X_list.append(X_flat)
            y_list.append(y_flat)
    
    return np.concatenate(X_list), np.concatenate(y_list)

def extract_balanced_pixels_memory_safe(files, target_samples_per_class, phase_name, debug=False):
    # Two-Pass Memory-Safe Extraction Algorithm for Capped Training Models (Decision Tree).
    # Since a standard Decision Tree cannot be batched, we must extract a memory-safe subset for it.
    
    if debug:
        files = files[:10]
        
    print(f"\n{phase_name}: Memory-Safe 1D Pixel Extraction (CAPPED for Decision Tree)")
    
    # PASS 1: Read only the labels (y) to calculate the global distribution without loading the heavy X features.
    print(f"Pass 1: Scanning {len(files)} files to compute global class distribution...")
    y_arrays = []
    pixels_per_file = []
    for f in files:
        with np.load(f) as data:
            y_flat = data['y'].flatten().astype(np.int32)
            y_arrays.append(y_flat)
            pixels_per_file.append(len(y_flat))
            
    y_global = np.concatenate(y_arrays)
    unique_classes, class_counts = np.unique(y_global, return_counts=True)
    
    # Calculate the actual cap. If the rarest class has fewer pixels than target_samples_per_class,
    # we bottleneck down to that number to ensure mathematically perfect class balance.
    min_pixels_available = np.min(class_counts)
    actual_samples = min(target_samples_per_class, min_pixels_available)
    
    print(f"Balancing Target: {target_samples_per_class} px/class | Bottleneck: {min_pixels_available} px")
    
    # Select the exact random global indices we want to keep for each class.
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
    
    # PASS 2: Iterate through the files again, loading the heavy X features 
    # but ONLY extracting the specific rows that match our pre-selected random indices.
    print(f"Pass 2: Extracting only the {len(selected_global_indices)} targeted pixels from disk...")
    
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
            
    # Final shuffle to ensure classes are mixed before feeding to standard ML algorithms
    shuffle_mask = np.random.permutation(total_selected)
    X_balanced = np.ascontiguousarray(X_balanced[shuffle_mask])
    y_balanced = np.ascontiguousarray(y_balanced[shuffle_mask])
    
    return X_balanced, y_balanced


def pipeline_H2Crop_standard_ML_algo_tiles(
    save_results_dir, 
    dataset_dir, 
    subset_id, 
    subset_classes, # <--- NEW PARAMETER
    modality, 
    taxonomy=3, 
    patch_size=32, 
    use_gpu=False, # Hardcoded to False as we are doing full CPU Out-Of-Core streaming
    train_samples_per_class=100000,
    debug=False
):
    mode = "DEBUG MODE" if debug else "PRODUCTION MODE (CPU OUT-OF-CORE BATCHING)"
    print(f"\nSTARTING ML SEGMENTATION PIPELINE FOR: {modality.upper()} | Subset {subset_id} | {mode}")

    results_out_dir = os.path.join(save_results_dir, modality)
    os.makedirs(results_out_dir, exist_ok=True)
    
    train_files = glob.glob(os.path.join(dataset_dir, "train", "*.npz"))
    test_files = glob.glob(os.path.join(dataset_dir, "test", "*.npz"))
    
    if debug:
        train_files, test_files = train_files[:10], test_files[:10]

    # Explicitly define classes (Background 0 + Target Crops) instead of scanning the disk
    subset_classes = np.array([0] + sorted(subset_classes))
    
    taxonomy_key = f'Taxonomy_{taxonomy}'
    current_taxonomy = h2crop_taxonomy_dict.get(taxonomy_key, {}) 
    target_names = [current_taxonomy.get(c, f"Class {c}") if c != 0 else "Background (0)" for c in subset_classes]

    # Phase 1: Out-of-Core Scaler Fitting
    # We use partial_fit to compute the mean and standard deviation incrementally.
    # This prevents the need to load the entire dataset into memory just to scale it.
    print("\nPhase 1: Streaming Global Feature Scaling")
    scaler = StandardScaler()
    
    for chunk in tqdm(list(chunked_file_generator(train_files, chunk_size=500)), desc="Fitting Scaler (Batches)"):
        X_chunk, _ = load_chunk_into_memory(chunk)
        scaler.partial_fit(X_chunk)
        del X_chunk
        gc.collect()
        
    scaler_dir = os.path.join("..", "checkpoints", "scalers", modality)
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, f"scaler_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{patch_size}.joblib"))

    # Phase 2: Capped Data Extraction (For Decision Tree Only)
    # The Decision Tree is the only model that mathematically cannot learn in incremental batches.
    # We extract a memory-safe subset specifically for it.
    X_train_capped, y_train_capped = extract_balanced_pixels_memory_safe(
        train_files, train_samples_per_class, "Train", debug=debug
    )
    X_train_capped = scaler.transform(X_train_capped).astype(np.float32)

    # Phase 3: Model Initialization
    # We define configurations mapped to specific training strategies (capped vs batched).
    # SGDClassifier(loss='log_loss') mathematically approximates Logistic Regression and supports partial_fit.
    # SGDClassifier(loss='hinge') mathematically approximates Linear SVM and supports partial_fit.
    model_configs = {
        "logistic_regression": {"type": "batched_linear", "fn": lambda: SGDClassifier(loss='log_loss', n_jobs=-1, random_state=42)},
        "linear_svm": {"type": "batched_linear", "fn": lambda: SGDClassifier(loss='hinge', n_jobs=-1, random_state=42)},
        "decision_tree": {"type": "capped", "fn": lambda: DecisionTreeClassifier(max_depth=15, class_weight='balanced', random_state=42)},
        "random_forest": {"type": "batched_tree", "fn": lambda: RandomForestClassifier(n_estimators=0, max_depth=15, warm_start=True, class_weight='balanced', n_jobs=-1, random_state=42)}
    }

    # Phase 4: Algorithm Pass
    for algo_name, config in model_configs.items():
        print(f"\nPROCESSING ALGORITHM: {algo_name.upper()} | Type: {config['type']}")
        
        model = config["fn"]()
        
        # PATH A: BATCHED OUT-OF-CORE LEARNING (SGD Linear Models)
        if config['type'] == 'batched_linear':
            print(f"Streaming Training Set for {algo_name}...")
            for chunk in tqdm(list(chunked_file_generator(train_files, chunk_size=500)), desc="Training Batches"):
                X_chunk, y_chunk = load_chunk_into_memory(chunk)
                X_chunk = scaler.transform(X_chunk).astype(np.float32)
                
                # partial_fit requires the list of all possible classes
                model.partial_fit(X_chunk, y_chunk, classes=subset_classes)
                
                del X_chunk, y_chunk
                gc.collect()

        # PATH B: BATCHED OUT-OF-CORE LEARNING (Random Forest)
        elif config['type'] == 'batched_tree':
            total_estimators = 150
            chunks = list(chunked_file_generator(train_files, chunk_size=500))
            
            # Distribute the 150 trees evenly across all data chunks
            trees_per_chunk = max(1, total_estimators // len(chunks))
            
            print(f"Streaming Training Set ({len(chunks)} chunks) for {algo_name}...")
            for chunk in tqdm(chunks, desc="Training Batches"):
                X_chunk, y_chunk = load_chunk_into_memory(chunk)
                X_chunk = scaler.transform(X_chunk).astype(np.float32)
                
                # --- SYNTHETIC CLASS INJECTION ---
                # To prevent broadcasting mismatch errors in Scikit-Learn during evaluation, 
                # we must guarantee that the Random Forest sees every single global class in this chunk.
                # We achieve this by copying the first pixel and artificially assigning it to all possible labels.
                dummy_X = np.tile(X_chunk[0], (len(subset_classes), 1))
                X_chunk = np.vstack((X_chunk, dummy_X))
                y_chunk = np.concatenate((y_chunk, subset_classes))
                #----------------------------------

                # Expand the forest by the allocated trees for this specific chunk
                model.n_estimators += trees_per_chunk
                model.fit(X_chunk, y_chunk)
                
                del X_chunk, y_chunk
                gc.collect()
                
        # PATH C: CAPPED IN-MEMORY LEARNING (Decision Tree Only)
        elif config['type'] == 'capped':
            print(f"Training {algo_name} on the capped, memory-safe dataset...")
            model.fit(X_train_capped, y_train_capped)

        # Phase 5: Unified Streaming Evaluation
        # We stream the entire test set for ALL models (even the capped Decision Tree) 
        # to guarantee evaluation on the full 87GB scale without OOM risks.
        print(f"Evaluating {algo_name} on the FULL uncapped test set...")
        global_cm = np.zeros((len(subset_classes), len(subset_classes)), dtype=np.int64)
        
        for chunk in tqdm(list(chunked_file_generator(test_files, chunk_size=500)), desc="Evaluating Batches"):
            X_chunk, y_chunk = load_chunk_into_memory(chunk)
            X_chunk = scaler.transform(X_chunk).astype(np.float32)
            
            y_pred = model.predict(X_chunk)
            batch_cm = confusion_matrix(y_chunk, y_pred, labels=subset_classes)
            global_cm += batch_cm
            
            del X_chunk, y_chunk, y_pred
            gc.collect()

        # Phase 6: Metrics & Reports
        print("Calculating performance metrics from Confusion Matrix...")
        
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
            f.write(f"Inference Complete\nAlgorithm: {algo_name}\n\n")
            f.write(f"Test Set Classification Report\n{report}")
            
        matrix_path = os.path.join(algo_dir, f"confusion_matrix_subset_{subset_id}.png")
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(confusion_matrix=global_cm, display_labels=target_names).plot(ax=ax, cmap='Blues', colorbar=False)
        plt.title(f"Confusion Matrix: {algo_name}\n({modality} | Subset {subset_id} | pSize {patch_size})")
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.tight_layout()
        plt.savefig(matrix_path, dpi=300)
        plt.close(fig)

        checkpoint_dir = os.path.join("..", "checkpoints", algo_name.lower(), modality)
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_filepath = os.path.join(checkpoint_dir, f"{algo_name}_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{patch_size}.joblib")
        
        joblib.dump(model, model_filepath)
        print(f"Saved checkpoint to: {model_filepath}")
        
        del model
        gc.collect()

    print(f"\nPipeline completed successfully for {modality.upper()} Subset {subset_id}!")