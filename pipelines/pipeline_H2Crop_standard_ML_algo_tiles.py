import os
import gc
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from H2Crop.data_structures import h2crop_taxonomy_dict
from utils import load_and_flatten_segmentation_tiles
from sklearn.tree import DecisionTreeClassifier
import cupy as cp
# NVIDIA RAPIDS cuML (GPU Models)
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLR
    from cuml.svm import LinearSVC as cuSVC
    from cuml.linear_model import MBSGDClassifier as cuMBSGD
except ImportError:
    pass




# The following 3 functions are for:
# - training the first 4 standard ML models over the tiles
# - same as before but with optuna
# - Trainining the Depp learning models that will be compared to the first 4 standard ML models
def pipeline_H2Crop_standard_ML_algo_tiles(
    save_results_dir, 
    dataset_dir, 
    subset_id, 
    modality, 
    taxonomy=3, 
    patch_size=32, 
    use_gpu=True, 
    max_train_pixels=500000,
    test_batch_size=50, 
    debug=False
):
    """
    Trains standard Machine Learning algorithms on pixel-wise tile data using a 
    memory-safe Two-Pass Architecture (Train first, purge RAM, then Evaluate).
    
    Metrics are calculated directly from the confusion matrix to bypass OOM crashes
    caused by massive dummy list allocations during inference on millions of pixels.

    Arguments:
    - save_results_dir (str): The base directory path where evaluation metrics, reports, and confusion matrices will be saved.
    - dataset_dir (str): The directory containing the extracted .npz tiles divided into 'train', 'validation', and 'test' subfolders.
    - subset_id (int): The identifier for the current crop subset being processed (e.g., 1, 2, 3, or 4).
    - modality (str): The type of satellite data being processed ('hyperspectral' or 'multispectral').
    - taxonomy (int): The taxonomic hierarchical level used for mapping the class labels (default: 3).
    - patch_size (int): The height and width of the square image tiles being processed (default: 32).
    - use_gpu (bool): If True, attempts to use NVIDIA cuML for GPU-accelerated model training. Falls back to CPU if False.
    - max_train_pixels (int): The absolute maximum number of pixels to load into memory for training. Uses stratified sampling to balance classes.
    - test_batch_size (int): The number of .npz files to load simultaneously during the batched evaluation phase.
    - debug (bool): If True, artificially restricts the dataset to just 10 files to rapidly test the pipeline plumbing without waiting.
    """
    print(f"\n{'='*70}")
    mode = "DEBUG MODE" if debug else "PRODUCTION MODE"
    print(f"STARTING ML SEGMENTATION PIPELINE FOR: {modality.upper()} | Subset {subset_id} | {mode}")
    print(f"{'='*70}")

    # Set the results output directory directly under save_results_dir
    results_out_dir = os.path.join(save_results_dir, modality)
    os.makedirs(results_out_dir, exist_ok=True)
    
    # CAPPED TRAINING LOADING & STRATIFIED DOWNSAMPLING
    if debug:
        print("Loading Train tiles (DEBUG MODE: Reading only 10 files)...")
    else:
        print(f"Loading Train tiles (Safety cap set to {max_train_pixels} pixels)...")
        
    X_train, y_train = load_and_flatten_segmentation_tiles(os.path.join(dataset_dir, "train"), debug=debug)
    
    # Apply memory safety cap with Stratified Sampling to preserve perfect class balance
    if len(y_train) > max_train_pixels:
        print(f"      [Memory Manager] Stratified downsampling from {len(y_train)} to {max_train_pixels} pixels...")
        _, X_train, _, y_train = train_test_split(
            X_train, y_train, 
            test_size=max_train_pixels, 
            stratify=y_train, 
            random_state=42
        )

    print(f"Final Train Pixels for fitting: {len(y_train)}")
    print("Fitting Scaler and scaling Train features...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    y_train = y_train.astype(np.int32)
    
    # Save the scaler to a centralized checkpoints directory
    scaler_dir = os.path.join("..", "checkpoints", "scalers", modality)
    os.makedirs(scaler_dir, exist_ok=True)
    scaler_filename = f"scaler_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{patch_size}.joblib"
    joblib.dump(scaler, os.path.join(scaler_dir, scaler_filename))
    
    del X_train
    gc.collect()

    # LAZY MODEL INITIALIZATION
    # We use lambda functions so the models don't exist in memory until we explicitly call them.
    # max_depth is reduced to 15 to prevent exponential RAM bloat from leaf nodes.
    model_configs = {
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=15, random_state=42)
    }
    
    if use_gpu:
        model_configs["random_forest"] = lambda: cuRF(n_estimators=150, max_depth=15, max_features='sqrt', random_state=42)
        model_configs["logistic_regression"] = lambda: cuLR(max_iter=1000)
        model_configs["linear_svm"] = lambda: cuSVC(max_iter=1000, penalty='l2')
    else:
        model_configs["random_forest"] = lambda: RandomForestClassifier(n_estimators=100, max_depth=15, n_jobs=1, random_state=42)
        model_configs["logistic_regression"] = lambda: LogisticRegression(max_iter=1000, n_jobs=1, random_state=42)
        model_configs["linear_svm"] = lambda: LinearSVC(max_iter=1000, dual=False, random_state=42)

    subset_classes = np.unique(y_train)
    taxonomy_key = f'Taxonomy_{taxonomy}'
    current_taxonomy = h2crop_taxonomy_dict.get(taxonomy_key, {})
    target_names = [current_taxonomy.get(c, f"Class {c}") if c != 0 else "Background (0)" for c in subset_classes]

    # TRAINING ONLY
    print("\n--- INITIATING TRAINING PASS ---")
    for algo_name, model_fn in model_configs.items():
        print(f"--> Training and saving {algo_name}...")
        gc.collect()
        if use_gpu:
            try:
                cp.get_default_memory_pool().free_all_blocks() 
                cp.get_default_pinned_memory_pool().free_all_blocks() 
            except Exception:
                pass
        
        model = model_fn() # Instantiate model
        model.fit(X_train_scaled, y_train) # Train model
        
        # Save model checkpoint to its specific algorithm directory
        checkpoint_dir = os.path.join("..", "checkpoints", algo_name.lower(), modality)
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_filepath = os.path.join(checkpoint_dir, f"{algo_name}_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{patch_size}.joblib")
        
        joblib.dump(model, model_filepath) # Serialize to disk
        print(f"    Saved checkpoint to: {model_filepath}")
        
        # Immediately destroy the trained model from RAM
        del model
        gc.collect()

    # THE MEMORY PURGE
    print("\n[Memory Manager] Purging training data from RAM to prepare for evaluation...")
    del X_train_scaled
    del y_train
    gc.collect()

    # EVALUATION ONLY 
    print("\n--- INITIATING EVALUATION PASS ---")
    test_files = glob.glob(os.path.join(dataset_dir, "test", "*.npz"))
    if debug:
        test_files = test_files[:10]  # Restrict test files in debug mode
    
    total_batches = (len(test_files) // test_batch_size) + 1

    for algo_name in model_configs.keys():
        print(f"\n--> Evaluating {algo_name} on Test Set ({len(test_files)} total tiles)...")
        
        # Load just this single model from disk
        checkpoint_dir = os.path.join("..", "checkpoints", algo_name.lower(), modality)
        model_filepath = os.path.join(checkpoint_dir, f"{algo_name}_tiles_subset_{subset_id}_tax_{taxonomy}_pSize_{patch_size}.joblib")
        model = joblib.load(model_filepath)
        
        global_cm = np.zeros((len(subset_classes), len(subset_classes)), dtype=np.int64)
        
        # Batched inference over the test set
        for batch_idx, i in enumerate(range(0, len(test_files), test_batch_size)):
            if batch_idx % 10 == 0:
                print(f"      [Progress] Processing batch {batch_idx}/{total_batches}...")

            batch_paths = test_files[i:i+test_batch_size]
            X_batch_list, y_batch_list = [], []
            
            for f in batch_paths:
                # Using 'with' forces Python to close the file and release memory instantly
                with np.load(f) as data:
                    X_img = data['X'].transpose(1, 2, 0).astype(np.float32)
                    X_batch_list.append(X_img.reshape(-1, X_img.shape[-1]))
                    y_batch_list.append(data['y'].flatten().astype(np.int32))
                
            X_batch = np.vstack(X_batch_list)
            y_batch = np.concatenate(y_batch_list)
            
            # Use the saved scaler to project the test data into the training space
            X_batch_scaled = scaler.transform(X_batch).astype(np.float32)
            y_pred = model.predict(X_batch_scaled)
            
            # Accumulate metrics
            cm = confusion_matrix(y_batch, y_pred, labels=subset_classes)
            global_cm += cm
            
            # Aggressive cleanup for the current batch
            del X_batch_list, y_batch_list, X_batch, y_batch, X_batch_scaled, y_pred
            gc.collect()

        print("      [Metrics] Calculating performance metrics directly from Confusion Matrix...")
        
        # RAM-SAFE CLASSIFICATION REPORT GENERATION
        report_lines = [
            f"{'':<25} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>15}\n"
        ]
        
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
            
            macro_p += p
            macro_r += r
            macro_f1 += f1
            
            weighted_p += p * support
            weighted_r += r * support
            weighted_f1 += f1 * support
            
        num_classes = len(target_names)
        macro_p /= num_classes
        macro_r /= num_classes
        macro_f1 /= num_classes
        
        weighted_p /= total_support if total_support > 0 else 1
        weighted_r /= total_support if total_support > 0 else 1
        weighted_f1 /= total_support if total_support > 0 else 1
        
        accuracy = np.trace(global_cm) / total_support if total_support > 0 else 0.0
        
        report_lines.append(f"\n{'accuracy':<25} {'':>10} {'':>10} {accuracy:>10.4f} {total_support:>15}")
        report_lines.append(f"{'macro avg':<25} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {total_support:>15}")
        report_lines.append(f"{'weighted avg':<25} {weighted_p:>10.4f} {weighted_r:>10.4f} {weighted_f1:>10.4f} {total_support:>15}")
        
        report = "\n".join(report_lines)
                    
        # Output directory and save logic
        algo_dir = os.path.join(results_out_dir, algo_name)
        os.makedirs(algo_dir, exist_ok=True)
        
        # Generate classification report text file
        report_path = os.path.join(algo_dir, f"performance_subset_{subset_id}.txt")
        with open(report_path, "w") as f:
            f.write(f"--- Batched Inference Complete ---\nAlgorithm: {algo_name}\n\n")
            f.write(f"--- Test Set Classification Report ---\n{report}")
            
        # Generate and save visual confusion matrix
        matrix_path = os.path.join(algo_dir, f"confusion_matrix_subset_{subset_id}.png")
        fig, ax = plt.subplots(figsize=(10, 8))
        ConfusionMatrixDisplay(confusion_matrix=global_cm, display_labels=target_names).plot(ax=ax, cmap='Blues', colorbar=False)
        plt.title(f"Confusion Matrix: {algo_name}\n({modality} | Subset {subset_id} | pSize {patch_size})")
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.tight_layout()
        plt.savefig(matrix_path, dpi=300)
        plt.close(fig)

        # Clean up the model before loading the next one
        del model
        gc.collect()

    print(f"\nPipeline completed successfully for {modality.upper()} Subset {subset_id}!")
