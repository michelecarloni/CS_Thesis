import os
import json
import gc
import copy
import glob
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from utils import load_hyperspectral_dataset, normalize_features
from H2Crop.data_structures import h2crop_taxonomy_dict
from H2Crop.H2CropTileDataset import H2CropTileDataset
from hyperparameter_tuning import optimize_hyperparameters, optimize_cnn_hyperparameters, optimize_unet_hyperparameters
from utils import load_and_flatten_segmentation_tiles
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import cupy as cp

# NVIDIA RAPIDS cuML (GPU Models)
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLR
    from cuml.svm import LinearSVC as cuSVC
    from cuml.linear_model import MBSGDClassifier as cuMBSGD
except ImportError:
    pass

"""
Pipeline called for the first experiment (First Baseline):
    - Datasets: indian_pines, salinas_valley, pavia_center, pavia_university
    - Algorithms: Decision Tree, Random Forest, Linear SVM, Linear Regression
"""
def pipeline_standard_ml_algo(dataset_config_dict, save_dir, use_undersampling=False):
    """
    Trains and evaluates 4 baseline ML models on multiple datasets.
    
    Args:
        dataset_config_dict (dict): Configuration mapping for datasets, dropped classes, and limits.
        save_dir (str): Mandatory root directory path to save all outputs.
        use_undersampling (bool): If True, randomly drops majority class samples.
                                  If False, uses all data but applies class_weight='balanced'.
    """
    # Dynamically set class_weight based on the chosen training strategy
    cw = None if use_undersampling else 'balanced'
    
    models = {
        'decision_tree': DecisionTreeClassifier(class_weight=cw, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=42, n_jobs=-1),
        'logistic_regression': LogisticRegression(max_iter=1000, class_weight=cw, random_state=42, n_jobs=-1),
        'linear_svm': LinearSVC(max_iter=1000, class_weight=cw, random_state=42, dual=False)
    }

    oa_results = {model_name: {} for model_name in models.keys()}
    
    # Ensure root save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for dataset_name, config in dataset_config_dict.items():
        classes_to_drop = config['drop']
        target_train_samples = config['train_samples']
        
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"Strategy: {'Undersampling' if use_undersampling else 'Cost-Sensitive Learning (All Data)'}")
        print(f"{'='*60}")
        
        # Load & Normalize
        print(f"Loading data (dropping classes: {classes_to_drop})...")
        X, y = load_hyperspectral_dataset(dataset_name, classes_to_drop=classes_to_drop)
        print("Normalizing features...")
        X_norm, _ = normalize_features(X)
        
        # Train/Test Split (80/20 Stratified)
        print("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y, test_size=0.2, random_state=42, stratify=y
        )

        # Apply Training Strategy
        if use_undersampling:
            print(f"Undersampling training data to max {target_train_samples} samples per class...")
            X_train_final, y_train_final = [], []
            for cls in np.unique(y_train):
                idx = np.where(y_train == cls)[0]
                n_to_sample = min(target_train_samples, len(idx))
                selected_idx = np.random.choice(idx, n_to_sample, replace=False)
                
                X_train_final.append(X_train[selected_idx])
                y_train_final.append(y_train[selected_idx])
                
            X_train_final = np.vstack(X_train_final)
            y_train_final = np.concatenate(y_train_final)
        else:
            print("Using 100% of training data with balanced class weights...")
            X_train_final, y_train_final = X_train, y_train
            
        print(f"  -> Final Train size: {len(X_train_final)}")
        print(f"  -> Test size (untouched): {len(X_test)}")
        
        # Train and Evaluate each model
        for model_name, model in models.items():
            print(f"\n  [Training {model_name}...]")
            
            # Fit & Predict
            model.fit(X_train_final, y_train_final)
            y_pred = model.predict(X_test)
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            oa_results[model_name][dataset_name] = acc  # Store for Markdown table
            report = classification_report(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            # Directory Setup
            dir_path = os.path.join(save_dir, model_name, dataset_name)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save Report
            report_path = os.path.join(dir_path, "performance.txt")
            with open(report_path, "w") as f:
                f.write(f"--- Classification Report ---\n")
                f.write(f"Model:    {model_name}\n")
                f.write(f"Dataset:  {dataset_name}\n")
                f.write(f"Strategy: {'Undersampling' if use_undersampling else 'Cost-Sensitive (Balanced Weights)'}\n")
                f.write(f"Overall Accuracy: {acc:.4f}\n")
                f.write("-" * 40 + "\n\n")
                f.write(report)
                
            # Save Confusion Matrix PNG
            cm_path = os.path.join(dir_path, "confusion_matrix.png")
            unique_classes = np.unique(y_test)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=unique_classes, yticklabels=unique_classes)
            plt.title(f"Confusion Matrix: {model_name.replace('_', ' ').title()}\nDataset: {dataset_name.replace('_', ' ').title()}", fontsize=14, pad=15)
            plt.ylabel("True Class Label", fontsize=12)
            plt.xlabel("Predicted Class Label", fontsize=12)
            plt.tight_layout()
            plt.savefig(cm_path, dpi=300)
            plt.close()
                
            print(f"     Saved -> {dir_path}/ (OA: {acc:.4f})")

    print(f"\n{'='*60}\nGenerating Overall Accuracy Markdown Table...\n{'='*60}")
    md_path = os.path.join(save_dir, "overall_accuracies.md")
    
    datasets = list(dataset_config_dict.keys())
    
    # Build Table Header
    md_content = f"# Overall Accuracies\n\n"
    md_content += f"**Training Strategy:** {'Undersampling' if use_undersampling else 'Cost-Sensitive Learning (Balanced Weights)'}\n\n"
    md_content += "| Algorithm | " + " | ".join([ds.replace('_', ' ').title() for ds in datasets]) + " |\n"
    md_content += "|---| " + " | ".join(["---"] * len(datasets)) + " |\n"
    
    # Build Table Rows
    for model_name in models.keys():
        row = f"| **{model_name.replace('_', ' ').title()}** | "
        for ds in datasets:
            acc = oa_results[model_name].get(ds, 0.0)
            row += f"{acc:.4f} | "
        md_content += row + "\n"
        
    with open(md_path, "w") as f:
        f.write(md_content)
        
    print(f"Saved accuracy summary table to: {md_path}")







#def pipeline_H2Crop_standard_ML_algo(save_results_dir, file_list, modality, loader=None, detail_layer=0, static=False, keep_prior=False, total_samples=100000, classes_to_drop=None):
#    """
#    Modular pipeline to train and evaluate 4 baseline ML algorithms on H2Crop data.
#    Takes a pre-defined list of files and a specific modality to process.
#    """
#
#    # Check Loader
#    if not loader:
#        print("Pipeline aborted: requiring loader")
#        return
#
#    # Check Modality
#    if modality.lower() not in ["hyperspectral", "multispectral"]:
#        print('Pipeline aborted: modality is neither "Hyperspectral" nor "Multispectral" ')
#        return
#
#    print(f"\n{'='*50}")
#    print(f"STARTING PIPELINE FOR: {modality.upper()}")
#    print(f"Processing {len(file_list)} files...")
#    print(f"{'='*50}")
#
#    # Directories setup
#    os.makedirs(os.path.join(save_results_dir, modality), exist_ok=True)
#    config_path = os.path.join(save_results_dir, modality, "configuration.txt")
#    
#    with open(config_path, "w") as f:
#        f.write("--- H2Crop ML Pipeline Configuration ---\n")
#        f.write(f"modality: {modality}\n")
#        f.write(f"num_files_processed: {len(file_list)}\n")
#        f.write(f"detail_layer: {detail_layer}\n")
#        f.write(f"static: {static}\n")
#        f.write(f"keep_prior: {keep_prior}\n")
#        f.write(f"total_samples (balancing target): {total_samples}\n")
#        f.write(f"classes_to_drop: {classes_to_drop}\n")
#        
#    print(f"Configuration saved to {config_path}")
#    
#    # Scikit-learn CPU Models
#    ml_models = {
#        "decision_tree": DecisionTreeClassifier(random_state=42),
#        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
#        # "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
#        "logistic_regression": LogisticRegression(max_iter=4000, solver='saga', C=0.1, random_state=42),
#        "linear_svm": LinearSVC(max_iter=2000, dual=False, random_state=42)
#    }
#    
#    print("Loading data...")
#
#    # Load the data using the explicitly provided file list and modality
#    batch = loader.load_h5_data(
#        file_list=file_list, 
#        detail_layer=detail_layer, 
#        static=static, 
#        data_type=modality, 
#        keep_prior=keep_prior
#    )
#
#    if not batch:
#        print(f"Pipeline aborted: No data loaded for {modality}.")
#        return
#
#    print("Data loaded successfully!")
#
#    # Preprocess and Flatten Pixels 
#    print("Flattening spatial grids into tabular (X, y) format...")
#    X_list = []
#    y_list = []
#    
#    for sample in batch:
#        X_img = sample[modality]
#        y_img = sample['labels']
#        
#        # Upsample hyperspectral from 64x64 to 192x192
#        if modality == "hyperspectral":
#            X_img = loader.upsample_hyperspectral(X_img)
#            
#        # Transpose necessary for stacking together
#        X_img = np.transpose(X_img, (1, 2, 0))
#        
#        # Flatten to tabular format (Pixels, Channels)
#        X_flat = X_img.reshape(-1, X_img.shape[-1])
#        y_flat = y_img.reshape(-1)
#        
#        # Handle Priors if requested
#        if keep_prior and 'prior' in sample:
#            prior_img = sample['prior']
#            prior_flat = prior_img.reshape(-1, 1)
#            X_flat = np.hstack((X_flat, prior_flat))
#            
#        # Filter before you stack
#        if classes_to_drop is not None:
#            valid_mask = ~np.isin(y_flat, classes_to_drop)
#            X_flat = X_flat[valid_mask]
#            y_flat = y_flat[valid_mask]
#
#        X_list.append(X_flat)
#        y_list.append(y_flat)
#        
#    X = np.vstack(X_list)
#    y = np.concatenate(y_list)
#    
#    # Drop unecessary classes
#    X, y = loader.drop_classes(X, y, classes_to_drop=classes_to_drop)
#
#    print("DEBUG")
#
#    # extract a balance dataset
#    X, y = loader.balance_pixels(X, y, total_samples=total_samples)
#
#    print("Splitting into Train/Test sets and scaling features...")
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#    
#    scaler = StandardScaler()
#    
#    # Scale Features (Keeping float32 to save RAM)
#    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
#    X_test_scaled = scaler.transform(X_test).astype(np.float32)
#    
#    # Cast Labels to int32: critical for classification algorithms
#    y_train = y_train.astype(np.int32)
#    y_test = y_test.astype(np.int32)
#
#    # Train loop (Cleaned up for CPU Scikit-Learn)
#    for algo_name, model in ml_models.items():
#        print(f"\n--> Training {algo_name}...")
#        
#        # Train the model
#        model.fit(X_train_scaled, y_train)
#        
#        # Predict all at once (System RAM handles this efficiently)
#        y_pred = model.predict(X_test_scaled)
#        
#        # -------------------------------------------------------------
#        # Save Classification Report & Confusion Matrix
#        # -------------------------------------------------------------
#        algo_dir = os.path.join(save_results_dir, modality, algo_name)
#        os.makedirs(algo_dir, exist_ok=True)
#        
#        # 1. Dynamically grab the right names for the current classes
#        taxonomy_key = f'Taxonomy_{detail_layer}'
#        current_taxonomy = h2crop_taxonomy_dict.get(taxonomy_key, {})
#        
#        # Map the IDs to strings. Fallback to "Class X" if something is missing.
#        target_names = [current_taxonomy.get(c, f"Class {c}") for c in model.classes_]
#        
#        # 2. Save Classification Report (.txt)
#        report_path = os.path.join(algo_dir, "performance.txt")
#        report = classification_report(y_test, y_pred, zero_division=0, target_names=target_names)
#        with open(report_path, "w") as f:
#            f.write(report)
#        print(f"    Saved: {report_path}")
#        
#        # 3. Save Confusion Matrix (.png)
#        matrix_path = os.path.join(algo_dir, "confusion_matrix.png")
#        
#        fig_size = max(10, len(target_names) * 0.4)
#        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
#        
#        ConfusionMatrixDisplay.from_predictions(
#            y_test, 
#            y_pred, 
#            ax=ax, 
#            cmap='Blues', 
#            colorbar=False,
#            display_labels=target_names
#        )
#        
#        plt.title(f"Confusion Matrix: {algo_name} ({modality})")
#        plt.xticks(rotation=45, ha='right', fontsize=9)
#        plt.yticks(fontsize=9)
#        
#        plt.tight_layout()
#        plt.savefig(matrix_path, dpi=300)
#        plt.close(fig)
#        print(f"    Saved: {matrix_path}")
#        
#    print(f"\nPipeline completed successfully! {modality.upper()} experiments saved.")










def pipeline_H2Crop_standard_ML_algo(save_results_dir, data_path, modality, detail_layer=0, use_gpu=True):
    """
    Modular pipeline to train and evaluate baseline ML algorithms on H2Crop data.
    Automatically extracts feature importances and coefficients for thesis analysis.
    GPU is enabled by default.
    """
    
    if not os.path.exists(data_path):
        print(f"Pipeline aborted: Extracted data not found at {data_path}")
        return

    print(f"\n{'='*60}")
    print(f"STARTING ML PIPELINE FOR: {modality.upper()} (GPU: {use_gpu})")
    print(f"Loading data from: {data_path}")
    print(f"{'='*60}")

    os.makedirs(os.path.join(save_results_dir, modality), exist_ok=True)
    
    # -------------------------------------------------------------
    # 1. Load Pre-Extracted Data
    # -------------------------------------------------------------
    print("Loading pre-extracted arrays into memory...")
    with np.load(data_path) as data:
        X = data['X']
        y = data['y']
        
    print(f"Data loaded successfully! Total samples: {X.shape[0]}, Features: {X.shape[1]}")

    # -------------------------------------------------------------
    # 2. 70/20/10 Train/Val/Test Split
    # -------------------------------------------------------------
    print("\nSplitting into Train (70%), Validation (20%), and Test (10%) sets...")
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.20 / 0.90), random_state=42, stratify=y_temp)
    
    del X, y, X_temp, y_temp
    gc.collect()
    
    # -------------------------------------------------------------
    # 3. Scaling
    # -------------------------------------------------------------
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    y_train = y_train.astype(np.int32)
    y_val = y_val.astype(np.int32)
    y_test = y_test.astype(np.int32)
    
    del X_train, X_val, X_test
    gc.collect()

    config_path = os.path.join(save_results_dir, modality, "configuration.txt")
    with open(config_path, "w") as f:
        f.write("--- H2Crop ML Pipeline Configuration ---\n")
        f.write(f"modality: {modality}\n")
        f.write(f"detail_layer: {detail_layer}\n")
        f.write(f"data_path: {data_path}\n")
        f.write(f"total_samples: {len(y_train) + len(y_val) + len(y_test)}\n")
        f.write(f"use_gpu: {use_gpu}\n")
        f.write(f"tuning: Optuna TPESampler\n")
        
    # -------------------------------------------------------------
    # 4. Model Tuning and Evaluation
    # -------------------------------------------------------------
    models_to_run = {
        "decision_tree": 15,
        "random_forest": 30,
        "logistic_regression": 20,
        "linear_svm": 20
    }

    for algo_name, n_trials in models_to_run.items():
        print(f"\n--> Tuning and Training {algo_name} with Optuna ({n_trials} trials)...")
        
        # *** GPU VRAM CLEANUP ***
        gc.collect()
        try:
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks() 
            pinned_mempool.free_all_blocks() 
        except Exception:
            pass
        
        best_model, best_params = optimize_hyperparameters(
            model_name=algo_name,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            n_trials=n_trials,
            random_state=42,
            use_gpu=use_gpu
        )
        
        print(f"    Evaluating Best Model on Test Set...")
        y_pred = best_model.predict(X_test_scaled)
        
        algo_dir = os.path.join(save_results_dir, modality, algo_name)
        os.makedirs(algo_dir, exist_ok=True)
        
        taxonomy_key = f'Taxonomy_{detail_layer}'
        current_taxonomy = h2crop_taxonomy_dict.get(taxonomy_key, {})
        target_names = [current_taxonomy.get(c, f"Class {c}") for c in np.unique(y_test)]
        
        # Save Performance Report
        report_path = os.path.join(algo_dir, "performance.txt")
        report = classification_report(y_test, y_pred, zero_division=0, target_names=target_names)
        
        with open(report_path, "w") as f:
            f.write(f"--- Best Optuna Hyperparameters ---\n")
            f.write(json.dumps(best_params, indent=4))
            f.write(f"\n\n--- Test Set Classification Report ---\n")
            f.write(report)
            
        # Save Confusion Matrix
        matrix_path = os.path.join(algo_dir, "confusion_matrix.png")
        fig_size = max(10, len(target_names) * 0.4)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
        
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax, cmap='Blues', colorbar=False, display_labels=target_names
        )
        
        plt.title(f"Confusion Matrix: {algo_name}\n({modality} | Tuned via Optuna)")
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(matrix_path, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # 5. Extract & Save Feature Significance (Gini & Coefficients)
        # -------------------------------------------------------------
        print(f"    Extracting Feature Significance metrics...")
        try:
            if algo_name in ["decision_tree", "random_forest"]:
                # Extract Gini importance from tree models
                importances = best_model.feature_importances_
                
                # Safely convert GPU array to CPU numpy array if necessary
                if hasattr(importances, 'get'):
                    importances = importances.get()
                elif hasattr(importances, 'to_numpy'):
                    importances = importances.to_numpy()
                else:
                    importances = np.array(importances)
                    
                df_imp = pd.DataFrame({
                    "Band_Index": np.arange(len(importances)),
                    "Gini_Importance": importances
                })
                df_imp.to_csv(os.path.join(algo_dir, "feature_importances.csv"), index=False)
                
            elif algo_name in ["logistic_regression", "linear_svm"]:
                # The model is wrapped in OneVsRestClassifier
                # Extract the coefficient matrix for each class
                coefs = []
                for estimator in best_model.estimators_:
                    c = estimator.coef_
                    
                    # Safely convert GPU array to CPU numpy array if necessary
                    if hasattr(c, 'get'):
                        c = c.get()
                    elif hasattr(c, 'to_numpy'):
                        c = c.to_numpy()
                    else:
                        c = np.array(c)
                        
                    coefs.append(c.flatten())
                
                coefs_matrix = np.array(coefs)
                
                # Create a CSV where rows are Crop Classes and columns are Hyperspectral Bands
                band_columns = [f"Band_{i}" for i in range(coefs_matrix.shape[1])]
                df_coef = pd.DataFrame(coefs_matrix, columns=band_columns)
                df_coef.insert(0, "Crop_Class", target_names) 
                
                df_coef.to_csv(os.path.join(algo_dir, "model_coefficients.csv"), index=False)

        except Exception as e:
            print(f"    [Warning] Could not extract feature metrics for {algo_name}: {str(e)}")
        
    print(f"\nPipeline completed successfully for {modality.upper()}!")






def pipeline_H2Crop_CNN(model, tiles_dir, save_results_dir, modality, batch_size=32, n_trials=10, epochs_per_trial=5, final_epochs=20, use_gpu=True):
    """
    Trains a pre-instantiated CNN using pre-split train/val/test folders.
    Delegates tuning to an external script using a 20% data subset for speed, 
    then evaluates and saves reports/checkpoints using standard FP32 math and tqdm.
    """
    import random
    from tqdm import tqdm
    
    model_name = getattr(model, 'name', 'Unknown_CNN_Model')
    
    print(f"\n{'='*70}")
    print(f"STARTING CNN PIPELINE FOR: {modality.upper()} | Model: {model_name}")
    print(f"{'='*70}")

    algo_dir = os.path.join(save_results_dir, modality, model_name)
    os.makedirs(algo_dir, exist_ok=True)

    # 1. Load file paths directly from the pre-split subdirectories
    train_dir = os.path.join(tiles_dir, "train")
    val_dir = os.path.join(tiles_dir, "validation")
    test_dir = os.path.join(tiles_dir, "test")

    train_files = glob.glob(os.path.join(train_dir, "*.npz"))
    val_files = glob.glob(os.path.join(val_dir, "*.npz"))
    test_files = glob.glob(os.path.join(test_dir, "*.npz"))

    if not train_files or not val_files or not test_files:
        raise ValueError(f"Missing one or more split folders (train/validation/test) in {tiles_dir}. Run extraction first.")
        
    print(f"Dataset Split Loaded: Train ({len(train_files)}), Val ({len(val_files)}), Test ({len(test_files)})")
    
    # 2. Setup DataLoaders (Optimized with pin_memory=True for faster I/O)
    train_loader = DataLoader(H2CropTileDataset(train_files), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(H2CropTileDataset(val_files), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(H2CropTileDataset(test_files), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Create a smaller 20% subset strictly for Optuna to speed up tuning
    tune_sample_size = max(1, int(len(train_files) * 0.20))
    tune_train_files = random.sample(train_files, tune_sample_size)
    tune_train_loader = DataLoader(H2CropTileDataset(tune_train_files), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"\n[Optimization] Using {tune_sample_size} tiles (20%) for Hyperparameter Tuning.")

    # 4. Save the initial "blank slate" weights of the model
    initial_model_state = copy.deepcopy(model.state_dict())

    # 5. Call external Optuna tuner with the tuned subset
    best_params = optimize_cnn_hyperparameters(
        model=model,
        train_loader=tune_train_loader,
        val_loader=val_loader,
        initial_model_state=initial_model_state,
        n_trials=n_trials,
        epochs_per_trial=epochs_per_trial,
        use_gpu=use_gpu
    )

    # =================================================================
    # 6. Final Retraining on 100% of Data with Best Parameters
    # =================================================================
    print(f"\n--- Retraining on 100% of Data for {final_epochs} Epochs ---")
    model.load_state_dict(copy.deepcopy(initial_model_state))
    
    if best_params["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=best_params["lr"], momentum=0.9, weight_decay=best_params["weight_decay"])
        
    criterion = nn.CrossEntropyLoss()
    
    checkpoint_dir = os.path.join("..", "checkpoints", model_name.lower(), modality)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(final_epochs):
        model.train()
        running_train_loss = 0.0
        
        # TQDM Train Loop
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{final_epochs}] [Train]", leave=False)
        
        for batch_X, batch_y in train_loop:
            if use_gpu and torch.cuda.is_available():
                batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                
            # Input trap to catch bad data
            if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                continue
                
            optimizer.zero_grad()
            
            # AMP DISABLED: Standard 32-bit Forward Pass (Improved with TF32)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Standard 32-bit Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_train_loss += loss.item() * batch_X.size(0)
            train_loop.set_postfix(loss=loss.item())
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        running_val_loss = 0.0
        
        # TQDM Validation Loop
        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{final_epochs}] [Val]", leave=False)
        
        with torch.no_grad():
            for batch_X, batch_y in val_loop:
                if use_gpu and torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                    
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item() * batch_X.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # This prints a clean summary after the tqdm bars disappear
        print(f"    Epoch [{epoch+1}/{final_epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

    plot_path = os.path.join(checkpoint_dir, "loss_curve.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, final_epochs + 1), train_losses, label='Training Loss', marker='o')
    ax.plot(range(1, final_epochs + 1), val_losses, label='Validation Loss', marker='o')
    ax.set_title(f"Loss Curve: {model_name} ({modality.capitalize()})")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved {final_epochs} checkpoints and loss curve to {checkpoint_dir}")
            
    # =================================================================
    # 7. Final Evaluation on Test Set
    # =================================================================
    print("\n--- Evaluating on Test Set ---")
    model.eval()
    all_preds, all_targets = [], []
    
    # TQDM Test Loop
    test_loop = tqdm(test_loader, desc="Testing", leave=False)
    
    with torch.no_grad():
        for batch_X, batch_y in test_loop:
            if use_gpu and torch.cuda.is_available():
                batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    # =================================================================
    # 8. Save Report and Confusion Matrix
    # =================================================================
    print(f"\n--- Saving Results to {algo_dir} ---")
    
    unique_classes = np.unique(np.concatenate((all_targets, all_preds)))
    target_names = [f"Class {c}" for c in unique_classes]
    
    report_path = os.path.join(algo_dir, "performance.txt")
    report = classification_report(
        all_targets, 
        all_preds, 
        labels=unique_classes, 
        target_names=target_names, 
        zero_division=0
    )
    
    with open(report_path, "w") as f:
        f.write(f"--- Best Optuna Hyperparameters ---\n")
        f.write(json.dumps(best_params, indent=4))
        f.write(f"\n\n--- Test Set Classification Report ---\n")
        f.write(report)
        
    matrix_path = os.path.join(algo_dir, "confusion_matrix.png")
    fig_size = max(10, len(unique_classes) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    
    ConfusionMatrixDisplay.from_predictions(
        all_targets, 
        all_preds, 
        labels=unique_classes, 
        ax=ax, 
        cmap='Blues', 
        colorbar=False, 
        display_labels=target_names
    )
    
    plt.title(f"Confusion Matrix: {model_name}\n({modality} | Tuned via Optuna)")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=300)
    plt.close(fig)
    
    print("Training Complete! All reports saved.")
    
    return model







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



def pipeline_H2Crop_unet_optuna(
    model, 
    model_name,
    save_results_dir, 
    dataset_dir, 
    subset_id, 
    subset_classes,
    modality, 
    taxonomy=3, 
    patch_size=32, 
    use_gpu=True, 
    n_trials=10,
    epochs_per_trial=5,
    final_epochs=20,
    batch_size=32, 
    debug=False
):
    """
    Optuna-powered Deep Learning segmentation pipeline.
    Processes a single PyTorch model at a time, driven by the main script.
    Deterministically locks output channels using the provided subset_classes.
    """
    print(f"\n{'='*70}")
    mode = "DEBUG MODE" if debug else "PRODUCTION MODE (DEEP LEARNING)"
    print(f"STARTING PIPELINE FOR: {model_name.upper()} | {modality.upper()} | Subset {subset_id} | {mode}")
    print(f"{'='*70}")

    # Results will now be saved directly into the dynamically generated folder from main
    results_out_dir = os.path.join(save_results_dir, modality)
    os.makedirs(results_out_dir, exist_ok=True)
    
    # Checkpoints organized by model name
    checkpoint_dir = os.path.join("..", "checkpoints", model_name, modality)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # LAZY DATALOADER INITIALIZATION
    print("\n--- Initializing PyTorch DataLoaders ---")
    
    # Pass the subset_classes into the Dataset so it builds the exact same map every time
    train_dataset = H2CropTileDataset(os.path.join(dataset_dir, "train"), subset_classes=subset_classes, debug=debug)
    val_dataset = H2CropTileDataset(os.path.join(dataset_dir, "validation"), subset_classes=subset_classes, debug=debug)
    test_dataset = H2CropTileDataset(os.path.join(dataset_dir, "test"), subset_classes=subset_classes, debug=debug)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=use_gpu)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=use_gpu)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=use_gpu)

    # We now know exactly what num_classes is deterministically!
    num_classes = len(subset_classes) + 1
    sample_y = [0] + sorted(subset_classes)
    
    taxonomy_key = f'Taxonomy_{taxonomy}'
    target_names = [h2crop_taxonomy_dict.get(taxonomy_key, {}).get(c, f"Class {c}") if c != 0 else "Background (0)" for c in sample_y]

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # MODEL SETUP
    model = model.to(device)
    initial_model_state = copy.deepcopy(model.state_dict())

    # OPTUNA HYPERPARAMETER TUNING
    active_trials = 2 if debug else n_trials
    active_epochs = 1 if debug else epochs_per_trial
    
    best_params = optimize_unet_hyperparameters(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        initial_model_state=initial_model_state,
        num_classes=num_classes,
        n_trials=active_trials,
        epochs_per_trial=active_epochs,
        use_gpu=use_gpu
    )
    
    with open(os.path.join(results_out_dir, f"best_params_subset_{subset_id}.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    # FINAL PRODUCTION TRAINING
    print(f"\n--- INITIATING FINAL TRAINING: {model_name} ---")
    
    model.load_state_dict(initial_model_state)
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    train_epochs = 2 if debug else final_epochs
    
    for epoch in range(train_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"    Epoch {epoch+1}/{train_epochs} | Loss: {running_loss/len(train_loader):.4f}")
        
    model_filepath = os.path.join(checkpoint_dir, f"{model_name}_subset_{subset_id}_optuna.pth")
    torch.save(model.state_dict(), model_filepath)
    print(f"    Saved checkpoint to: {model_filepath}")

    # RAM-SAFE TEST EVALUATION
    print(f"\n--- EVALUATING ON TEST SET ---")
    model.eval()
    global_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.long().to(device)
            
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            
            pred_flat = predicted.view(-1)
            true_flat = batch_y.view(-1)
            
            indices = num_classes * true_flat + pred_flat
            batch_cm = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
            global_cm += batch_cm
            
    print("      [Metrics] Calculating performance metrics directly from GPU Confusion Matrix...")
    cm_numpy = global_cm.cpu().numpy()
    
    report_lines = [f"{'':<25} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>15}\n"]
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0
    total_support = np.sum(cm_numpy)
    
    for idx, target_name in enumerate(target_names):
        tp = cm_numpy[idx, idx]
        fp = cm_numpy[:, idx].sum() - tp
        fn = cm_numpy[idx, :].sum() - tp
        support = cm_numpy[idx, :].sum()
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        report_lines.append(f"{target_name:<25} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {support:>15}")
        macro_p += p; macro_r += r; macro_f1 += f1
        weighted_p += p * support; weighted_r += r * support; weighted_f1 += f1 * support
        
    macro_p /= num_classes; macro_r /= num_classes; macro_f1 /= num_classes
    weighted_p /= total_support if total_support > 0 else 1
    weighted_r /= total_support if total_support > 0 else 1
    weighted_f1 /= total_support if total_support > 0 else 1
    accuracy = np.trace(cm_numpy) / total_support if total_support > 0 else 0.0
    
    report_lines.append(f"\n{'accuracy':<25} {'':>10} {'':>10} {accuracy:>10.4f} {total_support:>15}")
    report_lines.append(f"{'macro avg':<25} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {total_support:>15}")
    report_lines.append(f"{'weighted avg':<25} {weighted_p:>10.4f} {weighted_r:>10.4f} {weighted_f1:>10.4f} {total_support:>15}")
    
    with open(os.path.join(results_out_dir, f"performance_subset_{subset_id}_optuna.txt"), "w") as f:
        f.write(f"--- Deep Learning Optimized Inference ({model_name}) ---\n\n" + "\n".join(report_lines))
        
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm_numpy, display_labels=target_names).plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f"Confusion Matrix: {model_name} (Optuna)\n({modality} | Subset {subset_id} | pSize {patch_size})")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_out_dir, f"confusion_matrix_subset_{subset_id}_optuna.png"), dpi=300)
    plt.close(fig)

    # AGGRESSIVE GPU MEMORY CLEANUP
    del model, initial_model_state, global_cm, outputs
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\nPipeline completed successfully for {model_name} on {modality.upper()} Subset {subset_id}!")