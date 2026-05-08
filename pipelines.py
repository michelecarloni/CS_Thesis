import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from utils import load_hyperspectral_dataset, normalize_features
from H2Crop import H2Crop

# cuML models
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.linear_model import MBSGDClassifier as cuMBSGD
from cuml.linear_model import LogisticRegression as cuLogReg
from cuml.svm import LinearSVC as cuSVC
from cuml.multiclass import OneVsRestClassifier

import cupy as cp
import math

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







def pipeline_H2Crop_standard_ML_algo(save_results_dir, from_train, limit=1, path=None, detail_layer=None, static=False, keep_prior=False, total_samples=100000, classes_to_drop=None):
    """
    End-to-end pipeline to train and evaluate 4 baseline ML algorithms on H2Crop data.
    Automatically runs 8 experiments (4 algorithms x 2 modalities) on the same sampled files.
    BOOST: Accelerated with NVIDIA GPU using RAPIDS cuML.
    """
    
    # Directories setup
    os.makedirs(save_results_dir, exist_ok=True)
    config_path = os.path.join(save_results_dir, "configuration.txt")
    
    with open(config_path, "w") as f:
        f.write("--- H2Crop ML Pipeline Configuration ---\n")
        f.write(f"from_train: {from_train}\n")
        f.write(f"limit: {limit}\n")
        f.write(f"path: {path}\n")
        f.write(f"detail_layer: {detail_layer}\n")
        f.write(f"static: {static}\n")
        f.write(f"keep_prior: {keep_prior}\n")
        f.write(f"total_samples (balancing target): {total_samples}\n")
        
    print(f"Configuration saved to {config_path}")

    # Loader Initialization
    loader = H2Crop() 
    
    print("\nGenerating master file list...")
    master_file_list = loader.get_file_list(from_train=from_train, path=path, limit=limit)
    
    if not master_file_list:
        print("Pipeline aborted: No files found.")
        return

    # Loop Through Both Modalities (8 Experiments Total)
    for modality in ["hyperspectral", "multispectral"]:
        print(f"\n{'='*50}")
        print(f"STARTING PIPELINE FOR: {modality.upper()}")
        print(f"{'='*50}")
        
        ml_models = {
            # "decision_tree": cuRF(n_estimators=1, random_state=42),
            # "random_forest": cuRF(n_estimators=100, random_state=42),
            # "logistic_regression": OneVsRestClassifier(cuMBSGD(loss='log', batch_size=2048)),
            # "linear_svm": OneVsRestClassifier(cuMBSGD(loss='hinge', batch_size=2048))

            "decision_tree": DecisionTreeClassifier(random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
            "linear_svm": LinearSVC(max_iter=2000, dual=False, random_state=42)
        }
        
        print("Loading data...")

        # Load the data
        batch = loader.load_h5_data(
            file_list=master_file_list, 
            detail_layer=detail_layer, 
            static=static, 
            data_type=modality, 
            keep_prior=keep_prior
        )

        print("Data loaded succesfully!")
        
        if not batch:
            print(f"Skipping {modality} due to loading error.")
            continue

        # Preprocess and Flatten Pixels 
        print("Flattening spatial grids into tabular (X, y) format...")
        X_list = []
        y_list = []
        
        for sample in batch:
            X_img = sample[modality]
            y_img = sample['labels']
            
            # Upsample hyperspectral from 64x64 to 192x192
            if modality == "hyperspectral":
                X_img = loader.upsample_hyperspectral(X_img)
                
            # Transpose necessary for stacking together
            X_img = np.transpose(X_img, (1, 2, 0))
            
            # Flatten to tabular format (Pixels, Channels)
            X_flat = X_img.reshape(-1, X_img.shape[-1])
            y_flat = y_img.reshape(-1)
            
            # Handle Priors if requested
            if keep_prior and 'prior' in sample:
                prior_img = sample['prior']
                prior_flat = prior_img.reshape(-1, 1)
                X_flat = np.hstack((X_flat, prior_flat))
                
            X_list.append(X_flat)
            y_list.append(y_flat)
            
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        # Drop unecessary classes
        X, y = loader.drop_classes(X, y, classes_to_drop=classes_to_drop)

        # extract a balance dataset
        X, y = loader.balance_pixels(X, y, total_samples=total_samples)

        print("Splitting into Train/Test sets and scaling features...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        scaler = StandardScaler()
        
        # Cast Features to float32: critical for GPU memory efficiency
        X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
        X_test_scaled = scaler.transform(X_test).astype(np.float32)
        
        # Cast Labels to int32: critical for classification algorithms
        y_train = y_train.astype(np.int32)
        y_test = y_test.astype(np.int32)

        # Train loop
        for algo_name, model in ml_models.items():
            print(f"\n--> Training {algo_name} on NVIDIA GPU...")
            
            # Train the model
            model.fit(X_train_scaled, y_train)
            
            # Force cupy to release all cached VRAM back to the GPU
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
            batch_size = 2048  # Kept small to guarantee it fits in VRAM
            y_pred_list = []
            
            n_samples = X_test_scaled.shape[0]
            n_batches = math.ceil(n_samples / batch_size)
            
            print(f"    Predicting in {n_batches} batches to save memory...")
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                # Predict a small chunk
                batch_pred = model.predict(X_test_scaled[start_idx:end_idx])
                
                # Immediately pull the prediction back to standard CPU memory (NumPy)
                if hasattr(batch_pred, 'get'):
                    batch_pred = batch_pred.get()
                else:
                    batch_pred = np.array(batch_pred)
                    
                y_pred_list.append(batch_pred)
                
            # Combine all the chunks back together and ensure they are integers for sklearn
            y_pred = np.concatenate(y_pred_list).astype(np.int32)
            
            # Clean up GPU memory again before the next algorithm starts
            cp.get_default_memory_pool().free_all_blocks()
            
            # Save Classification Report & Confusion Matrix
            algo_dir = os.path.join(save_results_dir, modality, algo_name)
            os.makedirs(algo_dir, exist_ok=True)
            
            report_path = os.path.join(algo_dir, "performance.txt")
            report = classification_report(y_test, y_pred, zero_division=0)
            with open(report_path, "w") as f:
                f.write(report)
            print(f"    Saved: {report_path}")
            
            matrix_path = os.path.join(algo_dir, "confusion_matrix.png")
            fig, ax = plt.subplots(figsize=(10, 8))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues', colorbar=False)
            plt.title(f"Confusion Matrix: {algo_name} ({modality})")
            plt.tight_layout()
            plt.savefig(matrix_path, dpi=300)
            plt.close(fig)
            print(f"    Saved: {matrix_path}")
            
    print("\nPipeline completed successfully! All 8 GPU-accelerated experiments saved.")