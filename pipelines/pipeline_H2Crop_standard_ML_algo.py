import os
import sys

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)
    
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from H2Crop.data_structures import h2crop_taxonomy_dict
from hyperparams_tuning.optimize_hyperparameters import optimize_hyperparameters
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
