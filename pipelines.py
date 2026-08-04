import os
import json
import gc
import copy
import glob
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
from hyperparameter_tuning import optimize_hyperparameters, optimize_cnn_hyperparameters
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import cupy as cp

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
    then evaluates and saves reports/checkpoints using Mixed Precision (AMP).
    """
    import random
    
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
        
        for batch_X, batch_y in train_loader:
            if use_gpu and torch.cuda.is_available():
                batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                
            optimizer.zero_grad()
            
            # AMP DISABLED: Standard 32-bit Forward Pass (Improved with TF32)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Standard 32-bit Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_train_loss += loss.item() * batch_X.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                if use_gpu and torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                    
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item() * batch_X.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
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
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
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