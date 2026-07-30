import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.subplots as plt_subplots
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
from hyperparameter_tuning import optimize_hyperparameters
import json

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











def pipeline_H2Crop_standard_ML_algo(save_results_dir, file_list, modality, class_action="drop", class_list=None, loader=None, detail_layer=0, static=False, keep_prior=False, samples_per_class=25000, allow_resample=True):
    """
    Modular pipeline to train and evaluate 4 baseline ML algorithms on H2Crop data.
    Uses a dynamic file queue to guarantee a specific number of samples per class.
    Includes Optuna for hyperparameter optimization on a 20% validation split.
    """
    if not loader:
        print("Pipeline aborted: requiring loader")
        return

    if modality.lower() not in ["hyperspectral", "multispectral"]:
        print('Pipeline aborted: modality is neither "Hyperspectral" nor "Multispectral"')
        return

    class_action = class_action.lower()
    if class_action not in ["drop", "keep"]:
        print('Pipeline aborted: class_action must be either "drop" or "keep"')
        return

    print(f"\n{'='*60}")
    print(f"STARTING DYNAMIC PIPELINE FOR: {modality.upper()}")
    print(f"Action: {class_action.upper()} classes {class_list}")
    print(f"Target: {samples_per_class} samples per class")
    print(f"{'='*60}")

    # Directories setup
    os.makedirs(os.path.join(save_results_dir, modality), exist_ok=True)
    config_path = os.path.join(save_results_dir, modality, "configuration.txt")
    
    with open(config_path, "w") as f:
        f.write("--- H2Crop ML Dynamic Pipeline Configuration ---\n")
        f.write(f"modality: {modality}\n")
        f.write(f"detail_layer: {detail_layer}\n")
        f.write(f"static: {static}\n")
        f.write(f"keep_prior: {keep_prior}\n")
        f.write(f"samples_per_class: {samples_per_class}\n")
        f.write(f"allow_resample: {allow_resample}\n")
        f.write(f"class_action: {class_action}\n")
        f.write(f"class_list: {class_list}\n")
        f.write(f"tuning: Optuna TPESampler\n")
        
    print(f"Configuration saved to {config_path}")
    
    # -------------------------------------------------------------
    # Dynamic Queue and Resampling Logic
    # -------------------------------------------------------------
    master_queue = list(file_list)
    used_files = set(file_list)
    
    if allow_resample:
        print("\n[Resampling Enabled] Fetching global file list to act as buffer...")
        # Grab a large limit to ensure we have access to the whole dataset if needed
        all_possible_files = loader.get_file_list(from_train=False, limit=999999) 
        additional_files = [f for f in all_possible_files if f not in used_files]
        
        # Shuffle to ensure we aren't pulling geographically biased sequential files
        np.random.seed(42) 
        np.random.shuffle(additional_files)
        master_queue.extend(additional_files)
        print(f"Buffer populated. Total available files in queue: {len(master_queue)}")

    # Dictionary to collect X matrices per class: {class_id: [X_array1, X_array2, ...]}
    collected_X = {}
    target_classes = set(class_list) if (class_action == "keep" and class_list is not None) else None
    
    chunk_size = 50  # Load 50 files at a time to prevent RAM overload
    files_processed = 0

    print("\nStarting dynamic pixel extraction...")
    
    while master_queue:
        # Pop the next chunk of files
        batch_files = master_queue[:chunk_size]
        master_queue = master_queue[chunk_size:]
        files_processed += len(batch_files)
        
        batch = loader.load_h5_data(
            file_list=batch_files, 
            detail_layer=detail_layer, 
            static=static, 
            data_type=modality, 
            keep_prior=keep_prior
        )
        
        if not batch:
            continue
            
        for sample in batch:
            X_img = sample[modality]
            y_img = sample['labels']
            
            if modality == "hyperspectral":
                X_img = loader.upsample_hyperspectral(X_img)
                
            X_img = np.transpose(X_img, (1, 2, 0))
            X_flat = X_img.reshape(-1, X_img.shape[-1])
            y_flat = y_img.reshape(-1)
            
            if keep_prior and 'prior' in sample:
                prior_flat = sample['prior'].reshape(-1, 1)
                X_flat = np.hstack((X_flat, prior_flat))
                
            # Filter classes
            if class_list is not None:
                if class_action == "keep":
                    valid_mask = np.isin(y_flat, class_list)
                elif class_action == "drop":
                    valid_mask = ~np.isin(y_flat, class_list)
                    
                X_flat = X_flat[valid_mask]
                y_flat = y_flat[valid_mask]

            if len(y_flat) == 0:
                continue
                
            # Distribute pixels to their respective class bins
            for class_id in np.unique(y_flat):
                mask = (y_flat == class_id)
                if class_id not in collected_X:
                    collected_X[class_id] = []
                collected_X[class_id].append(X_flat[mask])
                
        # --- Check if target samples_per_class is reached ---
        if target_classes:
            done = True
            for c in target_classes:
                count = sum(len(arr) for arr in collected_X.get(c, []))
                if count < samples_per_class:
                    done = False
                    break
            if done:
                print(f"\n--> Success! Target of {samples_per_class} samples reached for all target classes.")
                break
        else:
            # If action is 'drop', check if all discovered classes hit the target
            if len(collected_X) > 0:
                done = True
                for c, arrs in collected_X.items():
                    if sum(len(arr) for arr in arrs) < samples_per_class:
                        done = False
                        break
                if done:
                    print(f"\n--> Success! Target of {samples_per_class} samples reached for all discovered classes.")
                    break

    print(f"Total files read to satisfy targets: {files_processed}")

    # -------------------------------------------------------------
    # Truncating & Balancing Phase
    # -------------------------------------------------------------
    X_final_list = []
    y_final_list = []
    
    print(f"\n--- Final Class Distribution (Target: {samples_per_class}) ---")
    for class_id, arrs in collected_X.items():
        X_c = np.vstack(arrs)
        total_found = len(X_c)
        
        if total_found > samples_per_class:
            # Randomly undersample to the exact target
            idx = np.random.choice(total_found, samples_per_class, replace=False)
            X_c = X_c[idx]
            print(f" - Class {class_id}: Found {total_found} -> Downsampled to {samples_per_class}")
        else:
            print(f" - Class {class_id}: Found {total_found} -> Kept {total_found} (WARNING: Under target!)")
            
        X_final_list.append(X_c)
        y_final_list.append(np.full(len(X_c), class_id, dtype=np.int32))
        
    if not X_final_list:
        print("Pipeline aborted: No valid pixels extracted.")
        return
        
    X = np.vstack(X_final_list)
    y = np.concatenate(y_final_list)

    print(f"\nTotal pixels extracted after balancing: {X.shape[0]}")

    # -------------------------------------------------------------
    # 70/20/10 Train/Val/Test Split & Scaling
    # -------------------------------------------------------------
    print("\nSplitting into Train (70%), Validation (20%), and Test (10%) sets...")
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.10, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.20 / 0.90), random_state=42, stratify=y_temp)
    
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    # -------------------------------------------------------------
    # Model Tuning and Evaluation
    # -------------------------------------------------------------
    models_to_run = {
        "decision_tree": 30,
        "random_forest": 30,
        "logistic_regression": 20,
        "linear_svm": 20
    }

    for algo_name, n_trials in models_to_run.items():
        print(f"\n--> Tuning and Training {algo_name} with Optuna ({n_trials} trials)...")
        
        best_model, best_params = optimize_hyperparameters(
            model_name=algo_name,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            n_trials=n_trials,
            random_state=42
        )
        
        print(f"    Evaluating Best Model on Test Set...")
        y_pred = best_model.predict(X_test_scaled)
        
        algo_dir = os.path.join(save_results_dir, modality, algo_name)
        os.makedirs(algo_dir, exist_ok=True)
        
        taxonomy_key = f'Taxonomy_{detail_layer}'
        current_taxonomy = h2crop_taxonomy_dict.get(taxonomy_key, {})
        target_names = [current_taxonomy.get(c, f"Class {c}") for c in best_model.classes_]
        
        # Save Classification Report & Parameters
        report_path = os.path.join(algo_dir, "performance.txt")
        report = classification_report(y_test, y_pred, zero_division=0, target_names=target_names)
        
        with open(report_path, "w") as f:
            f.write(f"--- Best Optuna Hyperparameters ---\n")
            f.write(json.dumps(best_params, indent=4))
            f.write(f"\n\n--- Test Set Classification Report ---\n")
            f.write(report)
            
        print(f"    Saved Report & Params: {report_path}")
        
        # Save Confusion Matrix
        matrix_path = os.path.join(algo_dir, "confusion_matrix.png")
        fig_size = max(10, len(target_names) * 0.4)
        fig, ax = plt_subplots.subplots(figsize=(fig_size, fig_size * 0.8))
        
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, ax=ax, cmap='Blues', colorbar=False, display_labels=target_names
        )
        
        plt.title(f"Confusion Matrix: {algo_name}\n({modality} | Tuned via Optuna)")
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(matrix_path, dpi=300)
        plt.close(fig)
        
        print(f"    Saved Matrix: {matrix_path}")
        
    print(f"\nPipeline completed successfully! {modality.upper()} experiments saved.")