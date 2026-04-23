import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from utils import load_hyperspectral_dataset, normalize_features


def pipeline_standard_ml_algo(dataset_config_dict):
    """
    Trains and evaluates 4 baseline ML models.
    Applies Random Undersampling ONLY to the training set to prevent bias,
    leaving the test set completely imbalanced to reflect reality.
    """
    models = {
        'decision_tree': DecisionTreeClassifier(random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'linear_svm': LinearSVC(max_iter=1000, random_state=42, dual=False)
    }

    for dataset_name, config in dataset_config_dict.items():
        classes_to_drop = config['drop']
        target_train_samples = config['train_samples']
        
        print(f"\n{'='*60}")
        print(f"PROCESSING DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Load Data
        print(f"Loading data (dropping classes: {classes_to_drop})...")
        X, y = load_hyperspectral_dataset(dataset_name, classes_to_drop=classes_to_drop)
        
        # Normalize Features 
        print("Normalizing features...")
        X_norm, _ = normalize_features(X)
        
        # Train/Test Split (80/20 Stratified)
        print("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # SMART UNDERSAMPLING ON TRAINING DATA ONLY
        print(f"Undersampling training data to max {target_train_samples} samples per class...")
        X_train_bal = []
        y_train_bal = []
        
        for cls in np.unique(y_train):
            # Find all indices in the training set belonging to this class
            idx = np.where(y_train == cls)[0]
            
            # Determine how many to pick: the target limit, or whatever is available if it's smaller
            n_to_sample = min(target_train_samples, len(idx))
            
            # Randomly pick without replacement
            selected_idx = np.random.choice(idx, n_to_sample, replace=False)
            
            X_train_bal.append(X_train[selected_idx])
            y_train_bal.append(y_train[selected_idx])
            
        # Recombine the balanced subsets back into a single training matrix
        X_train_bal = np.vstack(X_train_bal)
        y_train_bal = np.concatenate(y_train_bal)
        
        print(f"  -> Original train size: {len(X_train)}")
        print(f"  -> Balanced train size: {len(X_train_bal)}")
        print(f"  -> Test size (untouched): {len(X_test)}")
        
        # Train and Evaluate each model
        for model_name, model in models.items():
            print(f"\n  [Training {model_name}...]")
            
            # Fit on the BALANCED training set
            model.fit(X_train_bal, y_train_bal)
            
            # Predict on the IMBALANCED test set
            y_pred = model.predict(X_test)
            
            # Generate metrics
            report = classification_report(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            # Create Directory Structure (files_2/model_name/dataset_name/)
            dir_path = os.path.join("files_2", model_name, dataset_name)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save Performance Report
            report_path = os.path.join(dir_path, "performance.txt")
            with open(report_path, "w") as f:
                f.write(f"--- Classification Report ---\n")
                f.write(f"Model:   {model_name}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Dropped: {classes_to_drop}\n")
                f.write(f"Train samples per class cap: {target_train_samples}\n")
                f.write("-" * 40 + "\n\n")
                f.write(report)
                
            # Save Confusion Matrix
            cm_path = os.path.join(dir_path, "confusion_matrix.txt")
            with open(cm_path, "w") as f:
                f.write(f"--- Confusion Matrix ---\n")
                f.write(f"Model:   {model_name}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write("-" * 40 + "\n\n")
                np.savetxt(f, cm, fmt='%d')
                
            print(f"     Saved -> {dir_path}/")