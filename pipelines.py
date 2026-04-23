import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from utils import load_hyperspectral_dataset, normalize_features
import matplotlib.pyplot as plt
import seaborn as sns

def pipeline_standard_ml_algo(dataset_config_dict):
    """
    Trains and evaluates 4 baseline ML models.
    Applies Random Undersampling to the training set.
    Saves performance reports and confusion matrix plots in 'results_2'.
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
        
        # 1. Load Data
        print(f"Loading data (dropping classes: {classes_to_drop})...")
        X, y = load_hyperspectral_dataset(dataset_name, classes_to_drop=classes_to_drop)
        
        # 2. Normalize Features 
        print("Normalizing features...")
        X_norm, _ = normalize_features(X)
        
        # 3. Train/Test Split (80/20 Stratified)
        print("Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_norm, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 4. SMART UNDERSAMPLING ON TRAINING DATA ONLY
        print(f"Undersampling training data to max {target_train_samples} samples per class...")
        X_train_bal = []
        y_train_bal = []
        
        for cls in np.unique(y_train):
            idx = np.where(y_train == cls)[0]
            n_to_sample = min(target_train_samples, len(idx))
            selected_idx = np.random.choice(idx, n_to_sample, replace=False)
            
            X_train_bal.append(X_train[selected_idx])
            y_train_bal.append(y_train[selected_idx])
            
        X_train_bal = np.vstack(X_train_bal)
        y_train_bal = np.concatenate(y_train_bal)
        
        print(f"  -> Original train size: {len(X_train)}")
        print(f"  -> Balanced train size: {len(X_train_bal)}")
        print(f"  -> Test size (untouched): {len(X_test)}")
        
        # 5. Train and Evaluate each model
        for model_name, model in models.items():
            print(f"\n  [Training {model_name}...]")
            
            # Fit on the BALANCED training set
            model.fit(X_train_bal, y_train_bal)
            
            # Predict on the IMBALANCED test set
            y_pred = model.predict(X_test)
            
            # Generate metrics
            report = classification_report(y_test, y_pred, zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            # 6. Create Directory Structure (results_2/model_name/dataset_name/)
            dir_path = os.path.join("results_2", model_name, dataset_name)
            os.makedirs(dir_path, exist_ok=True)
            
            # 7. Save Performance Report
            report_path = os.path.join(dir_path, "performance.txt")
            with open(report_path, "w") as f:
                f.write(f"--- Classification Report ---\n")
                f.write(f"Model:   {model_name}\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"Dropped: {classes_to_drop}\n")
                f.write(f"Train samples per class cap: {target_train_samples}\n")
                f.write("-" * 40 + "\n\n")
                f.write(report)
                
            # 8. Save Confusion Matrix as a PNG image
            cm_path = os.path.join(dir_path, "confusion_matrix.png")
            
            # Dynamically get the unique classes that are actually in the test set for labeling
            unique_classes = np.unique(y_test)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=unique_classes, yticklabels=unique_classes)
            plt.title(f"Confusion Matrix: {model_name.replace('_', ' ').title()}\nDataset: {dataset_name.replace('_', ' ').title()}", fontsize=14, pad=15)
            plt.ylabel("True Class Label", fontsize=12)
            plt.xlabel("Predicted Class Label", fontsize=12)
            plt.tight_layout()
            
            # Save the figure and close it to free up memory
            plt.savefig(cm_path, dpi=300)
            plt.close()
                
            print(f"     Saved -> {dir_path}/")