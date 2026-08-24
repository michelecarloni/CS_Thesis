import optuna
import warnings
import copy
import torch
import torch.nn as nn
import torch.optim as optim

# Scikit-Learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# NVIDIA RAPIDS cuML (GPU Models)
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLR
    from cuml.svm import LinearSVC as cuSVC
except ImportError:
    pass

optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- STANDARD ML OPTIMIZATION ---
def optimize_hyperparameters(model_name, X_train, y_train, X_val, y_val, n_trials=20, random_state=42, use_gpu=True):
    """
    Unified Optuna tuner optimized for local 8GB VRAM and 16GB RAM.
    Bypasses OneVsRest classifiers to prevent fatal Out-Of-Memory crashes.
    """
    def objective(trial):
        if model_name == "decision_tree":
            params = {
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_depth': trial.suggest_int('max_depth', 3, 15), # Capped at 15 for memory safety
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': random_state
            }
            model = DecisionTreeClassifier(**params)

        elif model_name == "random_forest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 15), # Capped at 15 for memory safety
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': random_state
            }
            if use_gpu:
                model = cuRF(**params)
            else:
                model = RandomForestClassifier(**params, n_jobs=1)

        elif model_name == "linear_svm":
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            }
            if use_gpu:
                model = cuSVC(C=params['C'], max_iter=1000, penalty='l2')
            else:
                model = LinearSVC(C=params['C'], max_iter=1000, dual=False)

        elif model_name == "logistic_regression":
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            }
            if use_gpu:
                model = cuLR(C=params['C'], max_iter=1000)
            else:
                model = LogisticRegression(C=params['C'], max_iter=1000, n_jobs=1)
        else:
            raise ValueError(f"Unknown model_name: '{model_name}'")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
            
        return model.score(X_val, y_val)

    print(f"\n--- Running Optuna Tuning for {model_name} ({n_trials} Trials) ---")
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"    [Optuna] Best Val Accuracy: {study.best_value:.4f}")
    print(f"    [Optuna] Best Params: {study.best_params}")

    # Rebuild and train the absolute best model
    best_params = study.best_params
    best_model = _build_model(model_name, best_params, random_state, use_gpu)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_model.fit(X_train, y_train)

    return best_model, best_params

def _build_model(model_name, params, random_state, use_gpu):
    """Instantiates the optimal model strictly using native multiclass logic."""
    if model_name == "decision_tree":
        return DecisionTreeClassifier(**params, random_state=random_state)
    elif model_name == "random_forest":
        return cuRF(**params, random_state=random_state) if use_gpu else RandomForestClassifier(**params, n_jobs=1, random_state=random_state)
    elif model_name == "linear_svm":
        return cuSVC(C=params['C'], max_iter=1000) if use_gpu else LinearSVC(C=params['C'], max_iter=1000, dual=False)
    elif model_name == "logistic_regression":
        return cuLR(C=params['C'], max_iter=1000) if use_gpu else LogisticRegression(C=params['C'], max_iter=1000, n_jobs=1)

# --- DEEP LEARNING OPTIMIZATION ---
def optimize_cnn_hyperparameters(model, train_loader, val_loader, initial_model_state, n_trials=10, epochs_per_trial=5, use_gpu=True):
    """
    Optuna optimization logic for PyTorch CNNs utilizing Mixed Precision (AMP) and tqdm.
    Exclusively utilizes AdamW as the optimization algorithm[cite: 1].
    """
    from tqdm import tqdm 
    
    def objective(trial):
        model.load_state_dict(copy.deepcopy(initial_model_state))[cite: 1]
        
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)[cite: 1]
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)[cite: 1]
        
        # Strictly enforced AdamW optimizer per requirements
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            
        criterion = nn.CrossEntropyLoss()[cite: 1]
        
        for epoch in range(epochs_per_trial):[cite: 1]
            model.train()[cite: 1]
            
            train_loop = tqdm(train_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{epochs_per_trial} [Train]", leave=False)[cite: 1]
            
            for batch_X, batch_y in train_loop:[cite: 1]
                if use_gpu and torch.cuda.is_available():[cite: 1]
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()[cite: 1]
                    
                if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():[cite: 1]
                    continue[cite: 1]

                optimizer.zero_grad()[cite: 1]
                
                outputs = model(batch_X)[cite: 1]
                loss = criterion(outputs, batch_y)[cite: 1]
                loss.backward()[cite: 1]
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)[cite: 1]
                optimizer.step()[cite: 1]
                
                train_loop.set_postfix(loss=loss.item())[cite: 1]
                
            model.eval()[cite: 1]
            correct, total = 0, 0[cite: 1]
            
            val_loop = tqdm(val_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{epochs_per_trial} [Val]", leave=False)[cite: 1]
            
            with torch.no_grad():[cite: 1]
                for batch_X, batch_y in val_loop:[cite: 1]
                    if use_gpu and torch.cuda.is_available():[cite: 1]
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()[cite: 1]
                        
                    outputs = model(batch_X)[cite: 1]
                    _, predicted = torch.max(outputs.data, 1)[cite: 1]
                    total += batch_y.size(0)[cite: 1]
                    correct += (predicted == batch_y).sum().item()[cite: 1]
                    
            val_accuracy = correct / total[cite: 1]
            trial.report(val_accuracy, epoch)[cite: 1]
            
            if trial.should_prune():[cite: 1]
                raise optuna.exceptions.TrialPruned()[cite: 1]
                
        return val_accuracy[cite: 1]

    print(f"\n--- Running Optuna Tuning ({n_trials} Trials) ---")[cite: 1]
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))[cite: 1]
    study.optimize(objective, n_trials=n_trials)[cite: 1]
    
    print(f"\n[Optuna] Best Trial: {study.best_trial.number}")[cite: 1]
    print(f"[Optuna] Best Validation Accuracy: {study.best_value:.4f}")[cite: 1]
    
    return study.best_params[cite: 1]