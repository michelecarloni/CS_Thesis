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
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

# NVIDIA RAPIDS cuML (GPU Models)
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import LogisticRegression as cuLR
    from cuml.svm import LinearSVC as cuSVC
except ImportError:
    pass

optuna.logging.set_verbosity(optuna.logging.WARNING)

# STANDARD ML OPTIMIZATION
def optimize_hyperparameters(model_name, X_train, y_train, X_val, y_val, n_trials=20, random_state=42, use_gpu=True):
    """
    Unified Optuna tuner optimized for local 8GB VRAM and 16GB RAM.
    Optimizes for Macro F1-Score and applies class/sample weights to penalize majority classes.
    """
    
    # Precompute sample weights to use on GPU models that don't support 'class_weight' strings
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    def objective(trial):
        if model_name == "decision_tree":
            params = {
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': random_state,
                'class_weight': 'balanced'  # Native CPU balancing
            }
            model = DecisionTreeClassifier(**params)

        elif model_name == "random_forest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 15),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': random_state
            }
            if use_gpu:
                model = cuRF(**params)
            else:
                params['class_weight'] = 'balanced'
                model = RandomForestClassifier(**params, n_jobs=1)

        elif model_name == "linear_svm":
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            }
            if use_gpu:
                model = cuSVC(C=params['C'], max_iter=1000, penalty='l2')
            else:
                model = LinearSVC(C=params['C'], max_iter=1000, dual=False, class_weight='balanced')

        elif model_name == "logistic_regression":
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            }
            if use_gpu:
                model = cuLR(C=params['C'], max_iter=1000)
            else:
                model = LogisticRegression(C=params['C'], max_iter=1000, n_jobs=1, class_weight='balanced')
        else:
            raise ValueError(f"Unknown model_name: '{model_name}'")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Scikit-learn models natively handle class weighting in their constructors
            if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, LinearSVC)):
                model.fit(X_train, y_train)
            else:
                # For cuML, we attempt to pass the mathematically computed sample weights directly
                try:
                    model.fit(X_train, y_train, sample_weight=sample_weights)
                except TypeError:
                    # If a specific cuML version rejects sample_weight, fallback to standard fit.
                    # The Macro F1 Optuna target will still severely penalize it for missing crops.
                    model.fit(X_train, y_train)
            
        # Optimize for Macro F1 instead of Overall Accuracy
        y_val_pred = model.predict(X_val)
        macro_f1 = f1_score(y_val, y_val_pred, average='macro')
        
        return macro_f1

    print(f"\n--- Running Optuna Tuning for {model_name} ({n_trials} Trials) ---")
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"    [Optuna] Best Val Macro F1-Score: {study.best_value:.4f}")
    print(f"    [Optuna] Best Params: {study.best_params}")

    # Rebuild and train the absolute best model
    best_params = study.best_params
    best_model = _build_model(model_name, best_params, random_state, use_gpu)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(best_model, (DecisionTreeClassifier, RandomForestClassifier, LogisticRegression, LinearSVC)):
            best_model.fit(X_train, y_train)
        else:
            try:
                best_model.fit(X_train, y_train, sample_weight=sample_weights)
            except TypeError:
                best_model.fit(X_train, y_train)

    return best_model, best_params

def _build_model(model_name, params, random_state, use_gpu):
    """Instantiates the optimal model incorporating native balancing where supported."""
    if model_name == "decision_tree":
        return DecisionTreeClassifier(**params, random_state=random_state, class_weight='balanced')
        
    elif model_name == "random_forest":
        if use_gpu:
            return cuRF(**params, random_state=random_state)
        else:
            return RandomForestClassifier(**params, n_jobs=1, random_state=random_state, class_weight='balanced')
            
    elif model_name == "linear_svm":
        if use_gpu:
            return cuSVC(C=params['C'], max_iter=1000)
        else:
            return LinearSVC(C=params['C'], max_iter=1000, dual=False, class_weight='balanced')
            
    elif model_name == "logistic_regression":
        if use_gpu:
            return cuLR(C=params['C'], max_iter=1000)
        else:
            return LogisticRegression(C=params['C'], max_iter=1000, n_jobs=1, class_weight='balanced')

# --- DEEP LEARNING OPTIMIZATION ---
def optimize_cnn_hyperparameters(model, train_loader, val_loader, initial_model_state, n_trials=10, epochs_per_trial=5, use_gpu=True):
    """
    Optuna optimization logic for PyTorch CNNs utilizing Mixed Precision (AMP) and tqdm.
    Exclusively utilizes AdamW as the optimization algorithm.
    """
    from tqdm import tqdm 
    
    def objective(trial):
        model.load_state_dict(copy.deepcopy(initial_model_state))
        
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs_per_trial):
            model.train()
            train_loop = tqdm(train_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{epochs_per_trial} [Train]", leave=False)
            
            for batch_X, batch_y in train_loop:
                if use_gpu and torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                    
                if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                    continue

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loop.set_postfix(loss=loss.item())
                
            model.eval()
            correct, total = 0, 0
            val_loop = tqdm(val_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{epochs_per_trial} [Val]", leave=False)
            
            with torch.no_grad():
                for batch_X, batch_y in val_loop:
                    if use_gpu and torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                        
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
            val_accuracy = correct / total
            trial.report(val_accuracy, epoch)
            
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        return val_accuracy

    print(f"\n--- Running Optuna Tuning ({n_trials} Trials) ---")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n[Optuna] Best Trial: {study.best_trial.number}")
    print(f"[Optuna] Best Validation Accuracy: {study.best_value:.4f}")
    
    return study.best_params