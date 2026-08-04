import optuna
import warnings
import copy
import torch
import torch.nn as nn
import torch.optim as optim

# Scikit-Learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier as skOvR # Imported Scikit-Learn's stable meta-estimator

# NVIDIA RAPIDS cuML (GPU Models)
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.linear_model import MBSGDClassifier as cuMBSGD
except ImportError:
    pass

optuna.logging.set_verbosity(optuna.logging.WARNING)



# For Standard ML algorithm
def optimize_hyperparameters(model_name, X_train, y_train, X_val, y_val, n_trials=20, random_state=42, use_gpu=True):
    """
    Unified Optuna tuner optimized for local 8GB VRAM (RTX 4060).
    Decision Tree is ALWAYS CPU. Random Forest, SVM, and LogReg are ALWAYS GPU.
    """
    def objective(trial):
        # --- 1. Decision Tree (Always CPU) ---
        if model_name == "decision_tree":
            params = {
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': random_state
            }
            model = DecisionTreeClassifier(**params)

        # --- 2. Random Forest (Always GPU) ---
        elif model_name == "random_forest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 250, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),
                'random_state': random_state
            }
            model = cuRF(**params)

        # --- 3. Linear SVM (GPU via Mini-Batch SGD wrapped safely in skOvR) ---
        elif model_name == "linear_svm":
            params = {
                'loss': 'hinge', 
                'penalty': 'l2',
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'batch_size': 2048,
                'epochs': 100,
                'learning_rate': 'adaptive'
            }
            base_model = cuMBSGD(**params)
            model = skOvR(estimator=base_model) # Using stable CPU wrapper

        # --- 4. Logistic Regression (GPU via Mini-Batch SGD wrapped safely in skOvR) ---
        elif model_name == "logistic_regression":
            params = {
                'loss': 'log', 
                'penalty': 'l2',
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
                'batch_size': 2048, 
                'epochs': 100,
                'learning_rate': 'adaptive'
            }
            base_model = cuMBSGD(**params)
            model = skOvR(estimator=base_model) # Using stable CPU wrapper

        else:
            raise ValueError(f"Unknown model_name: '{model_name}'")

        # Catch convergence warnings to keep logs clean
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
            
        return model.score(X_val, y_val)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"    [Optuna] Best Val Accuracy for {model_name}: {study.best_value:.4f}")
    print(f"    [Optuna] Best Params: {study.best_params}")

    best_params = study.best_params
    best_model = _build_model(model_name, best_params, random_state)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_model.fit(X_train, y_train)

    return best_model, best_params


def _build_model(model_name, params, random_state):
    """Internal helper to instantiate a model from best parameters."""
    if model_name == "decision_tree":
        return DecisionTreeClassifier(**params, random_state=random_state)
        
    elif model_name == "random_forest":
        return cuRF(**params, random_state=random_state)
        
    elif model_name == "linear_svm":
        return skOvR(estimator=cuMBSGD(**params))
        
    elif model_name == "logistic_regression":
        return skOvR(estimator=cuMBSGD(**params))



# For models that works with a Convolutional layer
def optimize_cnn_hyperparameters(model, train_loader, val_loader, initial_model_state, n_trials=10, epochs_per_trial=5, use_gpu=True):
    """
    Optuna optimization logic for PyTorch CNNs utilizing Mixed Precision (AMP) and tqdm.
    """
    from tqdm import tqdm 
    
    def objective(trial):
        model.load_state_dict(copy.deepcopy(initial_model_state))
        
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        
        if optimizer_name == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
            
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(epochs_per_trial):
            model.train()
            
            # --- Added TQDM Progress Bar for Training ---
            train_loop = tqdm(train_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{epochs_per_trial} [Train]", leave=False)
            
            for batch_X, batch_y in train_loop:
                if use_gpu and torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                    
                # Keep the input trap just as a best practice!
                if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                    continue  

                optimizer.zero_grad()
                
                # AMP DISABLED: Standard 32-bit Forward Pass (Improved with TF32)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                                
                # Standard 32-bit Backward Pass
                loss.backward()
                
                # Keep gradient clipping to prevent optimizer explosions
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loop.set_postfix(loss=loss.item())
                
            model.eval()
            correct, total = 0, 0
            
            # Add TQDM progress bar
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