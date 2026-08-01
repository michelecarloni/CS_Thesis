import optuna
import warnings

# Scikit-Learn (CPU Models)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC as skSVC
from sklearn.linear_model import LogisticRegression as skLogReg

# NVIDIA RAPIDS cuML (GPU Models)
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    from cuml.svm import LinearSVC as cuSVC
    from cuml.linear_model import LogisticRegression as cuLogReg
except ImportError:
    pass # Fails gracefully if cuML isn't installed locally

optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_hyperparameters(model_name, X_train, y_train, X_val, y_val, n_trials=20, random_state=42, use_gpu=True):
    """
    Unified Optuna tuner utilizing CPU and GPU processing dynamically.
    use_gpu=True by default. 
    Decision Tree is ALWAYS CPU. Random Forest is ALWAYS GPU.
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

        # --- 3. Linear SVM (GPU or CPU) ---
        elif model_name == "linear_svm":
            if use_gpu:
                params = {
                    'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                    'max_iter': 5000
                }
                model = cuSVC(**params)
            else:
                params = {
                    'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                    'max_iter': 5000,
                    'random_state': random_state
                }
                model = skSVC(**params)

        # --- 4. Logistic Regression (GPU or CPU) ---
        elif model_name == "logistic_regression":
            if use_gpu:
                params = {
                    'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
                    'max_iter': 5000
                }
                model = cuLogReg(**params)
            else:
                params = {
                    'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
                    'solver': 'saga',
                    'max_iter': 5000,
                    'n_jobs': -1,
                    'random_state': random_state
                }
                model = skLogReg(**params)

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
    best_model = _build_model(model_name, best_params, random_state, use_gpu)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_model.fit(X_train, y_train)

    return best_model, best_params

def _build_model(model_name, params, random_state, use_gpu):
    """Internal helper to instantiate a model from best parameters dynamically."""
    if model_name == "decision_tree":
        return DecisionTreeClassifier(**params, random_state=random_state)
        
    elif model_name == "random_forest":
        return cuRF(**params, random_state=random_state)
        
    elif model_name == "linear_svm":
        if use_gpu:
            return cuSVC(**params, max_iter=5000)
        else:
            return skSVC(**params, random_state=random_state, max_iter=5000)
        
    elif model_name == "logistic_regression":
        if use_gpu:
            return cuLogReg(**params)
        else:
            return skLogReg(**params, random_state=random_state, n_jobs=-1)