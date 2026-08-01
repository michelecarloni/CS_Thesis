import optuna
import warnings

# Scikit-Learn (CPU Models - Kept only for Decision Tree)
from sklearn.tree import DecisionTreeClassifier

# NVIDIA RAPIDS cuML (GPU Models)
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
    # Import the Mini-Batch SGD Classifier which is safe for 8GB VRAM
    from cuml.linear_model import MBSGDClassifier as cuMBSGD
    # Import the GPU-accelerated One-vs-Rest wrapper for multiclass
    from cuml.multiclass import OneVsRestClassifier as cuOvR
except ImportError:
    pass

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

        # --- 3. Linear SVM (GPU via Mini-Batch SGD wrapped in OvR) ---
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
            model = cuOvR(estimator=base_model)

        # --- 4. Logistic Regression (GPU via Mini-Batch SGD wrapped in OvR) ---
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
            model = cuOvR(estimator=base_model)

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
        return cuOvR(estimator=cuMBSGD(**params))
        
    elif model_name == "logistic_regression":
        return cuOvR(estimator=cuMBSGD(**params))