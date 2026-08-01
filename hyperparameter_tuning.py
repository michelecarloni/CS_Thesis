import optuna
import warnings

# Scikit-Learn (CPU Model)
from sklearn.tree import DecisionTreeClassifier

# NVIDIA RAPIDS cuML (GPU Models)
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.svm import LinearSVC as cuSVC
from cuml.linear_model import LogisticRegression as cuLogReg

optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_hyperparameters(model_name, X_train, y_train, X_val, y_val, n_trials=20, random_state=42, use_gpu=True):
    """
    Unified Optuna tuner utilizing the HPC Cluster.
    use_gpu=True by default. Decision Tree remains on CPU; all others run on cuML.
    """
    def objective(trial):
        # --- 1. Decision Tree (Always CPU - Fast and robust) ---
        if model_name == "decision_tree":
            params = {
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': random_state
            }
            model = DecisionTreeClassifier(**params)

        # --- 2. Random Forest (GPU Accelerated) ---
        elif model_name == "random_forest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 250, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),
                'random_state': random_state
            }
            model = cuRF(**params)

        # --- 3. Linear SVM (GPU Accelerated) ---
        elif model_name == "linear_svm":
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'max_iter': 5000
            }
            model = cuSVC(**params)

        # --- 4. Logistic Regression (GPU Accelerated) ---
        elif model_name == "logistic_regression":
            params = {
                'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
                'max_iter': 5000
            }
            model = cuLogReg(**params)

        else:
            raise ValueError(f"Unknown model_name: '{model_name}'")

        # Catch cuML convergence warnings to keep HPC logs clean
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
        return cuSVC(**params, max_iter=5000)
        
    elif model_name == "logistic_regression":
        # cuLogReg doesn't support the random_state parameter directly in the same way Scikit-Learn does
        return cuLogReg(**params)