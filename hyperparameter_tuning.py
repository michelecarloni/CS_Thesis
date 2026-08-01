import optuna

# Import standard Scikit-Learn for the fast CPU Decision Tree
from sklearn.tree import DecisionTreeClassifier

# Import NVIDIA RAPIDS (cuML) for GPU-accelerated algorithms
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.linear_model import LogisticRegression as cuLogReg
from cuml.svm import LinearSVC as cuSVC

# Silence verbose Optuna logging 
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_hyperparameters(model_name, X_train, y_train, X_val, y_val, n_trials=20, random_state=42):
    """
    Unified Optuna tuner utilizing NVIDIA RAPIDS (cuML) for GPU acceleration.
    Returns the best fitted model trained on X_train.
    """
    
    def objective(trial):
        # --- 1. Decision Tree (CPU - Throttled Search Space) ---
        if model_name == "decision_tree":
            params = {
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_depth': trial.suggest_int('max_depth', 3, 15), # Throttled from 30 to 15
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
                # Note: cuML RF doesn't take min_samples_split in the exact same format, it relies on max_depth and min_samples_leaf
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 1.0]),
                'random_state': random_state
            }
            model = cuRF(**params)

        # --- 3. Linear SVC (GPU Accelerated) ---
        elif model_name == "linear_svm":
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'max_iter': 2000
                # cuML SVC runs natively on CUDA
            }
            model = cuSVC(**params)

        # --- 4. Logistic Regression (GPU Accelerated) ---
        elif model_name == "logistic_regression":
            params = {
                'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
                'max_iter': 4000
                # cuML automatically uses optimal GPU solvers (QN / L-BFGS)
            }
            model = cuLogReg(**params)

        else:
            raise ValueError(f"Unknown model_name: '{model_name}'")

        # Fit on Training Set & Evaluate on Validation Set
        # cuML expects float32/float64 inputs, which your pipeline already handles!
        model.fit(X_train, y_train)
        return model.score(X_val, y_val)

    # Run Optuna Study
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"    [Optuna] Best Val Accuracy for {model_name}: {study.best_value:.4f}")
    print(f"    [Optuna] Best Params: {study.best_params}")

    # Re-instantiate and return the best model trained on X_train
    best_params = study.best_params
    best_model = _build_model(model_name, best_params, random_state)
    best_model.fit(X_train, y_train)

    return best_model, best_params


def _build_model(model_name, params, random_state):
    """Internal helper to instantiate a model from best parameters."""
    if model_name == "decision_tree":
        return DecisionTreeClassifier(**params, random_state=random_state)
        
    elif model_name == "random_forest":
        return cuRF(**params, random_state=random_state)
        
    elif model_name == "linear_svm":
        return cuSVC(**params, max_iter=2000)
        
    elif model_name == "logistic_regression":
        return cuLogReg(**params, max_iter=4000)