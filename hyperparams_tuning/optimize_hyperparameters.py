import optuna
import warnings
import cupy as cp

# Scikit-Learn (CPU Models)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

# NVIDIA RAPIDS cuML (GPU Models - Only Random Forest kept)
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
except ImportError:
    pass

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_hyperparameters(model_name, X_train, y_train, X_val, y_val, n_trials=20, random_state=42, use_gpu=True):
    """
    Unified Optuna tuner optimized for local hardware constraints.
    Logistic Regression and Linear SVM are strictly pinned to the CPU to avoid
    cuBLAS serialization crashes. Random Forest can optionally run on GPU.
    Data is assumed to be physically balanced beforehand (no class_weight needed).
    """
    
    cpu_only_models = ["decision_tree", "logistic_regression", "linear_svm"]
    
    # Pre-cast data to the GPU memory pool ONLY if we are tuning cuRF.
    # This bypasses any buggy zero-copy conversions inside cuML.
    if use_gpu and model_name not in cpu_only_models:
        X_train_fit = cp.asarray(X_train)
    else:
        X_train_fit = X_train

    def objective(trial):
        if model_name == "decision_tree":
            params = {
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': random_state
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
                model = RandomForestClassifier(**params, n_jobs=-1)

        elif model_name == "linear_svm":
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            }
            # Strictly pinned to CPU
            model = LinearSVC(C=params['C'], max_iter=1000, dual=False)

        elif model_name == "logistic_regression":
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            }
            # Strictly pinned to CPU
            model = LogisticRegression(C=params['C'], max_iter=1000, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model_name: '{model_name}'")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_fit, y_train)
            
        # Evaluation Block
        if use_gpu and model_name not in cpu_only_models:
            X_val_gpu = cp.asarray(X_val)
            y_val_pred = model.predict(X_val_gpu)
            y_val_pred = cp.asnumpy(y_val_pred)
            del X_val_gpu
        else:
            y_val_pred = model.predict(X_val)
            
        macro_f1 = f1_score(y_val, y_val_pred, average='macro')
        
        return macro_f1

    print(f"\n--- Running Optuna Tuning for {model_name} ({n_trials} Trials) ---")
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"    [Optuna] Best Val Macro F1-Score: {study.best_value:.4f}")
    print(f"    [Optuna] Best Params: {study.best_params}")

    # Clean up GPU training data before rebuilding the final model
    if use_gpu and model_name not in cpu_only_models:
        del X_train_fit
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    # Rebuild and train the absolute best model
    best_params = study.best_params
    best_model = _build_model(model_name, best_params, random_state, use_gpu)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_gpu and model_name not in cpu_only_models:
            X_train_gpu = cp.asarray(X_train)
            best_model.fit(X_train_gpu, y_train)
            del X_train_gpu
        else:
            best_model.fit(X_train, y_train)

    return best_model, best_params

def _build_model(model_name, params, random_state, use_gpu):
    """Instantiates the optimal model, strictly keeping linear models on the CPU."""
    cpu_only_models = ["decision_tree", "logistic_regression", "linear_svm"]
    
    if model_name == "decision_tree":
        return DecisionTreeClassifier(**params, random_state=random_state)
        
    elif model_name == "random_forest":
        if use_gpu:
            return cuRF(**params, random_state=random_state)
        else:
            return RandomForestClassifier(**params, n_jobs=-1, random_state=random_state)
            
    elif model_name == "linear_svm":
        return LinearSVC(C=params['C'], max_iter=1000, dual=False)
            
    elif model_name == "logistic_regression":
        return LogisticRegression(C=params['C'], max_iter=1000, n_jobs=-1)