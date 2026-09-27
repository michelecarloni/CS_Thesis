import optuna
import warnings
import cupy as cp
from sklearn.metrics import f1_score

# Scikit-Learn (CPU Models)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# NVIDIA RAPIDS cuML (GPU Models)
try:
    from cuml.ensemble import RandomForestClassifier as cuRF
except ImportError:
    pass

optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_hyperparameters(model_name, X_train, y_train, X_val, y_val, n_trials=20, random_state=42, use_gpu=True):
    """
    Optuna tuner safely routing Logistic Regression and Linear SVM to the CPU, 
    while accelerating Random Forest and Decision Tree on the GPU.
    """
    cpu_only_models = ["logistic_regression", "linear_svm"]
    
    # Pre-cast to GPU if applicable to speed up the tuning loop
    if use_gpu and model_name not in cpu_only_models:
        X_train_fit = cp.asarray(X_train)
        X_val_eval = cp.asarray(X_val)
    else:
        X_train_fit = X_train
        X_val_eval = X_val

    def objective(trial):
        if model_name == "decision_tree":
            max_depth = trial.suggest_int('max_depth', 3, 15)
            if use_gpu:
                # Simulated GPU Decision Tree
                model = cuRF(n_estimators=1, max_depth=max_depth, max_features=1.0, random_state=random_state)
            else:
                criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
                model = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, random_state=random_state)

        elif model_name == "random_forest":
            max_depth = trial.suggest_int('max_depth', 5, 15)
            n_estimators = trial.suggest_int('n_estimators', 50, 200, step=50)
            if use_gpu:
                model = cuRF(n_estimators=n_estimators, max_depth=max_depth, max_features='sqrt', random_state=random_state)
            else:
                max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, n_jobs=-1, random_state=random_state)

        elif model_name == "linear_svm":
            C = trial.suggest_float('C', 1e-4, 1e2, log=True)
            model = LinearSVC(C=C, max_iter=1000, dual=False, random_state=random_state)

        elif model_name == "logistic_regression":
            C = trial.suggest_float('C', 1e-4, 1e2, log=True)
            model = LogisticRegression(C=C, max_iter=1000, n_jobs=-1, random_state=random_state)
            
        else:
            raise ValueError(f"Unknown model_name: '{model_name}'")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train_fit, y_train)
            y_val_pred = model.predict(X_val_eval)
            
            if use_gpu and model_name not in cpu_only_models:
                y_val_pred = cp.asnumpy(y_val_pred)
                
        return f1_score(y_val, y_val_pred, average='macro')

    print(f"--- Running Optuna Tuning for {model_name} ({n_trials} Trials) ---")
    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print(f"    [Optuna] Best Val Macro F1-Score: {study.best_value:.4f}")
    print(f"    [Optuna] Best Params: {study.best_params}")

    # Rebuild the final optimal model
    best_params = study.best_params
    
    if model_name == "decision_tree":
        if use_gpu:
            best_model = cuRF(n_estimators=1, max_depth=best_params['max_depth'], max_features=1.0, random_state=random_state)
        else:
            best_model = DecisionTreeClassifier(**best_params, random_state=random_state)
            
    elif model_name == "random_forest":
        if use_gpu:
            best_model = cuRF(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], max_features='sqrt', random_state=random_state)
        else:
            best_model = RandomForestClassifier(**best_params, n_jobs=-1, random_state=random_state)
            
    elif model_name == "linear_svm":
        best_model = LinearSVC(C=best_params['C'], max_iter=1000, dual=False, random_state=random_state)
        
    elif model_name == "logistic_regression":
        best_model = LogisticRegression(C=best_params['C'], max_iter=1000, n_jobs=-1, random_state=random_state)

    # Train final model on Train Set
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best_model.fit(X_train_fit, y_train)

    if use_gpu and model_name not in cpu_only_models:
        del X_train_fit, X_val_eval
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    return best_model, best_params