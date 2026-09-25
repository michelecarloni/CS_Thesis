import optuna
import warnings

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