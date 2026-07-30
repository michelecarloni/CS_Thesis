import optuna
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

# Silence verbose Optuna logging (optional, keeps console output clean)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_hyperparameters(model_name, X_train, y_train, X_val, y_val, n_trials=20, random_state=42):
    """
    Unified Optuna tuner for standard ML models.
    Returns the best fitted model trained on X_train.
    """
    
    def objective(trial):
        # --- 1. Decision Tree ---
        if model_name == "decision_tree":
            params = {
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': random_state
            }
            model = DecisionTreeClassifier(**params)

        # --- 2. Random Forest ---
        elif model_name == "random_forest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 250, step=50),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': random_state,
                'n_jobs': -1
            }
            model = RandomForestClassifier(**params)

        # --- 3. Linear SVC ---
        elif model_name == "linear_svm":
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
                'dual': False,
                'max_iter': 2000,
                'random_state': random_state
            }
            model = LinearSVC(**params)

        # --- 4. Logistic Regression ---
        elif model_name == "logistic_regression":
            params = {
                'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
                'solver': 'saga',
                'max_iter': 2000,
                'random_state': random_state,
                'n_jobs': -1
            }
            model = LogisticRegression(**params)

        else:
            raise ValueError(f"Unknown model_name: '{model_name}'")

        # Fit on Training Set & Evaluate on Validation Set
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
        return RandomForestClassifier(**params, random_state=random_state, n_jobs=-1)
    elif model_name == "linear_svm":
        return LinearSVC(**params, dual=False, max_iter=2000, random_state=random_state)
    elif model_name == "logistic_regression":
        return LogisticRegression(**params, solver='saga', max_iter=2000, random_state=random_state, n_jobs=-1)