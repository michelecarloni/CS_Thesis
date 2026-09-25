import optuna
import copy
import torch
import torch.nn as nn
import torch.optim as optim

# Scikit-Learn
from tqdm import tqdm


optuna.logging.set_verbosity(optuna.logging.WARNING)




# --- DEEP LEARNING OPTIMIZATION ---
def optimize_cnn_hyperparameters(model, train_loader, val_loader, initial_model_state, n_trials=10, epochs_per_trial=5, use_gpu=True):
    """
    Optuna optimization logic for PyTorch CNNs utilizing Mixed Precision (AMP) and tqdm.
    Exclusively utilizes AdamW as the optimization algorithm.
    """
    
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