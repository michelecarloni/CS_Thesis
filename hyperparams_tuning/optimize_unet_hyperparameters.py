import optuna
import warnings
import copy
import torch
import torch.nn as nn
import torch.optim as optim

# Scikit-Learn
from loss import combined_loss
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)



def optimize_unet_hyperparameters(model, train_loader, val_loader, initial_model_state, num_classes, n_trials=10, epochs_per_trial=5, use_gpu=True):
    """
    Optuna optimization logic specifically designed for U-Net Semantic Segmentation.
    Evaluates on Pixel-wise Macro F1-Score using a RAM-safe GPU confusion matrix.
    Now utilizes the custom Combined Focal + Dice Loss to handle background dominance.
    """
    
    def objective(trial):
        # Reset model to its initial random weights before every trial
        model.load_state_dict(copy.deepcopy(initial_model_state))
        
        # Hyperparameters to tune
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        
        # Strictly enforced AdamW optimizer
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        for epoch in range(epochs_per_trial):
            # TRAINING PHASE
            model.train()
            train_loop = tqdm(train_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{epochs_per_trial} [Train]", leave=False)
            
            for batch_X, batch_y in train_loop:
                if use_gpu and torch.cuda.is_available():
                    batch_X = batch_X.cuda()
                    batch_y = batch_y.long().cuda()
                    
                if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                    continue

                optimizer.zero_grad()
                
                # Forward pass: Output shape is (Batch, num_classes, 32, 32)
                outputs = model(batch_X)
                
                # Calculate the loss using your combined Focal + Dice function
                loss = combined_loss(outputs, batch_y)
                
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loop.set_postfix(loss=loss.item())
                
            # VALIDATION PHASE (RAM-SAFE MACRO F1)
            model.eval()
            
            # Initialize a 0-MB Confusion Matrix directly on the GPU
            device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            global_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
            
            val_loop = tqdm(val_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/{epochs_per_trial} [Val]", leave=False)
            
            with torch.no_grad():
                for batch_X, batch_y in val_loop:
                    if use_gpu and torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.long().cuda()
                        
                    outputs = model(batch_X)
                    
                    # Get the most likely class per pixel. Output shape: (Batch, 32, 32)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # Flatten predictions and targets to 1D arrays
                    pred_flat = predicted.view(-1)
                    true_flat = batch_y.view(-1)
                    
                    # Fast PyTorch bincount to build the confusion matrix instantly on GPU
                    indices = num_classes * true_flat + pred_flat
                    batch_cm = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
                    global_cm += batch_cm
                    
            # Move matrix back to CPU to calculate Macro F1 safely
            cm = global_cm.cpu().numpy()
            macro_f1 = 0.0
            
            for i in range(num_classes):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                macro_f1 += f1
                
            macro_f1 /= num_classes
            
            # Report to Optuna to guide the next trial
            trial.report(macro_f1, epoch)
            
            # Prune unpromising trials early to save time
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        return macro_f1

    print(f"\n--- Running Optuna Tuning for U-Net ({n_trials} Trials) ---")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n[Optuna] Best Trial: {study.best_trial.number}")
    print(f"[Optuna] Best Validation Macro F1: {study.best_value:.4f}")
    
    return study.best_params