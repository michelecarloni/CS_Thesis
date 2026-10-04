import optuna
import warnings
import copy
import torch
import torch.nn as nn
import torch.optim as optim

from models.unet import UNet 
from loss import OriginalCombinedLoss
from tqdm import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_unet_hyperparameters(model_name, in_channels, encoder_name, train_loader, val_loader, num_classes, n_trials=10, epochs_per_trial=5, use_gpu=True):
    """
    Optuna optimization logic specifically designed for U-Net Semantic Segmentation.
    Instantiates the OriginalCombinedLoss dynamically (evaluates all pixels, including background).
    """
    
    def objective(trial):
        suggested_depth = trial.suggest_int("encoder_depth", 2, 4)
        
        model = UNet(
            in_channels=in_channels,
            num_classes=num_classes,
            encoder_name=encoder_name,
            encoder_depth=suggested_depth
        )
        
        device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Instantiate the original loss (no alpha weights, no ignore index)
        criterion = OriginalCombinedLoss()
        
        # Initialize the modern AMP Scaler for Optuna trials
        scaler = torch.amp.GradScaler('cuda', enabled=use_gpu and torch.cuda.is_available())
        
        for epoch in range(epochs_per_trial):
            # TRAINING PHASE
            model.train()
            train_loop = tqdm(train_loader, desc=f"Trial {trial.number} | Depth {suggested_depth} | Epoch {epoch+1}/{epochs_per_trial} [Train]", leave=False)
            
            for batch_X, batch_y in train_loop:
                if use_gpu and torch.cuda.is_available():
                    batch_X = batch_X.cuda()
                    batch_y = batch_y.long().cuda()
                    
                # SENSOR GUARD: Skip corrupted EnMAP data
                if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                    continue

                # DYNAMIC NORMALIZATION to prevent FP16 overflow
                b_mean = batch_X.mean(dim=(2, 3), keepdim=True)
                b_std = batch_X.std(dim=(2, 3), keepdim=True)
                batch_X = (batch_X - b_mean) / (b_std + 1e-5)

                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda', enabled=use_gpu):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_loop.set_postfix(loss=loss.item())
                
            # VALIDATION PHASE
            model.eval()
            global_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
            val_loop = tqdm(val_loader, desc=f"Trial {trial.number} | Depth {suggested_depth} | Epoch {epoch+1}/{epochs_per_trial} [Val]", leave=False)
            
            with torch.no_grad():
                for batch_X, batch_y in val_loop:
                    if use_gpu and torch.cuda.is_available():
                        batch_X, batch_y = batch_X.cuda(), batch_y.long().cuda()
                    
                    if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                        continue
                        
                    # DYNAMIC NORMALIZATION
                    b_mean = batch_X.mean(dim=(2, 3), keepdim=True)
                    b_std = batch_X.std(dim=(2, 3), keepdim=True)
                    batch_X = (batch_X - b_mean) / (b_std + 1e-5)

                    with torch.amp.autocast('cuda', enabled=use_gpu):
                        outputs = model(batch_X)
                        
                    _, predicted = torch.max(outputs.data, 1)
                    
                    pred_flat = predicted.view(-1)
                    true_flat = batch_y.view(-1)
                    
                    indices = num_classes * true_flat + pred_flat
                    batch_cm = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
                    global_cm += batch_cm
                    
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
            
            trial.report(macro_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        # Clean up memory after trial
        del model, optimizer
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return macro_f1

    print(f"\n--- Running Optuna Tuning for U-Net ({n_trials} Trials) ---")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\n[Optuna] Best Trial: {study.best_trial.number}")
    print(f"[Optuna] Best Validation Macro F1: {study.best_value:.4f}")
    
    return study.best_params