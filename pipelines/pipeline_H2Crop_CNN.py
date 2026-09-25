import os
import sys

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

import json
import copy
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from H2Crop.H2CropTileDataset import H2CropTileDataset
from hyperparams_tuning.optimize_cnn_hyperparameters import optimize_cnn_hyperparameters
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader



def pipeline_H2Crop_CNN(model, tiles_dir, save_results_dir, modality, batch_size=32, n_trials=10, epochs_per_trial=5, final_epochs=20, use_gpu=True):
    """
    Trains a pre-instantiated CNN using pre-split train/val/test folders.
    Delegates tuning to an external script using a 20% data subset for speed, 
    then evaluates and saves reports/checkpoints using standard FP32 math and tqdm.
    """
    import random
    from tqdm import tqdm
    
    model_name = getattr(model, 'name', 'Unknown_CNN_Model')
    
    print(f"\n{'='*70}")
    print(f"STARTING CNN PIPELINE FOR: {modality.upper()} | Model: {model_name}")
    print(f"{'='*70}")

    algo_dir = os.path.join(save_results_dir, modality, model_name)
    os.makedirs(algo_dir, exist_ok=True)

    # 1. Load file paths directly from the pre-split subdirectories
    train_dir = os.path.join(tiles_dir, "train")
    val_dir = os.path.join(tiles_dir, "validation")
    test_dir = os.path.join(tiles_dir, "test")

    train_files = glob.glob(os.path.join(train_dir, "*.npz"))
    val_files = glob.glob(os.path.join(val_dir, "*.npz"))
    test_files = glob.glob(os.path.join(test_dir, "*.npz"))

    if not train_files or not val_files or not test_files:
        raise ValueError(f"Missing one or more split folders (train/validation/test) in {tiles_dir}. Run extraction first.")
        
    print(f"Dataset Split Loaded: Train ({len(train_files)}), Val ({len(val_files)}), Test ({len(test_files)})")
    
    # 2. Setup DataLoaders (Optimized with pin_memory=True for faster I/O)
    train_loader = DataLoader(H2CropTileDataset(train_files), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(H2CropTileDataset(val_files), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(H2CropTileDataset(test_files), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Create a smaller 20% subset strictly for Optuna to speed up tuning
    tune_sample_size = max(1, int(len(train_files) * 0.20))
    tune_train_files = random.sample(train_files, tune_sample_size)
    tune_train_loader = DataLoader(H2CropTileDataset(tune_train_files), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"\n[Optimization] Using {tune_sample_size} tiles (20%) for Hyperparameter Tuning.")

    # 4. Save the initial "blank slate" weights of the model
    initial_model_state = copy.deepcopy(model.state_dict())

    # 5. Call external Optuna tuner with the tuned subset
    best_params = optimize_cnn_hyperparameters(
        model=model,
        train_loader=tune_train_loader,
        val_loader=val_loader,
        initial_model_state=initial_model_state,
        n_trials=n_trials,
        epochs_per_trial=epochs_per_trial,
        use_gpu=use_gpu
    )

    # =================================================================
    # 6. Final Retraining on 100% of Data with Best Parameters
    # =================================================================
    print(f"\n--- Retraining on 100% of Data for {final_epochs} Epochs ---")
    model.load_state_dict(copy.deepcopy(initial_model_state))
    
    if best_params["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=best_params["lr"], momentum=0.9, weight_decay=best_params["weight_decay"])
        
    criterion = nn.CrossEntropyLoss()
    
    checkpoint_dir = os.path.join("..", "checkpoints", model_name.lower(), modality)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(final_epochs):
        model.train()
        running_train_loss = 0.0
        
        # TQDM Train Loop
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{final_epochs}] [Train]", leave=False)
        
        for batch_X, batch_y in train_loop:
            if use_gpu and torch.cuda.is_available():
                batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                
            # Input trap to catch bad data
            if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                continue
                
            optimizer.zero_grad()
            
            # AMP DISABLED: Standard 32-bit Forward Pass (Improved with TF32)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Standard 32-bit Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_train_loss += loss.item() * batch_X.size(0)
            train_loop.set_postfix(loss=loss.item())
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        model.eval()
        running_val_loss = 0.0
        
        # TQDM Validation Loop
        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{final_epochs}] [Val]", leave=False)
        
        with torch.no_grad():
            for batch_X, batch_y in val_loop:
                if use_gpu and torch.cuda.is_available():
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                    
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                running_val_loss += loss.item() * batch_X.size(0)
                
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        
        # This prints a clean summary after the tqdm bars disappear
        print(f"    Epoch [{epoch+1}/{final_epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)

    plot_path = os.path.join(checkpoint_dir, "loss_curve.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, final_epochs + 1), train_losses, label='Training Loss', marker='o')
    ax.plot(range(1, final_epochs + 1), val_losses, label='Validation Loss', marker='o')
    ax.set_title(f"Loss Curve: {model_name} ({modality.capitalize()})")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved {final_epochs} checkpoints and loss curve to {checkpoint_dir}")
            
    # =================================================================
    # 7. Final Evaluation on Test Set
    # =================================================================
    print("\n--- Evaluating on Test Set ---")
    model.eval()
    all_preds, all_targets = [], []
    
    # TQDM Test Loop
    test_loop = tqdm(test_loader, desc="Testing", leave=False)
    
    with torch.no_grad():
        for batch_X, batch_y in test_loop:
            if use_gpu and torch.cuda.is_available():
                batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    # =================================================================
    # 8. Save Report and Confusion Matrix
    # =================================================================
    print(f"\n--- Saving Results to {algo_dir} ---")
    
    unique_classes = np.unique(np.concatenate((all_targets, all_preds)))
    target_names = [f"Class {c}" for c in unique_classes]
    
    report_path = os.path.join(algo_dir, "performance.txt")
    report = classification_report(
        all_targets, 
        all_preds, 
        labels=unique_classes, 
        target_names=target_names, 
        zero_division=0
    )
    
    with open(report_path, "w") as f:
        f.write(f"--- Best Optuna Hyperparameters ---\n")
        f.write(json.dumps(best_params, indent=4))
        f.write(f"\n\n--- Test Set Classification Report ---\n")
        f.write(report)
        
    matrix_path = os.path.join(algo_dir, "confusion_matrix.png")
    fig_size = max(10, len(unique_classes) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
    
    ConfusionMatrixDisplay.from_predictions(
        all_targets, 
        all_preds, 
        labels=unique_classes, 
        ax=ax, 
        cmap='Blues', 
        colorbar=False, 
        display_labels=target_names
    )
    
    plt.title(f"Confusion Matrix: {model_name}\n({modality} | Tuned via Optuna)")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    plt.savefig(matrix_path, dpi=300)
    plt.close(fig)
    
    print("Training Complete! All reports saved.")
    
    return model