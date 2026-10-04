import os
import sys
import json
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from H2Crop.data_structures import h2crop_taxonomy_dict
from H2Crop.H2CropTileDataset import H2CropTileDataset
from hyperparams_tuning.optimize_unet_hyperparameters import optimize_unet_hyperparameters
from models.unet import UNet  
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import OriginalCombinedLoss

def pipeline_H2Crop_unet_optuna(
    in_channels,
    encoder_name,
    model_name,
    save_results_dir, 
    dataset_dir, 
    subset_id, 
    subset_classes,
    modality, 
    taxonomy=3, 
    patch_size=32, 
    use_gpu=True, 
    n_trials=10,
    epochs_per_trial=5,
    final_epochs=20,
    batch_size=32, 
    debug=False
):
    print(f"\n{'='*70}")
    mode = "DEBUG MODE" if debug else "PRODUCTION MODE (DEEP LEARNING)"
    print(f"STARTING PIPELINE FOR: {model_name.upper()} | {modality.upper()} | Subset {subset_id} | {mode}")
    print(f"{'='*70}")

    results_out_dir = os.path.join(save_results_dir, modality)
    os.makedirs(results_out_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join("..", "checkpoints", model_name, modality, f"subset_{subset_id}_pSize_{patch_size}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\n--- Initializing PyTorch DataLoaders ---")
    train_dataset = H2CropTileDataset(os.path.join(dataset_dir, "train"), subset_classes=subset_classes, debug=debug)
    val_dataset = H2CropTileDataset(os.path.join(dataset_dir, "validation"), subset_classes=subset_classes, debug=debug)
    test_dataset = H2CropTileDataset(os.path.join(dataset_dir, "test"), subset_classes=subset_classes, debug=debug)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_gpu)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_gpu)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_gpu)

    num_classes = len(subset_classes) + 1
    sample_y = [0] + sorted(subset_classes)
    
    taxonomy_key = f'Taxonomy_{taxonomy}'
    target_names = [h2crop_taxonomy_dict.get(taxonomy_key, {}).get(c, f"Class {c}") if c != 0 else "Background (0)" for c in sample_y]

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # OPTUNA HYPERPARAMETER TUNING
    active_trials = 2 if debug else n_trials
    active_epochs = 1 if debug else epochs_per_trial
    
    # Note: alpha_weights removed from the function call
    best_params = optimize_unet_hyperparameters(
        model_name=model_name,
        in_channels=in_channels,
        encoder_name=encoder_name,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        n_trials=active_trials,
        epochs_per_trial=active_epochs,
        use_gpu=use_gpu
    )
    
    with open(os.path.join(results_out_dir, f"best_params_subset_{subset_id}.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    # FINAL PRODUCTION TRAINING
    print(f"\n--- INITIATING FINAL TRAINING: {model_name} (Depth: {best_params['encoder_depth']}) ---")
    
    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_depth=best_params['encoder_depth']
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    
    # Instantiate the original loss (no alpha, evaluates background)
    criterion = OriginalCombinedLoss()
    
    # Initialize the modern AMP Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_gpu and torch.cuda.is_available())
    
    train_epochs = 2 if debug else final_epochs
    history_train_loss = []
    history_test_loss = []
    
    for epoch in range(train_epochs):
        # ==========================================
        # 1. TRAINING PHASE
        # ==========================================
        model.train()
        running_train_loss = 0.0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_epochs} [Train]", leave=False)
        for batch_X, batch_y in train_loop:
            batch_X, batch_y = batch_X.to(device), batch_y.long().to(device)
            
            # SENSOR GUARD & NORMALIZATION
            if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                continue
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
            
            running_train_loss += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")
            
            del batch_X, batch_y, outputs, loss
            
        avg_train_loss = running_train_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)
        
        # ==========================================
        # 2. VALIDATION PHASE
        # ==========================================
        model.eval()
        running_test_loss = 0.0
        
        test_loop = tqdm(test_loader, desc=f"Epoch {epoch+1}/{train_epochs} [Test]", leave=False)
        with torch.no_grad():
            for batch_X, batch_y in test_loop:
                batch_X, batch_y = batch_X.to(device), batch_y.long().to(device)
                
                if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                    continue
                b_mean = batch_X.mean(dim=(2, 3), keepdim=True)
                b_std = batch_X.std(dim=(2, 3), keepdim=True)
                batch_X = (batch_X - b_mean) / (b_std + 1e-5)
                
                with torch.amp.autocast('cuda', enabled=use_gpu):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                running_test_loss += loss.item()
                test_loop.set_postfix(loss=f"{loss.item():.4f}")
                
                del batch_X, batch_y, outputs, loss
                
        avg_test_loss = running_test_loss / len(test_loader)
        history_test_loss.append(avg_test_loss)
            
        print(f"    Epoch {epoch+1}/{train_epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        
        epoch_filepath = os.path.join(checkpoint_dir, f"epoch_{epoch+1:02d}.pth")
        torch.save(model.state_dict(), epoch_filepath)
        
    print(f"    Saved {train_epochs} epoch checkpoints to: {checkpoint_dir}")

    # ==========================================
    # 3. PLOT LEARNING CURVE
    # ==========================================
    print("\n--- GENERATING LEARNING CURVE ---")
    fig_lc, ax_lc = plt.subplots(figsize=(10, 6))
    ax_lc.plot(range(1, train_epochs + 1), history_train_loss, label='Train Combined Loss', marker='o')
    ax_lc.plot(range(1, train_epochs + 1), history_test_loss, label='Test Combined Loss', marker='s')
    ax_lc.set_xlabel('Epoch')
    ax_lc.set_ylabel('Combined Focal+Dice Loss (Lower is Better)')
    ax_lc.set_title(f'Learning Curve: {model_name.upper()}\n({modality} | Subset {subset_id} | pSize {patch_size})')
    ax_lc.legend()
    ax_lc.grid(True, linestyle='--', alpha=0.7)
    
    learning_curve_path = os.path.join(checkpoint_dir, "learning_curve.png")
    plt.tight_layout()
    plt.savefig(learning_curve_path, dpi=300)
    plt.close(fig_lc)
    print(f"    Saved Learning Curve to: {learning_curve_path}")

    # ==========================================
    # 4. TEST EVALUATION
    # ==========================================
    print(f"\n--- EVALUATING ON TEST SET ---")
    model.eval()
    global_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    
    eval_loop = tqdm(test_loader, desc="Final Evaluation", leave=False)
    with torch.no_grad():
        for batch_X, batch_y in eval_loop:
            batch_X, batch_y = batch_X.to(device), batch_y.long().to(device)
            
            if torch.isnan(batch_X).any() or torch.isinf(batch_X).any():
                continue
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
            
            del batch_X, batch_y, outputs, predicted
            
    print("      [Metrics] Calculating performance metrics directly from GPU Confusion Matrix...")
    cm_numpy = global_cm.cpu().numpy()
    
    report_lines = [f"{'':<25} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>15}\n"]
    macro_p, macro_r, macro_f1 = 0.0, 0.0, 0.0
    weighted_p, weighted_r, weighted_f1 = 0.0, 0.0, 0.0
    total_support = np.sum(cm_numpy)
    
    for idx, target_name in enumerate(target_names):
        tp = cm_numpy[idx, idx]
        fp = cm_numpy[:, idx].sum() - tp
        fn = cm_numpy[idx, :].sum() - tp
        support = cm_numpy[idx, :].sum()
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        report_lines.append(f"{target_name:<25} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {support:>15}")
        macro_p += p; macro_r += r; macro_f1 += f1
        weighted_p += p * support; weighted_r += r * support; weighted_f1 += f1 * support
        
    macro_p /= num_classes; macro_r /= num_classes; macro_f1 /= num_classes
    weighted_p /= total_support if total_support > 0 else 1
    weighted_r /= total_support if total_support > 0 else 1
    weighted_f1 /= total_support if total_support > 0 else 1
    accuracy = np.trace(cm_numpy) / total_support if total_support > 0 else 0.0
    
    report_lines.append(f"\n{'accuracy':<25} {'':>10} {'':>10} {accuracy:>10.4f} {total_support:>15}")
    report_lines.append(f"{'macro avg':<25} {macro_p:>10.4f} {macro_r:>10.4f} {macro_f1:>10.4f} {total_support:>15}")
    report_lines.append(f"{'weighted avg':<25} {weighted_p:>10.4f} {weighted_r:>10.4f} {weighted_f1:>10.4f} {total_support:>15}")
    
    with open(os.path.join(results_out_dir, f"performance_subset_{subset_id}_optuna.txt"), "w") as f:
        f.write(f"--- Deep Learning Optimized Inference ({model_name}) ---\n\n" + "\n".join(report_lines))
        
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm_numpy, display_labels=target_names).plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f"Confusion Matrix: {model_name} (Optuna)\n({modality} | Subset {subset_id} | pSize {patch_size})")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_out_dir, f"confusion_matrix_subset_{subset_id}_optuna.png"), dpi=300)
    plt.close(fig)

    del model, global_cm
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\nPipeline completed successfully for {model_name} on {modality.upper()} Subset {subset_id}!")