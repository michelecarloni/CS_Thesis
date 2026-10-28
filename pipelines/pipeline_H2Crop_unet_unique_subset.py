import os
import gc
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from H2Crop.data_structures import h2crop_taxonomy_dict
from H2Crop.H2CropTileDataset import H2CropTileDataset
from models.unet import UNet  
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from loss import CombinedLoss

def pipeline_H2Crop_unet_unique(
    in_channels,
    encoder_name,
    model_name,
    save_results_dir, 
    dataset_dir, 
    split_id, 
    subset_classes,
    modality, 
    taxonomy=3, 
    patch_size=64, 
    use_gpu=True, 
    epochs=100,
    batch_size=32,
    encoder_depth=4,
    lr=1e-3,
    weight_decay=1e-4,
    debug=False,
    train_aug_mode="random_aug"
):
    # INPUT VALIDATION: Enforce strict limits on augmentation strategies for the unified dataset
    valid_modes = ["no_aug", "random_aug"]
    if train_aug_mode not in valid_modes:
        raise ValueError(f"Invalid train_aug_mode '{train_aug_mode}'. For the unique subset, it must be exactly one of: {valid_modes}")

    print(f"\n{'='*70}")
    mode = "DEBUG MODE" if debug else "PRODUCTION MODE (CV SPLIT TRAINING)"
    print(f"STARTING PIPELINE FOR: {model_name.upper()} | {modality.upper()} | Split {split_id} | {mode}")
    print(f"Augmentation Mode: {train_aug_mode}")
    print(f"{'='*70}")

    results_out_dir = os.path.join(save_results_dir, modality)
    os.makedirs(results_out_dir, exist_ok=True)
    
    # Checkpointing Trackers (using ../ to target root-level checkpoints folder)
    checkpoint_dir = os.path.join("..", "checkpoints", "unique_subset", model_name, modality, f"split_{split_id}_pSize_{patch_size}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print("\n--- Initializing PyTorch DataLoaders ---")
    # Pass train_aug_mode to train_dataset. Pass empty list for minority classes since we don't have them anymore.
    train_dataset = H2CropTileDataset(
        os.path.join(dataset_dir, "train"), 
        subset_classes=subset_classes, 
        minority_classes=[], 
        debug=debug,
        train_aug_mode=train_aug_mode
    )
    val_dataset = H2CropTileDataset(
        os.path.join(dataset_dir, "validation"), 
        subset_classes=subset_classes, 
        minority_classes=[], 
        debug=debug,
        train_aug_mode="no_aug"
    )
    test_dataset = H2CropTileDataset(
        os.path.join(dataset_dir, "test"), 
        subset_classes=subset_classes, 
        minority_classes=[], 
        debug=debug,
        train_aug_mode="no_aug"
    )
    
    num_workers = 0 if debug else min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_gpu)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=use_gpu)

    num_classes = len(subset_classes) + 1
    sample_y = [0] + sorted(subset_classes)
    
    taxonomy_key = f'Taxonomy_{taxonomy}'
    target_names = [h2crop_taxonomy_dict.get(taxonomy_key, {}).get(c, f"Class {c}") if c != 0 else "Background (0)" for c in sample_y]

    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}")

    # FINAL PRODUCTION TRAINING
    print(f"\n--- INITIATING FINAL TRAINING: {model_name} (Depth: {encoder_depth}) ---")
    
    model = UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_depth=encoder_depth
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    train_epochs = 2 if debug else epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_epochs, eta_min=1e-5)
    
    # Instantiate the loss
    criterion = CombinedLoss(num_classes=num_classes, bg_weight=0.15, gamma=2.0).to(device)
    
    history_train_loss = []
    history_val_loss = []
    history_val_f1_per_class = {name: [] for name in target_names}
    history_val_macro_f1_crops = []
    
    # Files for logging per-epoch validation F1 scores per class
    val_f1_csv_path = os.path.join(results_out_dir, f"val_f1_per_epoch_split_{split_id}.csv")
    val_f1_txt_path = os.path.join(results_out_dir, f"val_f1_per_epoch_split_{split_id}.txt")
    
    with open(val_f1_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Epoch", "Val_Loss", "Crop_Macro_F1"] + target_names)
        
    with open(val_f1_txt_path, mode='w') as txt_file:
        header_cols = " | ".join([f"{name[:15]:>15}" for name in target_names])
        txt_file.write(f"--- Validation F1-Score Per Class Across Epochs ({model_name} | {modality} | Split {split_id}) ---\n\n")
        txt_file.write(f"{'Epoch':>5} | {'Val Loss':>8} | {'Crop mF1':>8} | {header_cols}\n")
        txt_file.write("-" * (32 + 18 * len(target_names)) + "\n")
    
    best_val_macro_f1_crops = -1.0
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    for epoch in range(train_epochs):
        # Resample common tiles & deterministic augmentations for this epoch based on train_aug_mode
        train_dataset.set_epoch(epoch)

        # 1. TRAINING PHASE
        model.train()
        running_train_loss = 0.0
        valid_train_batches = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_epochs} [Train]", leave=False)
        for batch_X, batch_y in train_loop:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.long().to(device, non_blocking=True)
            
            # Sanitize any isolated NaN/Inf values without dropping the batch
            batch_X = torch.nan_to_num(batch_X, nan=0.0, posinf=0.0, neginf=0.0)
            
            optimizer.zero_grad()
            
            # Forward Pass (Global per-band normalization happens inside UNet.input_norm)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
                
            # Backward Pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_train_loss += loss.item()
            valid_train_batches += 1
            train_loop.set_postfix(loss=f"{loss.item():.4f}")
            
            del batch_X, batch_y, outputs, loss
            
        scheduler.step()
        avg_train_loss = running_train_loss / max(valid_train_batches, 1)
        history_train_loss.append(avg_train_loss)
        
        # 2. VALIDATION PHASE (WITH GPU CONFUSION MATRIX FOR F1)
        model.eval()
        running_val_loss = 0.0
        valid_val_batches = 0
        val_global_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{train_epochs} [Validation]", leave=False)
        with torch.no_grad():
            for batch_X, batch_y in val_loop:
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.long().to(device, non_blocking=True)
                
                batch_X = torch.nan_to_num(batch_X, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Forward Pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    running_val_loss += loss.item()
                    valid_val_batches += 1
                
                # Calculate predictions for CM
                _, predicted = torch.max(outputs.data, 1)
                pred_flat = predicted.view(-1)
                true_flat = batch_y.view(-1)
                
                indices = num_classes * true_flat + pred_flat
                batch_cm = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
                val_global_cm += batch_cm
                
                val_loop.set_postfix(loss=f"{loss.item():.4f}")
                del batch_X, batch_y, outputs, loss, predicted
                
        avg_val_loss = running_val_loss / max(valid_val_batches, 1)
        history_val_loss.append(avg_val_loss)
        
        # Calculate Validation F1 for ALL classes and Macro F1 specifically for crops (ignoring Background class 0)
        val_cm_numpy = val_global_cm.cpu().numpy()
        val_macro_f1_crops = 0.0
        epoch_class_f1s = []
        
        for idx, class_name in enumerate(target_names):
            tp = val_cm_numpy[idx, idx]
            fp = val_cm_numpy[:, idx].sum() - tp
            fn = val_cm_numpy[idx, :].sum() - tp
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            
            epoch_class_f1s.append(f1)
            history_val_f1_per_class[class_name].append(f1)
            
            # Index > 0 corresponds to target crop classes
            if idx > 0:
                val_macro_f1_crops += f1
            
        val_macro_f1_crops /= (num_classes - 1) # Average only over crop classes
        history_val_macro_f1_crops.append(val_macro_f1_crops)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Append this epoch's per-class F1 scores to CSV and TXT logs
        with open(val_f1_csv_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch + 1, f"{avg_val_loss:.4f}", f"{val_macro_f1_crops:.4f}"] + [f"{score:.4f}" for score in epoch_class_f1s])
            
        with open(val_f1_txt_path, mode='a') as txt_file:
            f1_cols = " | ".join([f"{score:>15.4f}" for score in epoch_class_f1s])
            txt_file.write(f"{epoch+1:>5d} | {avg_val_loss:>8.4f} | {val_macro_f1_crops:>8.4f} | {f1_cols}\n")
            
        per_class_str = " | ".join([f"{name[:10]}: {score:.3f}" for name, score in zip(target_names, epoch_class_f1s)])
        print(f"    Epoch {epoch+1:02d}/{train_epochs} | LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Crop mF1: {val_macro_f1_crops:.4f}")
        print(f"      -> Val F1 per class: [{per_class_str}]")
        
        # CHECKPOINTING: Save best model based on Crop F1-Score
        if val_macro_f1_crops > best_val_macro_f1_crops:
            best_val_macro_f1_crops = val_macro_f1_crops
            print(f"      [*] Crop F1-Score improved! Saving best model.")
            torch.save(model.state_dict(), best_model_path)
        
        # Save every epoch unconditionally for your records
        epoch_filepath = os.path.join(checkpoint_dir, f"epoch_{epoch+1:02d}.pth")
        torch.save(model.state_dict(), epoch_filepath)
        
    print(f"    Saved {train_epochs} epoch checkpoints to: {checkpoint_dir}")
    print(f"    Saved per-epoch Validation F1 logs to: {val_f1_txt_path} and {val_f1_csv_path}")

    # 3. PLOT LEARNING CURVE & PER-CLASS VALIDATION F1 CURVE
    print("\n--- GENERATING LEARNING CURVES ---")
    fig_lc, ax_lc = plt.subplots(figsize=(10, 6))
    ax_lc.plot(range(1, train_epochs + 1), history_train_loss, label='Train Combined Loss', marker='o')
    ax_lc.plot(range(1, train_epochs + 1), history_val_loss, label='Validation Combined Loss', marker='s')
    ax_lc.set_xlabel('Epoch')
    ax_lc.set_ylabel('Focal + Foreground Dice Loss (Lower is Better)')
    ax_lc.set_title(f'Learning Curve: {model_name.upper()}\n({modality} | Split {split_id} | pSize {patch_size})')
    ax_lc.legend()
    ax_lc.grid(True, linestyle='--', alpha=0.7)
    
    learning_curve_path = os.path.join(checkpoint_dir, "learning_curve.png")
    plt.tight_layout()
    plt.savefig(learning_curve_path, dpi=300)
    plt.close(fig_lc)
    print(f"    Saved Learning Curve to: {learning_curve_path}")
    
    # Plot Per-Class Validation F1 Evolution
    fig_f1, ax_f1 = plt.subplots(figsize=(10, 6))
    for class_name, f1_list in history_val_f1_per_class.items():
        ax_f1.plot(range(1, train_epochs + 1), f1_list, label=class_name, marker='o', markersize=4)
    ax_f1.plot(range(1, train_epochs + 1), history_val_macro_f1_crops, label='Macro Avg (Crops Only)', linestyle='--', linewidth=2, color='black')
    ax_f1.set_xlabel('Epoch')
    ax_f1.set_ylabel('Validation F1-Score')
    ax_f1.set_title(f'Validation F1-Score per Class Across Epochs: {model_name.upper()}\n({modality} | Split {split_id} | pSize {patch_size})')
    ax_f1.legend(loc='best', fontsize=9)
    ax_f1.grid(True, linestyle='--', alpha=0.7)
    
    val_f1_curve_path = os.path.join(results_out_dir, f"val_f1_per_class_split_{split_id}.png")
    plt.tight_layout()
    plt.savefig(val_f1_curve_path, dpi=300)
    plt.close(fig_f1)
    print(f"    Saved Per-Class Validation F1 Curve to: {val_f1_curve_path}")

    # 4. TEST EVALUATION
    print(f"\n--- EVALUATING ON TEST SET ---")
    
    # EXPLICITLY LOAD BEST WEIGHTS BEFORE TESTING
    print(f"      [*] Loading best model weights (Val Crop mF1: {best_val_macro_f1_crops:.4f}) for final test...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    global_cm = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    
    eval_loop = tqdm(test_loader, desc="Final Evaluation", leave=False)
    with torch.no_grad():
        for batch_X, batch_y in eval_loop:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.long().to(device, non_blocking=True)
            
            batch_X = torch.nan_to_num(batch_X, nan=0.0, posinf=0.0, neginf=0.0)
            
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
    
    with open(os.path.join(results_out_dir, f"performance_split_{split_id}.txt"), "w") as f:
        f.write(f"--- Deep Learning Standard Inference ({model_name}) ---\n\n" + "\n".join(report_lines))
        
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(confusion_matrix=cm_numpy, display_labels=target_names).plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title(f"Confusion Matrix: {model_name}\n({modality} | Split {split_id} | pSize {patch_size})")
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(results_out_dir, f"confusion_matrix_split_{split_id}.png"), dpi=300)
    plt.close(fig)

    del model, global_cm
    if use_gpu and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"\nPipeline completed successfully for {model_name} on {modality.upper()} Split {split_id}!")