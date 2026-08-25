import os
import sys

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

from H2Crop.H2Crop import H2Crop
from pipelines import pipeline_H2Crop_unet_optuna
from models.unet import UNet  

if __name__ == "__main__":
    # ==========================================
    # HYPERPARAMETERS & CONFIGURATION
    # ==========================================
    modalities = ["hyperspectral", "multispectral"]
    taxonomy = 3
    patch_sizes = [32]
    save_results_dir = "../results_5"
    
    # New: Define encoders to test
    encoders = ["resnet18", "resnet50"]
    
    use_gpu = True
    n_trials = 2            # 8         
    epochs_per_trial = 5         
    final_epochs = 2       # 20  
    batch_size = 32              
    debug = True                
    
    subsets = {
        1: [8, 11, 23, 56],
        2: [13, 30, 38, 53, 61],
        3: [1, 2, 10, 17],
        4: [29, 50, 64, 76]
    }
    
    print("Initializing H2Crop Loader...")
    loader = H2Crop()

    # ==========================================
    # 1. SUBSET EXTRACTION PHASE
    # ==========================================
    print("\n--- Starting Subset Extraction Phase ---")
    
    for patch_size in patch_sizes:
        print(f"\n{'='*60}")
        print(f"STARTING PROCESSING FOR PATCH SIZE: {patch_size}x{patch_size}")
        print(f"{'='*60}")
        
        for subset_id, subset_classes in subsets.items():
            print(f"\n{'='*50}")
            print(f"PROCESSING SUBSET {subset_id}: {subset_classes} (Patch Size: {patch_size})")
            print(f"{'='*50}")
            
            save_base_dir = f"../ds/H2Crop_tiles_ds_subset_{subset_id}"
            
            for mod in modalities:
                final_save_dir = os.path.join(save_base_dir, f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
                summary_file_path = os.path.join(final_save_dir, "split_summary.txt")
                
                if os.path.exists(final_save_dir) and os.path.exists(summary_file_path):
                    print(f"[*] {mod.upper()} data for Subset {subset_id} (pSize: {patch_size}) already exists. Skipping extraction.")
                else:
                    print(f"[*] No existing splits found for {mod.upper()} Subset {subset_id} (pSize: {patch_size}). Starting extraction...")
                    loader.extract_and_save_tiles_subset(
                        save_base_dir=save_base_dir, 
                        subset_classes=subset_classes, 
                        modality=mod, 
                        taxonomy=taxonomy, 
                        patch_size=patch_size
                    )

    # ==========================================
    # 2. TRAINING PHASE (Deep Learning U-Net)
    # ==========================================
    print("\n--- Starting Deep Learning Phase (U-Net) ---")
    
    # Outer loop: Iterate through encoders as requested
    for encoder in encoders:
        for patch_size in patch_sizes:
            
            # Construct the exact results directory format: e.g., results_5/unet_enc_resnet18_32
            current_results_dir = os.path.join(save_results_dir, f"unet_enc_{encoder}_{patch_size}")
            
            for subset_id, subset_classes in subsets.items():
                save_base_dir = f"../ds/H2Crop_tiles_ds_subset_{subset_id}"
                
                for mod in modalities:
                    dataset_dir = os.path.join(save_base_dir, f"{mod}_taxonomy_{taxonomy}_pSize_{patch_size}")
                    
                    if os.path.exists(dataset_dir):
                        in_channels = 218 if mod == "hyperspectral" else 10
                        num_classes = len(subset_classes) + 1
                        model_name = f"unet_enc_{encoder}"
                        
                        print(f"\nInitializing {model_name} for {mod.upper()} Subset {subset_id}...")
                        
                        model = UNet(
                            in_channels=in_channels, 
                            num_classes=num_classes, 
                            encoder_name=encoder, 
                            encoder_depth=3
                        )
                        
                        pipeline_H2Crop_unet_optuna(
                            model=model,
                            model_name=model_name,
                            save_results_dir=current_results_dir,
                            dataset_dir=dataset_dir,
                            subset_id=subset_id,
                            subset_classes=subset_classes,
                            modality=mod,
                            taxonomy=taxonomy,
                            patch_size=patch_size,
                            use_gpu=use_gpu,
                            n_trials=n_trials,
                            epochs_per_trial=epochs_per_trial,
                            final_epochs=final_epochs,
                            batch_size=batch_size,
                            debug=debug
                        )
                    else:
                        print(f"[Error] Dataset directory {dataset_dir} missing. Skipping training for this config.")