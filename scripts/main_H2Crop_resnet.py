import os
import sys
import torch # Imported to handle VRAM clearing

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

from H2Crop.H2Crop import H2Crop
from pipelines import pipeline_H2Crop_CNN
from models.resnet18 import ResNet18
from models.resnet50 import ResNet50

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if __name__ == "__main__":
    modality = ["hyperspectral", "multispectral"]
    taxonomy = 3
    save_tile_ds_dir = "../ds/H2Crop_tiles_ds"
    save_results_dir = "../results"
    
    # EXACT channel mapping based on the PyTorch error
    channels_map = {
        "hyperspectral": 218, # Updated to 218 bands!
        "multispectral": 10   # Double check this is correct for your Sentinel-2 data
    }
    
    num_classes = 101 # Set to your highest taxonomy class count

    loader = H2Crop()

    # ==========================================
    # 1. TILE EXTRACTION PHASE
    # ==========================================
    print("\n--- Starting Tile Extraction ---")
    for mod in modality:
        target_dir = os.path.join(save_tile_ds_dir, f"{mod}_taxonomy_{taxonomy}")
        train_subfolder = os.path.join(target_dir, "train")
        
        # Check if the 'train' subfolder exists and has files inside it
        if os.path.exists(train_subfolder) and len(os.listdir(train_subfolder)) > 0:
            print(f"[*] Pre-split data already exists for {mod.upper()}. Skipping extraction. (Found at {target_dir})")
            continue
            
        print(f"[*] No existing splits found for {mod.upper()}. Starting extraction & splitting...")
        
        loader.extract_and_save_tiles(
            save_base_dir=save_tile_ds_dir, 
            modality=mod, 
            taxonomy=taxonomy, 
            patch_size=32,
        )

    # ==========================================
    # 2. TRAINING & TUNING PHASE
    # ==========================================
    print("\n--- Starting CNN Pipelines ---")
    
    # Define which models to build inside the loop
    model_blueprints = [
        ("ResNet-18", ResNet18),
        ("ResNet-50", ResNet50)
    ]
    
    for mod in modality:
        current_in_channels = channels_map[mod]
        tiles_dir = os.path.join(save_tile_ds_dir, f"{mod}_taxonomy_{taxonomy}")
        
        for model_name, ModelClass in model_blueprints:
            print(f"\n[VRAM Manager] Building {model_name} for {mod} (Channels: {current_in_channels})")
            
            # Instantiate ONLY the current model
            model = ModelClass(in_channels=current_in_channels, num_classes=num_classes, use_gpu=True)
            model.name = model_name
            
            # Run the pipeline
            trained_model = pipeline_H2Crop_CNN(
                model=model,
                tiles_dir=tiles_dir,
                save_results_dir=save_results_dir,
                modality=mod,
                batch_size=32,       
                n_trials=10,         
                epochs_per_trial=1,
                final_epochs=20,
                use_gpu=True
            )
            
            # CLEAR VRAM BEFORE THE NEXT ITERATION
            print(f"[VRAM Manager] Deleting {model_name} from GPU...")
            del trained_model
            del model
            torch.cuda.empty_cache()