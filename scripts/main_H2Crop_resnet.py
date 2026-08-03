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
        # Reconstruct the exact path where the tiles are expected to be saved
        target_dir = os.path.join(save_tile_ds_dir, f"{mod}_taxonomy_{taxonomy}")
        
        # Check if the folder exists AND is not empty
        if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
            print(f"[*] Data already extracted for {mod.upper()}. Skipping extraction. (Found at {target_dir})")
            continue
        
        loader.extract_and_save_tiles(
            save_base_dir=save_tile_ds_dir, 
            modality=mod, 
            taxonomy=taxonomy, 
            patch_size=32,
            max_files=100
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
                epochs_per_trial=5,
                use_gpu=True
            )
            
            # CLEAR VRAM BEFORE THE NEXT ITERATION
            print(f"[VRAM Manager] Deleting {model_name} from GPU...")
            del trained_model
            del model
            torch.cuda.empty_cache()