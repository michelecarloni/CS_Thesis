import os
import sys

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

from H2Crop.H2Crop import H2Crop
from pipelines import pipeline_H2Crop_CNN

# Import the wrapper classes we built earlier
from models.resnet18 import ResNet18
from models.resnet50 import ResNet50

if __name__ == "__main__":
    modality = ["hyperspectral", "multispectral"]
    taxonomy = 3
    save_tile_ds_dir = "../ds/H2Crop_tiles_ds"
    save_results_dir = "../results_4_H2Crop_tiles_resnet" # Added to store the pipeline's output
    
    # Define channel mapping based on your sensor data (Adjust these if needed)
    channels_map = {
        "hyperspectral": 200, # Example: EnMAP bands
        "multispectral": 10   # Example: Sentinel-2 bands
    }
    
    num_classes = 101 # Set to your highest taxonomy class count

    loader = H2Crop()

    # ==========================================
    # 1. TILE EXTRACTION PHASE
    # ==========================================
    print("\n--- Starting Tile Extraction ---")
    for mod in modality:
        # Calls the updated extractor method to slice and save .npz files
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
    for mod in modality:
        # Determine the correct number of channels for the current modality
        current_in_channels = channels_map[mod]
        
        # Instantiate models dynamically so the first Conv2D layer matches the channel count
        resnet18 = ResNet18(in_channels=current_in_channels, num_classes=num_classes, use_gpu=True)
        
        resnet50 = ResNet50(in_channels=current_in_channels, num_classes=num_classes, use_gpu=True)
        
        models = [resnet18, resnet50]
        
        # Target the exact sub-directory created by the extractor function
        tiles_dir = os.path.join(save_tile_ds_dir, f"{mod}_taxonomy_{taxonomy}")
        
        for model in models:
            # Run the pipeline with Optuna tuning and automated result saving
            trained_model = pipeline_H2Crop_CNN(
                model=model,
                tiles_dir=tiles_dir,
                save_results_dir=save_results_dir,
                modality=mod,
                batch_size=32,       # Keep at 32 to protect your 8GB VRAM
                n_trials=10,         # Optuna configuration
                epochs_per_trial=5,
                use_gpu=True
            )