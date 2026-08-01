import os
import sys

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

from H2Crop.H2Crop import H2Crop

# Import your newly updated pipeline
from pipelines import pipeline_H2Crop_standard_ML_algo

if __name__ == "__main__":

    loader = H2Crop()

    # Safely create root results directory
    os.makedirs("../results_3_H2Crop_similar_classes", exist_ok=True)

    # Define subsets
    sub_1 = [8, 23, 11, 56]
    sub_2 = [13, 30, 38, 53, 61]
    sub_3 = [1, 2, 10, 17]
    sub_4 = [29, 50, 64, 76]

    # Define paths
    sub_1_path = "../results_3_H2Crop_similar_classes/sub_1"
    sub_2_path = "../results_3_H2Crop_similar_classes/sub_2"
    sub_3_path = "../results_3_H2Crop_similar_classes/sub_3"
    sub_4_path = "../results_3_H2Crop_similar_classes/sub_4"

    # Pair paths with their respective class lists
    experiments = [
        (sub_3_path, sub_3),
        (sub_2_path, sub_2),
        (sub_1_path, sub_1),
        (sub_4_path, sub_4)
    ]

    modality = ["hyperspectral", "multispectral"]

    # Generate the chunked list ONCE for the entire script
    chunked_file_list = loader.get_chunked_file_list(300)

    # Execute pipeline
    for save_dir, class_list in experiments:
        
        # Ensure the specific subset directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        for mod in modality:
            print(f"\n{'*'*60}")
            print(f"*** RUNNING EXPERIMENT: {save_dir.split('/')[-1]} | {mod.upper()} ***")
            print(f"{'*'*60}")
            
            # 1. Extract (or load) the cached pixels
            data_path = loader.extract_H2Crop_pixel_sample(
                save_dir=save_dir,
                file_chunks_list=chunked_file_list,
                modality=mod,
                class_action="keep",
                class_list=class_list,
                detail_layer=3,
                static=True,
                keep_prior=False,
                samples_per_class=50000
            )
            
            # Check if extraction was successful
            if not data_path:
                print(f"Skipping {mod} for {save_dir} due to extraction failure.")
                continue

            # 2. Run the ML pipeline using the cached .npz file
            pipeline_H2Crop_standard_ML_algo(
                save_results_dir=save_dir, 
                data_path=data_path,       
                modality=mod,
                detail_layer=3,            
            )