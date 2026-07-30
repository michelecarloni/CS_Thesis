import os
import sys

# Setup project root path
project_root = os.path.abspath('..')
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import read_file_sample
from H2Crop.H2Crop import H2Crop

# Import your newly updated pipeline
from pipelines import pipeline_H2Crop_standard_ML_algo

if __name__ == "__main__":
    
    file_sample_path = '../H2Crop/file_sample_list.txt'
    file_list = read_file_sample(file_sample_path)

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

    # Execute pipeline
    for save_dir, class_list in experiments:
        
        # Ensure the specific subset directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        for mod in modality:
            print(f"\n{'*'*60}")
            print(f"*** RUNNING EXPERIMENT: {save_dir.split('/')[-1]} | {mod.upper()} ***")
            print(f"{'*'*60}")
            
            pipeline_H2Crop_standard_ML_algo(
                save_results_dir=save_dir, 
                file_list=file_list,
                modality=mod,
                class_action="keep",       
                class_list=class_list,     
                loader=loader,
                detail_layer=3,            
                static=True,
                samples_per_class=50000,
                allow_resample=True     
            )