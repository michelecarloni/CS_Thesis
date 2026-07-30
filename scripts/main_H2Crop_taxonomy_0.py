from pipelines import pipeline_H2Crop_standard_ML_algo
from H2Crop import H2Crop
import os

if __name__ == '__main__':

    # Generate file_sample
    from_train = False
    limit = 300
    save_result_dir = 'results_2_H2Crop/baseline_layer_0_drop_0_3'

    loader = H2Crop()
    file_sample_list = loader.get_file_list(from_train=from_train,
                                            limit=limit)

    print("Saving file_sample_list...")

    os.makedirs(save_result_dir, exist_ok=True)
    with open(os.path.join(save_result_dir, "file_sample_list.txt"), "w") as f:
        for file_name in file_sample_list:
            f.write(file_name + '\n')
    
    print("file sample list succesfully saved")

    modality = ["hyperspectral", "multispectral"]
    for mod in modality:
        pipeline_H2Crop_standard_ML_algo(save_results_dir=save_result_dir, 
                                         file_list=file_sample_list,
                                         modality=mod,
                                         loader=loader,
                                         detail_layer=0,
                                         static=True,
                                         keep_prior=False,
                                         total_samples=150000,
                                         classes_to_drop=[0,3]
                                        )  