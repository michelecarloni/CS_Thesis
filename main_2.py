from pipelines import pipeline_H2Crop_standard_ML_algo
from H2Crop.H2Crop import H2Crop

if __name__ == '__main__':

    from_train = False
    limit = 300

    loader = H2Crop()
    file_sample_list = loader.get_file_list(from_train=from_train,
                                            limit=limit)

    modality = ["hyperspectral", "multispectral"]
    for mod in modality:
        pipeline_H2Crop_standard_ML_algo(save_results_dir='results_2_H2Crop/baseline_layer_0_drop_0_3',
                                         from_train=from_train,
                                         limit=limit,
                                         path=None,
                                         detail_layer=0,
                                         static=True,
                                         keep_prior=False,
                                         total_samples=150000,
                                         classes_to_drop=[0,3]
                                        )  