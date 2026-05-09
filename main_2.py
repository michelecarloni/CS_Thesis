from pipelines import pipeline_H2Crop_standard_ML_algo

if __name__ == '__main__':

    pipeline_H2Crop_standard_ML_algo(save_results_dir='results_3_H2Crop/baseline_layer_0_drop_0',
                                     from_train=False,
                                     limit=300,
                                     path=None,
                                     detail_layer=0,
                                     static=True,
                                     keep_prior=False,
                                     total_samples=100000,
                                     classes_to_drop=[0]
                                     )