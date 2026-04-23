from pipelines import pipeline_standard_ml_algo

dataset_config = {
    'indian_pines': {
        'drop': [0, 1, 7, 9, 16],
        'train_samples': None
    },
    'salinas_valley': {
        'drop': [0],
        'train_samples': None
    },
    'pavia_center': {
        'drop': [0],
        'train_samples': None
    },
    'pavia_university': {
        'drop': [0],
        'train_samples': None
    }
}

if __name__ == '__main__':
    pipeline_standard_ml_algo(dataset_config_dict=dataset_config, save_dir="results_2_undersampling_false")