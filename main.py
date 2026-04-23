from pipelines import pipeline_standard_ml_algo

dataset_config = {
    'indian_pines': {
        'drop': [0, 1, 7, 9, 16],
        'train_samples': 200
    },
    'salinas_valley': {
        'drop': [0],
        'train_samples': 800
    },
    'pavia_center': {
        'drop': [0],
        'train_samples': 800
    },
    'pavia_university': {
        'drop': [0],
        'train_samples': 800
    }
}

if __name__ == '__main__':
    pipeline_standard_ml_algo(dataset_config_dict=dataset_config)