import scipy.io as sio
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def load_hyperspectral_dataset(dataset_name, classes_to_drop=None):
    """
    Loads a hyperspectral dataset, flattens it for ML models, and drops specified classes.
    
    Args:
        dataset_name (str): 'indian_pines', 'salinas_valley', 'pavia_center', or 'pavia_university'.
        classes_to_drop (list): List of class integers to remove (e.g., [0] to drop background).
        
    Returns:
        X (np.ndarray): Flattened feature matrix of shape (n_pixels, n_bands).
        y (np.ndarray): Flattened label array of shape (n_pixels,).
    """
    
    # dataset paths
    paths = {
        'indian_pines': {
            'data': 'ds/indian_pines/indian_pines_corrected.mat',
            'gt': 'ds/indian_pines/indian_pines_gt.mat'
        },
        'salinas_valley': {
            'data': 'ds/salinas_valley/salinas_valley_corrected.mat',
            'gt': 'ds/salinas_valley/salinas_valley_gt.mat'
        },
        'pavia_center': {
            'data': 'ds/pavia_center/pavia_center.mat',
            'gt': 'ds/pavia_center/pavia_center_gt.mat'
        },
        'pavia_university': {
            'data': 'ds/pavia_university/pavia_university.mat',
            'gt': 'ds/pavia_university/pavia_university_gt.mat'
        }
    }
    
    if dataset_name not in paths:
        raise ValueError(f"Dataset '{dataset_name}' not found. Choose from {list(paths.keys())}")
        
    data_dict = sio.loadmat(paths[dataset_name]['data'])
    gt_dict = sio.loadmat(paths[dataset_name]['gt'])
    
    # Helper to dynamically extract the actual data array, ignoring meta-keys
    def extract_array(mat_dict):
        for key in mat_dict:
            if not key.startswith('__'):
                return mat_dict[key]
        raise KeyError("Could not find the data array in the .mat file.")

    data_cube = extract_array(data_dict)
    ground_truth = extract_array(gt_dict)
    
    # Flatten the spatial dimensions: (H, W, Bands) -> (H * W, Bands)
    n_bands = data_cube.shape[2]
    X = data_cube.reshape(-1, n_bands)
    y = ground_truth.reshape(-1)
    
    # Drop specified classes
    if classes_to_drop is not None:
        # Create a boolean mask of pixels that are NOT in the drop list
        mask = ~np.isin(y, classes_to_drop)
        X = X[mask]
        y = y[mask]
        
    return X, y







def print_class_distribution(dataset_name, y):

    """
    Counts and prints the number of samples for each class in the dataset.
    
    Args:
        dataset_name (str): Name of the dataset to map numeric labels to strings.
        y (np.ndarray): Flattened array of labels.
    """

    # Standard class mappings for the 4 hyperspectral datasets
    DATASET_CLASS_NAMES = {
        'indian_pines': {
            0: 'Background', 1: 'Alfalfa', 2: 'Corn-notill', 3: 'Corn-mintill', 
            4: 'Corn', 5: 'Grass-pasture', 6: 'Grass-trees', 7: 'Grass-pasture-mowed',
            8: 'Hay-windrowed', 9: 'Oats', 10: 'Soybean-notill', 11: 'Soybean-mintill',
            12: 'Soybean-clean', 13: 'Wheat', 14: 'Woods', 15: 'Buildings-Grass-Trees-Drives',
            16: 'Stone-Steel-Towers'
        },
        'salinas_valley': {
            0: 'Background', 1: 'Brocoli_green_weeds_1', 2: 'Brocoli_green_weeds_2', 
            3: 'Fallow', 4: 'Fallow_rough_plow', 5: 'Fallow_smooth', 6: 'Stubble', 
            7: 'Celery', 8: 'Grapes_untrained', 9: 'Soil_vinyard_develop', 
            10: 'Corn_senesced_green_weeds', 11: 'Lettuce_romaine_4wk', 
            12: 'Lettuce_romaine_5wk', 13: 'Lettuce_romaine_6wk', 14: 'Lettuce_romaine_7wk', 
            15: 'Vinyard_untrained', 16: 'Vinyard_vertical_trellis'
        },
        'pavia_university': {
            0: 'Background', 1: 'Asphalt', 2: 'Meadows', 3: 'Gravel', 4: 'Trees',
            5: 'Painted metal sheets', 6: 'Bare Soil', 7: 'Bitumen',
            8: 'Self-Blocking Bricks', 9: 'Shadows'
        },
        'pavia_center': {
            0: 'Background', 1: 'Water', 2: 'Trees', 3: 'Asphalt', 
            4: 'Self-Blocking Bricks', 5: 'Bitumen', 6: 'Tiles', 7: 'Shadows', 
            8: 'Meadows', 9: 'Bare Soil'
        }
    }

    if dataset_name not in DATASET_CLASS_NAMES:
        raise ValueError(f"Dataset '{dataset_name}' mappings not found.")
        
    class_mapping = DATASET_CLASS_NAMES[dataset_name]
    
    # Get unique classes and their corresponding counts
    unique_classes, counts = np.unique(y, return_counts=True)
    
    print(f"\n{'='*50}")
    print(f"CLASS DISTRIBUTION: {dataset_name.upper()}")
    print(f"{'='*50}")
    print(f"{'ID':<4} | {'Class Name':<30} | {'Samples':<10}")
    print("-" * 50)
    
    total_samples = 0
    
    for cls_id, count in zip(unique_classes, counts):
        # Fallback just in case there's an unexpected class ID
        cls_name = class_mapping.get(cls_id, f"Unknown_Class_{cls_id}") 
        print(f"{cls_id:<4} | {cls_name:<30} | {count:<10}")
        total_samples += count
        
    print("-" * 50)
    print(f"{'TOTAL':<37} {total_samples}")
    print(f"{'='*50}\n")





def normalize_features(X):
    """
    Normalizes the dataset across each feature (band) independently 
    using Min-Max scaling to bring all values into the [0, 1] range.
    
    Args:
        X (np.ndarray): Flattened feature matrix of shape (n_pixels, n_bands).
                      
    Returns:
        X_normalized (np.ndarray): The normalized dataset.
        scaler (MinMaxScaler): The fitted scikit-learn scaler object. 
                               Keep this for Domain Adaptation later!
    """
    scaler = MinMaxScaler()
    
    # fit_transform calculates the min/max for each band and applies the scaling
    X_normalized = scaler.fit_transform(X)
    
    # Quick sanity check print
    print("Dataset normalized using Min-Max scaling (Feature-wise).")
    print(f"Global Min: {np.min(X_normalized):.2f}, Global Max: {np.max(X_normalized):.2f}")
        
    return X_normalized, scaler