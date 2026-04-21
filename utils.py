import scipy.io as sio
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from data_structures import *
import matplotlib.pyplot as plt


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
    
    if dataset_name not in DATASET_PATHS:
        raise ValueError(f"Dataset '{dataset_name}' not found. Choose from {list(DATASET_PATHS.keys())}")
        
    data_dict = sio.loadmat(DATASET_PATHS[dataset_name]['data'])
    gt_dict = sio.loadmat(DATASET_PATHS[dataset_name]['gt'])
    
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
        
    return X_normalized









def get_wavelengths(dataset_name, n_bands):
    """
    Approximates the true physical wavelengths (in nanometers) for the dataset's bands.
    Accounts for the specific water absorption bands removed from AVIRIS datasets.
    """
    if dataset_name in ['indian_pines', 'salinas_valley']:
        # AVIRIS Sensor: ~400 nm to 2500 nm originally over 224 bands
        full_wavelengths = np.linspace(400, 2500, 224)
        
        if dataset_name == 'indian_pines' and n_bands == 200:
            # Indian Pines removed bands: [104-108], [150-163], [220-224] (1-indexed)
            drop_indices = list(range(103, 108)) + list(range(149, 163)) + list(range(219, 224))
            return np.delete(full_wavelengths, drop_indices)
            
        elif dataset_name == 'salinas_valley' and n_bands == 204:
            # Salinas removed bands: [108-112], [154-167], 224 (1-indexed)
            drop_indices = list(range(107, 112)) + list(range(153, 167)) + [223]
            return np.delete(full_wavelengths, drop_indices)
            
        else:
            # Fallback if you have a slightly different AVIRIS cut
            return np.linspace(400, 2500, n_bands)
            
    elif dataset_name in ['pavia_university', 'pavia_center']:
        # ROSIS Sensor: ~430 nm to 860 nm
        return np.linspace(430, 860, n_bands)
        
    else:
        # Generic fallback
        return np.arange(1, n_bands + 1)


def plot_spectral_signatures(dataset_name, classes_to_plot=None, background=False):
    """
    Plots the mean spectral signature against actual Wavelengths (nm).

    background: True -> background included. False -> Background removed
    """
    if dataset_name not in DATASET_CLASS_NAMES:
        raise ValueError(f"Dataset '{dataset_name}' mappings not found.")
        
    class_mapping = DATASET_CLASS_NAMES[dataset_name]
    
    # Load data
    classes_to_drop = [] if background else [0]
    X, y = load_hyperspectral_dataset(dataset_name, classes_to_drop=classes_to_drop)
    
    # Determine classes
    if classes_to_plot is None:
        classes_to_plot = np.unique(y).tolist()
    else:
        classes_to_plot = list(classes_to_plot)
        if not background and 0 in classes_to_plot:
            classes_to_plot.remove(0)

    # Get actual physical wavelengths for the x-axis
    n_bands = X.shape[1]
    wavelengths = get_wavelengths(dataset_name, n_bands)
    
    plt.figure(figsize=(12, 6))
    
    for cls_id in classes_to_plot:
        X_cls = X[y == cls_id]
        if len(X_cls) == 0:
            continue
            
        cls_name = class_mapping.get(cls_id, f"Class {cls_id}")
        mean_signature = np.mean(X_cls, axis=0)
        std_signature = np.std(X_cls, axis=0)
        
        # Plot using wavelengths
        line = plt.plot(wavelengths, mean_signature, label=f"{cls_name} (n={len(X_cls)})", linewidth=2)[0]
        
        plt.fill_between(wavelengths, 
                         mean_signature - std_signature, 
                         mean_signature + std_signature, 
                         color=line.get_color(), alpha=0.2)

    plt.title(f"Raw Mean Spectral Signatures - {dataset_name.replace('_', ' ').title()}", fontsize=14, pad=15)
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Raw Reflectance Intensity (16-bit)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()