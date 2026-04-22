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









def get_sensor_specs(dataset_name, n_bands):
    """
    Returns the full wavelength array and the indices of the valid bands,
    along with the specific ranges to shade red (water absorption).
    """
    if dataset_name in ['indian_pines', 'salinas_valley']:
        # AVIRIS Sensor: 224 continuous bands from 400nm to 2500nm
        full_wavelengths = np.linspace(400, 2500, 224)
        
        if dataset_name == 'indian_pines' and n_bands == 200:
            # 0-indexed dropped bands
            drop_ranges = [(103, 108), (149, 163), (219, 224)] 
        elif dataset_name == 'salinas_valley' and n_bands == 204:
            drop_ranges = [(107, 112), (153, 167), (223, 224)]
        else:
            return full_wavelengths, np.arange(n_bands), []
            
        # Build a list of valid indices by excluding the drop ranges
        valid_indices = []
        for i in range(224):
            if not any(start <= i < end for start, end in drop_ranges):
                valid_indices.append(i)
                
        # Calculate the actual nanometer ranges to draw the red boxes
        red_spans = [(full_wavelengths[start], full_wavelengths[end-1]) for start, end in drop_ranges]
        
        return full_wavelengths, valid_indices, red_spans
        
    elif dataset_name in ['pavia_university', 'pavia_center']:
        # ROSIS Sensor: No dropped bands, continuous ~430 to 860nm
        full_wavelengths = np.linspace(430, 860, n_bands)
        return full_wavelengths, np.arange(n_bands), []

def plot_spectral_signatures(dataset_name, classes_to_plot=None, background=False):
    """
    Plots the mean spectral signature with true physical gaps and red shaded
    regions for missing water absorption bands.
    """
    if dataset_name not in DATASET_CLASS_NAMES:
        raise ValueError(f"Dataset '{dataset_name}' mappings not found.")
        
    class_mapping = DATASET_CLASS_NAMES[dataset_name]
    
    # Load data
    classes_to_drop = [] if background else [0]
    X, y = load_hyperspectral_dataset(dataset_name, classes_to_drop=classes_to_drop)
    
    if classes_to_plot is None:
        classes_to_plot = np.unique(y).tolist()
    else:
        classes_to_plot = list(classes_to_plot)
        if not background and 0 in classes_to_plot:
            classes_to_plot.remove(0)

    # Get true physical specs
    n_bands = X.shape[1]
    full_wavelengths, valid_indices, red_spans = get_sensor_specs(dataset_name, n_bands)
    
    plt.figure(figsize=(12, 6))
    
    # Draw the red "No Data" regions first so they sit in the background
    for start_nm, end_nm in red_spans:
        plt.axvspan(start_nm, end_nm, color='red', alpha=0.15, label='Water Absorption (Removed)')
    
    # Fix duplicate labels for the red spans in the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    for cls_id in classes_to_plot:
        X_cls = X[y == cls_id]
        if len(X_cls) == 0:
            continue
            
        cls_name = class_mapping.get(cls_id, f"Class {cls_id}")
        
        # Calculate stats for the valid 200 bands
        mean_valid = np.mean(X_cls, axis=0)
        std_valid = np.std(X_cls, axis=0)
        
        # Create empty arrays of size 224 filled with NaN
        mean_full = np.full(len(full_wavelengths), np.nan)
        std_full = np.full(len(full_wavelengths), np.nan)
        
        # Insert the valid data into the correct physical slots
        mean_full[valid_indices] = mean_valid
        std_full[valid_indices] = std_valid
        
        # Plot the data (Matplotlib automatically breaks the line at NaNs!)
        line = plt.plot(full_wavelengths, mean_full, label=f"{cls_name} (n={len(X_cls)})", linewidth=2)[0]
        
        plt.fill_between(full_wavelengths, 
                         mean_full - std_full, 
                         mean_full + std_full, 
                         color=line.get_color(), alpha=0.2)
                         
        by_label[f"{cls_name} (n={len(X_cls)})"] = line

    # Formatting
    plt.title(f"Raw Mean Spectral Signatures - {dataset_name.replace('_', ' ').title()}", fontsize=14, pad=15)
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Raw Reflectance Intensity (16-bit)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Draw legend using the deduplicated dictionary
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()