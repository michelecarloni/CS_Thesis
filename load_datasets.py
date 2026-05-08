import os
import h5py
import numpy as np


def load_H2_Crop(data_dir, sample_id):
    """
    Loads the EnMAP, Sentinel-2, Label, and Prior data for a given sample.
    
    Note: You may need to adjust the file paths and the exact internal HDF5 
    keys ('hyperspectral', 'multispectral', etc.) based on what the 
    inspect_h5_structure function reveals.
    """
    
    data_path = os.path.join(data_dir, 'h5_data', f'{sample_id}.h5')
    
    dataset_dict = {}
    
    try:
        with h5py.File(data_path, 'r') as f:
            dataset_dict['hyperspectral'] = np.array(f['EnMAP_data']) 
            dataset_dict['multispectral'] = np.array(f['S2_data'])
            dataset_dict['labels']        = np.array(f['label'])
            dataset_dict['prior']         = np.array(f['priors'])
            
        print(f"Successfully loaded sample: {sample_id}")
        
    except FileNotFoundError as e:
        print(f"File missing for sample {sample_id}: {e}")
    except KeyError as e:
        print(f"Internal HDF5 key not found: {e}")
        
    return dataset_dict