import os
import h5py
import numpy as np
import random

class H2Crop:
    def __init__(self, dataset_path='/media/michele/T7/datasets/H2Crop'):
        """
        Initializes the H2Crop dataset loader.
        """
        self.dataset_path = dataset_path
        self.h5_dir = os.path.join(self.dataset_path, 'h5_data')
        
        # Verify the directory exists upon initialization
        if not os.path.exists(self.h5_dir):
            print(f"Warning: Directory not found at {self.h5_dir}")


    def get_file_list(self, from_train, limit, path=None):
        """
        Generates a fixed list of random sample IDs to ensure consistency 
        across multiple data loading runs.
        
        Parameters:
        - from_train (bool): If True, reads from the train.txt file. If False, reads directory contents.
        - path (str): The full path to either the train.txt file or the h5_data directory.
        - limit (int): The maximum number of filenames to sample.
        
        Returns:
        - list of strings: The randomly sampled filenames (without the .h5 extension).
        """
        file_names = []
        
        if from_train:
            # Read from the training text file
            if not os.path.isfile(path):
                print(f"Error: Training list file not found at {path}")
                return []
                
            with open(path, 'r') as f:
                file_names = [line.strip().replace('.h5', '') for line in f.readlines() if line.strip()]
        else:
            # Read directly from the h5_data directory
            if path is None:
                path = self.h5_dir
            if not os.path.isdir(path):
                print(f"Error: Directory not found at {path}")
                return []
                
            file_names = [f.replace('.h5', '') for f in os.listdir(path) if f.endswith('.h5')]
            
        if not file_names:
            print(f"Error: No valid files/entries found at {path}.")
            return []
            
        # Safely cap the limit if the user asks for more files than exist
        actual_limit = min(limit, len(file_names))
        
        # Randomly sampling without dusplicates
        sampled_ids = random.sample(file_names, actual_limit)
        
        print(f"Successfully sampled {len(sampled_ids)} file IDs.")
        return sampled_ids


    def _load_single_file(self, filename, detail_layer=None, static=False):
        """
        Internal helper method to load a single .h5 file and apply filtering logic.
        """
        file_path = os.path.join(self.h5_dir, filename)
        
        dataset_dict = {'sample_id': filename.replace('.h5', '')}
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Get Hyperspectral
                dataset_dict['hyperspectral'] = np.array(f['EnMAP_data'])
                
                # Get Multispectral (+ process for filtering temporal dimension if necessary)
                s2_full = np.array(f['S2_data'])
                if static:
                    month_str = filename[4:6]
                    s2_time_index = int(month_str) - 1
                    dataset_dict['multispectral'] = s2_full[s2_time_index] 
                else:
                    dataset_dict['multispectral'] = s2_full
                
                # Get labels
                labels_full = np.array(f['label'])

                # Get priors
                priors_full = np.array(f['priors'])
                
                if detail_layer is not None and isinstance(detail_layer, int) and 0 <= detail_layer <= 3:
                    dataset_dict['labels'] = labels_full[detail_layer]
                    dataset_dict['prior'] = priors_full[detail_layer]
                else:
                    dataset_dict['labels'] = labels_full
                    dataset_dict['prior'] = priors_full
                    
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return None
            
        return dataset_dict


    def get_random_h5_sample(self, detail_layer=None, static=False):
        """
        Picks a random file from the h5_data directory and returns its data.
        
        Parameters:
        - detail_layer (int): 0, 1, 2, or 3. If None, returns all layers.
        - static (bool): If True, returns only the S2 month matching the EnMAP data.
        """
        all_files = [f for f in os.listdir(self.h5_dir) if f.endswith('.h5')]
        if not all_files:
            print("No .h5 files found in directory.")
            return None
            
        random_file = random.choice(all_files)
        return self._load_single_file(random_file, detail_layer, static)


    def load_h5_data(self, file_list, detail_layer=None, static=False, data_type="hyperspectral", keep_prior=False):
        """
        Loads data based on a pre-determined list of sample IDs, 
        filtering the modalities on the fly to maximize RAM efficiency.
        
        Parameters:
        - file_list (list of str): The list of sample IDs generated by get_file_list().
        - ... (other parameters remain the same)
        """
        if data_type not in ["hyperspectral", "multispectral"]:
            raise ValueError("data_type must be either 'hyperspectral' or 'multispectral'")
            
        if not file_list:
            print("Error: Provided file_list is empty.")
            return []
            
        key_to_remove = "multispectral" if data_type == "hyperspectral" else "hyperspectral"
        
        loaded_data = []
        
        for sample_id in file_list:
            filename = f"{sample_id}.h5"
            data_dict = self._load_single_file(filename, detail_layer, static)
            
            if data_dict:
                # Filter on the fly
                if key_to_remove in data_dict:
                    data_dict.pop(key_to_remove)
                    
                if not keep_prior and 'prior' in data_dict:
                    data_dict.pop('prior')
                    
                loaded_data.append(data_dict)
                
        print(f"Successfully loaded {len(loaded_data)} files into memory.")
        print(f"-> Modality kept: '{data_type}'. Priors kept: {keep_prior}.")
        
        return loaded_data


    def upsample_hyperspectral(self, hyper_data):
        """
        Upsamples the 64x64 hyperspectral array to 192x192 to match Sentinel-2 and labels resolution.
        Uses nearest-neighbor interpolation to maintain exact spectral purity.
        
        Parameters:
        - hyper_data: numpy array of shape (Channels, 64, 64)
        
        Returns:
        - numpy array of shape (Channels, 192, 192)
        """
        # Repeat 3 times along the height (axis 1), then 3 times along the width (axis 2)
        upsampled = np.repeat(np.repeat(hyper_data, 3, axis=1), 3, axis=2)
        return upsampled