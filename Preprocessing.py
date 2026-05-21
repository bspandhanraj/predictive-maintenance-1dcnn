import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration
DATA_DIR = r"C:\Users\spand\Downloads\predictive-maintenance-1dcnn\dataset\archive\raw"  # Folder containing your .mat files
WINDOW_SIZE = 128           # Number of samples per window
STEP_SIZE = 64              # Overlap step size

def get_cwru_label(file_name):
    """
    Maps CWRU dataset filenames to binary classes.
    Class 0: Normal operation
    Class 1: Fault detected (Ball, Inner Race, or Outer Race)
    """
    name = file_name.upper()
    
    # Check for CWRU Fault prefixes
    if name.startswith('B0') or name.startswith('IR') or name.startswith('OR'):
        return 1
        
    # Check for Normal files (often named 97.mat to 100.mat, or contain 'NORMAL')
    if 'NORMAL' in name or name.startswith('97') or name.startswith('98') or \
       name.startswith('99') or name.startswith('100'):
        return 0
        
    return None

def load_and_window_mat_files(data_dir, window_size, step_size):
    X = []
    y = []
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.mat'):
            file_path = os.path.join(data_dir, file_name)
            
            # 1. Determine class based on CWRU filename
            label = get_cwru_label(file_name)
            
            if label is None:
                print(f"Skipping {file_name}: Could not identify as Normal or Fault.")
                continue
                
            # 2. Load the .mat file
            mat_data = sio.loadmat(file_path)
            
            # Dynamically find the sensor data key (e.g., 'X110_DE_time' in CWRU)
            # We filter out Python internals ('__') and focus on the Drive End (DE) data if multiple exist
            data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
            
            # Prefer Drive End (DE) time series if available, otherwise take the first array
            target_key = data_keys[0]
            for k in data_keys:
                if 'DE_time' in k:
                    target_key = k
                    break
                    
            raw_signal = mat_data[target_key].flatten()
            
            # 3. Sliding window segmentation
            for i in range(0, len(raw_signal) - window_size, step_size):
                window = raw_signal[i:i + window_size]
                X.append(window)
                y.append(label)
                
            print(f"Loaded {file_name} -> Class {label} (Using key: {target_key})")
                
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("Loading and segmenting CWRU .mat files...")
    X, y = load_and_window_mat_files(DATA_DIR, WINDOW_SIZE, STEP_SIZE)
    
    if len(X) == 0:
        raise ValueError("No data loaded. Check the DATA_DIR path and filenames.")

    # Reshape for 1D CNN: (samples, spatial_dim, channels)
    X = np.expand_dims(X, axis=-1)
    
    # Split into Train and Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Normalize features across the time steps
    scaler = StandardScaler()
    num_train, seq_len, num_features = X_train.shape
    num_test, _, _ = X_test.shape
    
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(num_train, seq_len, num_features)
    X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(num_test, seq_len, num_features)
    
    # Save processed arrays
    np.save("X_train.npy", X_train_scaled)
    np.save("X_test.npy", X_test_scaled)
    np.save("y_train.npy", y_train)
    np.save("y_test.npy", y_test)
    
    print(f"\nPreprocessing complete!")
    print(f"Total samples: {len(X)}")
    print(f"Normal samples (Class 0): {np.sum(y == 0)}")
    print(f"Fault samples (Class 1): {np.sum(y == 1)}")
    print(f"Training shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")