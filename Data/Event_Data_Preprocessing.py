import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    # Load data
    data = pd.read_excel(file_path)
    
    # Example preprocessing: Normalize cpu_utility, memory_utility, and steer
    scaler = MinMaxScaler()
    data[['cpu_utility_scaled', 'memory_utility_scaled', 'steer_scaled']] = scaler.fit_transform(
        data[['cpu_utility', 'memory_utility', 'steer']]
    )
    
    # Select feature and target columns
    feature_cols = ['cpu_utility_scaled', 'memory_utility_scaled', 'steer_scaled']
    target_cols = ['event_vehicle_detection', 'event_lane_keeping']  # Example target labels
    
    # Convert to PyTorch tensors
    features = torch.tensor(data[feature_cols].values, dtype=torch.float32)
    targets = torch.tensor(data[target_cols].values, dtype=torch.float32)
    
    return features, targets