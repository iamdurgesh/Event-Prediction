import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    # Define continuous and binary columns
    continuous_columns = [
        'cpu_utility', 'memory_utility', 'longitude', 'latitude', 'acc_x', 'acc_y', 'acc_z',
        'gyro_x', 'gyro_y', 'gyro_z', 'compass', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 
        's8', 's9', 's10', 's11', 's12', 'radar_distance', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 
        'c7', 'c8', 'loc_x', 'loc_y', 'loc_z', 'rot_yaw', 'vel_x', 'vel_y', 'vel_z', 'v_distance', 
        'v_x', 'v_y', 'v_z', 'v_yaw', 'p_distance', 'p_x', 'p_y', 'p_z', 'p_yaw', 'l_x', 'l_y', 
        'l_z', 'Local_remaining_points_len', 'throttle', 'steer', 'brake'
    ]
    binary_columns = [col for col in data.columns if col.startswith("Task") or col.startswith("event")]
    
    # Handle missing values
    data = handle_missing_values(data, continuous_columns, binary_columns)
    
    # Normalize continuous columns
    scaler = MinMaxScaler()
    data[continuous_columns] = scaler.fit_transform(data[continuous_columns])
    
    # Ensure binary columns are in 0/1 format
    data[binary_columns] = data[binary_columns].fillna(0).astype(int)
    
    # Feature engineering: Add task duration, rolling averages, cumulative counts, and lagged values
    data = add_features(data)
    
    return data

def handle_missing_values(data, continuous_columns, binary_columns):
    # Interpolate or forward fill for timestamp columns
    timestamp_cols = ['timestamp', 'carla_timestamp', 'task_timestamp']
    data[timestamp_cols] = data[timestamp_cols].interpolate(method='linear', limit_direction='forward').fillna(method='bfill')
    
    # Interpolate or use median for continuous columns
    for col in continuous_columns:
        if data[col].isnull().any():
            data[col] = data[col].interpolate(method='linear', limit_direction='forward').fillna(data[col].median())
    
    # Fill binary columns with 0 (indicating inactive/no event)
    for col in binary_columns:
        data[col] = data[col].fillna(0).astype(int)
    
    return data

def add_features(data):
    # Task Duration (difference between `event_task_start` and `event_task_end`)
    if 'event_task_start' in data.columns and 'event_task_end' in data.columns:
        data['task_duration'] = data['event_task_end'] - data['event_task_start']
    
    # Rolling averages for CPU and memory utility over a window of 3
    data['cpu_utility_rolling_mean'] = data['cpu_utility'].rolling(window=3, min_periods=1).mean()
    data['memory_utility_rolling_mean'] = data['memory_utility'].rolling(window=3, min_periods=1).mean()
    
    # Cumulative count of `event_task_start`
    if 'event_task_start' in data.columns:
        data['cumulative_task_start'] = data['event_task_start'].cumsum()
    
    # Lagged values for `cpu_utility` and `memory_utility`
    data['cpu_utility_lag_1'] = data['cpu_utility'].shift(1).fillna(0)
    data['cpu_utility_lag_2'] = data['cpu_utility'].shift(2).fillna(0)
    data['memory_utility_lag_1'] = data['memory_utility'].shift(1).fillna(0)
    data['memory_utility_lag_2'] = data['memory_utility'].shift(2).fillna(0)
    
    return data