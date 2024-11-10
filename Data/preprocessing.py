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
        # Define high-impact events
    high_impact_events = [
        'event_vehicle_detection', 'event_pedestrian_detection', 'event_traffic_sign', 
        'event_red_traffic_light', 'lane_change', 'left_lane_change', 'right_lane_change', 
        'event_global_path', 'event_intersection'   
    ]
    
    # Filter the dataset to retain only the high-impact events and context features
    filtered_data = data[high_impact_events + ['timestamp', 'cycle', 'cpu_utility', 'memory_utility']]
    
    return filtered_data 

def handle_missing_values(data, continuous_columns, binary_columns):
    # Interpolate or forward fill for timestamp columns
    timestamp_cols = ['timestamp', 'carla_timestamp', 'task_timestamp']
    data[timestamp_cols] = data[timestamp_cols].interpolate(method='linear', limit_direction='forward').bfill()
    
    # Interpolate or use median for continuous columns
    for col in continuous_columns:
        if data[col].isnull().any():
            data[col] = data[col].interpolate(method='linear', limit_direction='forward').fillna(data[col].median())
    
    # Fill binary columns with 0 (indicating inactive/no event)
    for col in binary_columns:
        data[col] = data[col].fillna(0).astype(int)
    
    return data


def add_features(data):
    # Create a dictionary for new columns to avoid fragmentation
    new_columns = {}
    
    # Task Duration (difference between `event_task_start` and `event_task_end`)
    if 'event_task_start' in data.columns and 'event_task_end' in data.columns:
        new_columns['task_duration'] = data['event_task_end'] - data['event_task_start']
    
    # Rolling averages for CPU and memory utility
    new_columns['cpu_utility_rolling_mean'] = data['cpu_utility'].rolling(window=3, min_periods=1).mean()
    new_columns['memory_utility_rolling_mean'] = data['memory_utility'].rolling(window=3, min_periods=1).mean()
    
    # Cumulative count of `event_task_start`
    if 'event_task_start' in data.columns:
        new_columns['cumulative_task_start'] = data['event_task_start'].cumsum()
    
    # Lagged values for `cpu_utility` and `memory_utility`
    new_columns['cpu_utility_lag_1'] = data['cpu_utility'].shift(1).fillna(0)
    new_columns['cpu_utility_lag_2'] = data['cpu_utility'].shift(2).fillna(0)
    new_columns['memory_utility_lag_1'] = data['memory_utility'].shift(1).fillna(0)
    new_columns['memory_utility_lag_2'] = data['memory_utility'].shift(2).fillna(0)

    # Advanced features for cycles
    cycle_size = 78
    new_columns['cycle_mean_cpu'] = data['cpu_utility'].rolling(window=cycle_size, min_periods=1).mean()
    new_columns['cycle_std_memory'] = data['memory_utility'].rolling(window=cycle_size, min_periods=1).std()
    
    # Time since last task start
    new_columns['time_since_last_task_start'] = data['timestamp'] - data['timestamp'].where(data['event_task_start'] == 1).ffill().fillna(0)
    
    # Exponential moving average for CPU utility
    new_columns['cpu_utility_ema'] = data['cpu_utility'].ewm(span=5, adjust=False).mean()
    
    # Percentage change and delta feature for cpu_utility
    new_columns['cpu_utility_pct_change'] = data['cpu_utility'].pct_change().fillna(0)
    new_columns['cpu_utility_delta'] = data['cpu_utility'].diff().fillna(0)
    
    # Concatenate new columns to avoid fragmentation
    data = pd.concat([data, pd.DataFrame(new_columns)], axis=1)
    
    return data

    
def add_advanced_features(data, cycle_size=78):
# Mean, median, and std per cycle
    data['cycle_mean_cpu'] = data['cpu_utility'].rolling(window=cycle_size, min_periods=1).mean()
    data['cycle_std_memory'] = data['memory_utility'].rolling(window=cycle_size, min_periods=1).std()

# Time since last event_task_start
    data['time_since_last_task_start'] = data['timestamp'] - data['timestamp'].where(data['event_task_start'] == 1).ffill().fillna(0)

# Exponential moving average for CPU utility
    data['cpu_utility_ema'] = data['cpu_utility'].ewm(span=5, adjust=False).mean()

# Percentage change and delta feature for cpu_utility
    data['cpu_utility_pct_change'] = data['cpu_utility'].pct_change().fillna(0)
    data['cpu_utility_delta'] = data['cpu_utility'].diff().fillna(0)

    return data

def segment_by_cycle(data, cycle_size=78):
    # Segment data into cycles of specified size (default 78 rows per cycle)
    cycles = [data.iloc[i:i + cycle_size] for i in range(0, len(data), cycle_size)]
    return cycles


# Define high-impact events based on unique interactions, detection events, and lane changes
high_impact_events = [
    'event_vehicle_detection', 'event_pedestrian_detection', 'event_traffic_sign', 
    'event_red_traffic_light', 'lane_change', 'left_lane_change', 'right_lane_change', 
    'event_global_path', 'event_intersection'
]

# Example of how to use the function on a dataset
# data = pd.read_csv('your_dataset.csv')
# preprocessed_data = preprocess_data(data)
# segmented_cycles = segment_by_cycle(preprocessed_data, cycle_size =78)

def load_and_preprocess_data(input_path, output_path=None):
    # Load the raw dataset
    data = pd.read_csv(input_path)
    
    # Preprocess the data using defined functions
    preprocessed_data = preprocess_data(data)
    
    # Optionally save the preprocessed data to a file
    if output_path:
        preprocessed_data.to_csv(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}")
    
    return preprocessed_data

# Example usage to load, preprocess, and optionally save the preprocessed data
if __name__ == "__main__":
    input_path = 'data/FGC.csv'  # Path to your raw data file
    output_path = 'data/preprocessed_data.csv'   # Path to save preprocessed data
    preprocessed_data = load_and_preprocess_data(input_path, output_path)