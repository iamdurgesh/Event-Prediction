import pandas as pd
import json
import pickle

# Define constants
cycle_size = 78  # Number of rows per cycle

def extract_global_context_per_cycle(cycles, event_columns):
    global_context_data = []
    for cycle in cycles:
        # Filter only high-impact event columns for the current cycle
        event_sequence = cycle[event_columns].values  # Get as numpy array for easier manipulation
        
        # Collect non-zero (true) events in order of occurrence for the cycle
        cycle_context = []
        for row in event_sequence:
            events_in_row = [event_columns[i] for i, event in enumerate(row) if event == 1]
            cycle_context.extend(events_in_row)  # Append events from the row if they occurred
        
        # Store the collected context (event sequence) for this cycle
        global_context_data.append(cycle_context)

    return global_context_data

def create_global_context_dataset(filtered_data, high_impact_event_columns, save_format="json"):
    # Segment the filtered data into cycles
    segmented_cycles = [filtered_data.iloc[i:i + cycle_size] for i in range(0, len(filtered_data), cycle_size)]
    
    # Extract the global context for each cycle
    global_context_dataset = extract_global_context_per_cycle(segmented_cycles, high_impact_event_columns)
    
    # Save the dataset in the chosen format
    if save_format == "json":
        with open('data/final_global_context_dataset.json', 'w') as f:
            json.dump(global_context_dataset, f)
        print("Dataset saved as JSON.")
    
    elif save_format == "csv":
        # Convert each event sequence to a single comma-separated string
        global_context_strings = [", ".join(context) for context in global_context_dataset]
        df = pd.DataFrame(global_context_strings, columns=['global_context'])
        df.to_csv('data/final_global_context_dataset.csv', index=False)
        print("Dataset saved as CSV.")
    
    elif save_format == "pickle":
        with open('data/final_global_context_dataset.pkl', 'wb') as f:
            pickle.dump(global_context_dataset, f)
        print("Dataset saved as Pickle.")
    else:
        raise ValueError("Unsupported save format. Choose 'json', 'csv', or 'pickle'.")

# Example usage:
# Assuming 'filtered_data_no_duplicates' is your filtered DataFrame with high-impact events
# This should be the final output from data_preprocessing.py
high_impact_event_columns = [
    'event_vehicle_detection', 'event_pedestrian_detection', 'event_traffic_sign', 
    'event_red_traffic_light', 'lane_change', 'left_lane_change', 'right_lane_change', 
    'event_global_path', 'event_intersection'
]

# Call the function with the filtered data
# Uncomment the line below and ensure 'filtered_data_no_duplicates' is passed from preprocessing output
# create_global_context_dataset(filtered_data_no_duplicates, high_impact_event_columns, save_format="json")
