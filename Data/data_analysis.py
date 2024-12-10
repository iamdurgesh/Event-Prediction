import pandas as pd

def extract_event_sequence(file_path, output_file='sequence.csv'):
    """
    This function loads a dataset, extracts all event columns that start with 'event_',
    sorts them by column index, saves the sequence to a CSV file, and returns the sorted sequence.
    
    Parameters:
    file_path (str): The file path to the dataset (supports .csv and .xlsx formats)
    output_file (str): The output file name to save the sorted event sequence
    
    Returns:
    List[Tuple[str, int]]: A sorted list of event column names with their respective IDs (column indices)
    """
    # Load the dataset based on file extension
    if file_path.endswith('.csv'):
        try:
            data = pd.read_csv(file_path, delimiter=',')
        except:
            data = pd.read_csv(file_path, delimiter=';')
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx files.")
    
    # Print column names for verification
    print("Column names in dataset:", data.columns.tolist())
    
    # Extract event columns starting with 'event_'
    event_columns = [col.strip() for col in data.columns if col.strip().startswith('event_')]
    if not event_columns:
        print("No columns starting with 'event_' were found.")
        return []
    
    # Get column indices and sort by original order
    event_ids = [(event, data.columns.get_loc(event)) for event in event_columns]
    sorted_event_sequence = sorted(event_ids, key=lambda x: x[1])
    
    # Save to CSV
    sequence_df = pd.DataFrame(sorted_event_sequence, columns=['Event', 'Column_Index'])
    sequence_df.to_csv(output_file, index=False)
    print(f"Event sequence saved to {output_file}")
    
    return sorted_event_sequence

def create_event_only_dataset(file_path, output_file='event_only_dataset.csv'):
    """
    Creates a dataset with only the first column (assumed to be 'cycle') and event columns, 
    ordered as they appear in the original dataset, with duplicated/unique labeling.
    
    Parameters:
    file_path (str): The file path to the dataset (supports .csv and .xlsx formats)
    output_file (str): The output file name to save the new dataset
    
    Returns:
    pd.DataFrame: A new DataFrame with 'cycle' column and labeled event columns (0 for duplicated, 1 for unique)
    """
    # Load the dataset
    if file_path.endswith('.csv'):
        try:
            data = pd.read_csv(file_path, delimiter=',')
            if data.shape[1] == 1:
                data = pd.read_csv(file_path, delimiter=';')
        except:
            data = pd.read_csv(file_path, delimiter=';')
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx files.")
    
    # Identify 'cycle' column and event columns
    cycle_column_name = data.columns[0]
    columns_to_include = [cycle_column_name] + [col for col in data.columns if col.startswith('event_')]
    event_only_data = data[columns_to_include]
    
    # Prepare to label events as duplicated (0) or unique (1)
    cycle_column = event_only_data[cycle_column_name].astype(int)
    event_data = event_only_data.iloc[:, 1:].astype(int)  # Event columns only
    
    # Initialize sequence label DataFrame
    sequence_labels = event_data.copy()
    
    # Loop through each cycle to mark duplicated and unique events
    for cycle in cycle_column.unique():
        cycle_data = event_data[cycle_column == cycle]
        duplicated_events = cycle_data.columns[cycle_data.sum() > 1]
        
        for event in sequence_labels.columns:
            if event in duplicated_events:
                sequence_labels.loc[cycle_column == cycle, event] = 0  # Duplicated events
            else:
                sequence_labels.loc[cycle_column == cycle, event] = 1  # Unique events
    
    # Concatenate the cycle column back with labeled event data
    labeled_data = pd.concat([cycle_column.reset_index(drop=True), sequence_labels.reset_index(drop=True)], axis=1)
    labeled_data.columns = [cycle_column_name] + list(sequence_labels.columns)  # Restore column names
    
    # Save labeled data
    labeled_data.to_csv(output_file, index=False)
    print(f"Labeled event-only dataset saved to {output_file}")
    
    return labeled_data

# Example usage
file_path = 'data/FGC.csv'  # Replace with the path to your main dataset file
output_file = 'data/event_only_dataset1.csv' 
event_only_data = create_event_only_dataset(file_path, output_file)
print(event_only_data.head())
