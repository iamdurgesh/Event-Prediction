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
        # Try to load with comma delimiter first, then fall back to semicolon if needed
        try:
            data = pd.read_csv(file_path, delimiter=',')
        except:
            data = pd.read_csv(file_path, delimiter=';')
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx files.")
    
    # Check and print column names for verification
    print("Column names in dataset:", data.columns.tolist())
    
    # Extract all column names that start with 'event_'
    event_columns = [col.strip() for col in data.columns if col.strip().startswith('event_')]
    
    # Check if any event columns were found
    if not event_columns:
        print("No columns starting with 'event_' were found.")
        return []
    
    # Get the column indices for each event
    event_ids = [(event, data.columns.get_loc(event)) for event in event_columns]
    
    # Sort the events by their column index to preserve the dataset order
    sorted_event_sequence = sorted(event_ids, key=lambda x: x[1])
    
    # Convert the sorted sequence to a DataFrame and save it to CSV
    sequence_df = pd.DataFrame(sorted_event_sequence, columns=['Event', 'Column_Index'])
    sequence_df.to_csv(output_file, index=False)
    print(f"Event sequence saved to {output_file}")
    
    return sorted_event_sequence

# Example usage
file_path = 'data/FGC.csv'  # Replace with the path to your main dataset file
output_file = 'data/sequence.csv' 
event_sequence = extract_event_sequence(file_path, output_file)
print(event_sequence)


# Event Dataset extraction

def create_event_only_dataset(file_path, output_file='event_only_dataset.csv'):
    """
    This function creates a new dataset with only the first column (assumed to be 'cycle') 
    and all event columns, ordered as they appear in the original dataset.
    
    Parameters:
    file_path (str): The file path to the dataset (supports .csv and .xlsx formats)
    output_file (str): The output file name to save the new dataset
    
    Returns:
    pd.DataFrame: A new DataFrame containing only the 'cycle' column and sorted event columns
    """
    # Load the dataset based on file extension
    if file_path.endswith('.csv'):
        # Try with comma delimiter first, then fall back to semicolon if needed
        try:
            data = pd.read_csv(file_path, delimiter=',')
            if data.shape[1] == 1:  # If all columns are merged, retry with semicolon
                data = pd.read_csv(file_path, delimiter=';')
        except:
            data = pd.read_csv(file_path, delimiter=';')
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .xlsx files.")
    
    # Use the first column as 'cycle' and select all event columns
    cycle_column_name = data.columns[0]  # Get the first column name
    columns_to_include = [cycle_column_name] + [col for col in data.columns if col.startswith('event_')]
    
    # Select only the specified columns to create the new dataset
    event_only_data = data[columns_to_include]
    
    # Save the new dataset to a CSV file
    event_only_data.to_csv(output_file, index=False)
    print(f"Event-only dataset saved to {output_file}")
    
    return event_only_data

# Example usage
file_path = 'data/FGC.csv'  # Replace with the path to your main dataset file
output_file = 'data/event_only_dataset.csv' 
event_only_data = create_event_only_dataset(file_path, output_file)
print(event_only_data.head())