import pandas as pd
import matplotlib.pyplot as plt

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
    
    print("Column names in dataset:", data.columns.tolist())
    
    # Extract all column names that start with 'event_'
    event_columns = [col.strip() for col in data.columns if col.strip().startswith('event_')]
    if not event_columns:
        print("No columns starting with 'event_' were found.")
        return []
    
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
#Plotting a graph
# Plotting the event sequence
plt.figure(figsize=(10, 6))
plt.plot(event_sequence['Column_Index'], event_sequence['Event'], marker='o', linestyle='-', color='b')
plt.xlabel('Column Index')
plt.ylabel('Event')
plt.title('Event Sequence by Column Index')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

