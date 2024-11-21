import pandas as pd
import matplotlib.pyplot as plt

def extract_task_sequence(file_path, output_file='task_sequence.csv'):
    """
    This function loads a dataset, extracts all columns that start with 'Task',
    sorts them by column index, saves the sequence to a CSV file, and returns the sorted sequence.
    
    Parameters:
    file_path (str): The file path to the dataset (supports .csv and .xlsx formats)
    output_file (str): The output file name to save the sorted task sequence
    
    Returns:
    List[Tuple[str, int]]: A sorted list of task column names with their respective IDs (column indices)
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
    
    # Extract all column names that start with 'Task'
    task_columns = [col.strip() for col in data.columns if col.strip().startswith('Task')]
    if not task_columns:
        print("No columns starting with 'Task' were found.")
        return []
    
    task_ids = [(task, data.columns.get_loc(task)) for task in task_columns]
    
    # Sort the tasks by their column index to preserve the dataset order
    sorted_task_sequence = sorted(task_ids, key=lambda x: x[1])
    
    # Convert the sorted sequence to a DataFrame and save it to CSV
    sequence_df = pd.DataFrame(sorted_task_sequence, columns=['Task', 'Column_Index'])
    sequence_df.to_csv(output_file, index=False)
    print(f"Task sequence saved to {output_file}")
    
    return sorted_task_sequence

# Example usage
file_path = 'data/FGC.csv'  # Replace with the path to your main dataset file
output_file = 'data/task_sequence.csv' 
task_sequence = extract_task_sequence(file_path, output_file)
print(task_sequence)

# Plotting a graph
# Convert task sequence to DataFrame for plotting
if task_sequence:
    task_sequence_df = pd.DataFrame(task_sequence, columns=['Task', 'Column_Index'])
    
    # Plot the task sequence
    plt.figure(figsize=(10, 6))
    plt.plot(task_sequence_df['Column_Index'], task_sequence_df['Task'], marker='o', linestyle='-', color='b')
    plt.xlabel('Column Index')
    plt.ylabel('Task')
    plt.title('Task Sequence by Column Index')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
