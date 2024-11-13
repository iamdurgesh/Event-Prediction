import pandas as pd
import torch
from torch.utils.data import Dataset

class EventSequenceDataset(Dataset):
    def __init__(self, sequences, labels, mask):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.float)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.mask[idx]

def create_labeled_dataset_with_mask(file_path, padding_token_id=0):
    """
    Load the dataset, generate labels and mask for duplicated events and padding tokens.
    """
    # Load the dataset based on file extension
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx files.")

    # Extract 'cycle' column and event columns starting from the second row (skipping the header)
    cycle_column = data.iloc[1:, 0].astype(int)
    event_data = data.iloc[1:, 1:].astype(int)

    # Initialize sequence labels and mask
    sequence_labels = event_data.copy()
    mask = event_data.copy().applymap(lambda x: 1)  # Initialize mask to 1 (for unique events)

    # Label duplicated events as 0 in both labels and mask
    for cycle in cycle_column.unique():
        cycle_data = event_data[cycle_column == cycle]
        duplicated_events = cycle_data.columns[cycle_data.sum() > 1]
        for event in duplicated_events:
            sequence_labels.loc[cycle_column == cycle, event] = 0  # Label duplicated as 0
            mask.loc[cycle_column == cycle, event] = 0  # Mask duplicated events as 0

    # Set padding tokens to 0 in mask
    mask[sequence_labels == padding_token_id] = 0

    return sequence_labels.values, mask.values
