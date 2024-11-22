import torch
from torch.utils.data import Dataset
import pandas as pd
import logging

class EventDataset(Dataset):
    def __init__(self, file_path, event_vocab):
        """
        Args:
            file_path (str): Path to the dataset CSV file.
            event_vocab (dict): Mapping of event names to unique IDs.
        """
        data = pd.read_csv(file_path)

        self.inputs = []
        self.targets = []
        self.masks = []
        self.event_vocab = event_vocab

        logging.info(f"Loaded dataset with {len(data)} rows from {file_path}")

        for cycle, group in data.groupby("cycle"):
            logging.debug(f"Processing cycle: {cycle}, Group size: {len(group)}")

            # Map events to indices, replacing invalid events with 0
            sequence = (
                group.drop(columns=["cycle"])
                .idxmax(axis=1)  # Get column names of max values
                .map(self.event_vocab)
                .fillna(0)  # Replace invalid events with a placeholder
                .astype(int)
                .tolist()
            )

            if len(sequence) <= 1:
                logging.warning(f"Skipping cycle {cycle}: Sequence too short.")
                continue

            # Create input and target sequences
            input_seq = sequence[:-1]  # All steps except the last
            target_seq = sequence[1:]  # All steps except the first

            # Create mask (1 for valid events, 0 for placeholder or padding)
            mask = [1 if event != 0 else 0 for event in input_seq]

            self.inputs.append(input_seq)
            self.targets.append(target_seq)
            self.masks.append(mask)

        if len(self.inputs) == 0:
            logging.error(f"No valid sequences found in dataset: {file_path}")
            raise ValueError("Dataset is empty after processing.")

        logging.info(f"Loaded {len(self.inputs)} sequences (including placeholder events) from {file_path}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
            torch.tensor(self.masks[idx], dtype=torch.float),
        )
