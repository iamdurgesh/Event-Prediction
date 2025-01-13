import torch
from torch.utils.data import Dataset
import pandas as pd

# uncomment for unmasked unique events
# class EventDataset(Dataset):
#     def __init__(self, file_path, event_vocab, max_len=77, debug=False):
#         """
#         Args:
#             file_path (str): Path to the dataset CSV file.
#             event_vocab (dict): Mapping of event names to unique IDs.
#             max_len (int): Maximum length for padding sequences.
#             debug (bool): Enable debug logging for dataset creation.
#         """
#         data = pd.read_csv(file_path)

#         self.inputs = []
#         self.targets = []
#         self.event_vocab = event_vocab
#         self.max_len = max_len
#         self.debug = debug

#         for cycle, group in data.groupby("cycle"):
#             if self.debug:
#                 print(f"Processing cycle {cycle}, size: {len(group)}")

#             # Map events to indices
#             sequence = (
#                 group.drop(columns=["cycle"])  # Drop the "cycle" column
#                 .idxmax(axis=1)  # Get column names of max values
#                 .map(event_vocab)  # Map to vocabulary IDs
#                 .fillna(event_vocab.get("<UNK>", 0))  # Replace unmapped events with <PAD> or <UNK>
#                 .astype(int)
#                 .tolist()
#             )

#             print(f"Event columns being processed: {group.drop(columns=['cycle']).columns.tolist()}")


#             if self.debug:
#                 print(f"Mapped sequence: {sequence}")

#             # Skip cycles with only padding
#             if all(event == 0 for event in sequence):
#                 print(f"Skipping cycle {cycle} (all events are <PAD>).")
#                 continue

#             # Check for out-of-range values
#             if any(event >= len(event_vocab) for event in sequence):
#                 print(f"Skipping cycle {cycle} due to out-of-range event: {sequence}")
#                 continue

#             # Padding the input sequence
#             padded_sequence = sequence[:max_len] + [0] * (max_len - len(sequence))

#             # Input and target sequences
#             self.inputs.append(padded_sequence[:-1])  # Remove last for input
#             self.targets.append(padded_sequence[1:])  # Remove first for target

#         # Convert to tensors
#         self.inputs = torch.tensor(self.inputs, dtype=torch.long)
#         self.targets = torch.tensor(self.targets, dtype=torch.long)

#         if self.debug:
#             print(f"Dataset initialized with {len(self.inputs)} sequences.")

#         # Ensure dataset is not empty
#         if len(self.inputs) == 0:
#             raise ValueError(f"No valid sequences found in the dataset: {file_path}")
        
#     def __len__(self):
#         return len(self.inputs)  # Return the number of samples

#     def __getitem__(self, idx):
#         return self.inputs[idx], self.targets[idx]


# ----------------------------For Maked unique event prediction--------------------------------

class EventDataset(Dataset):
    def __init__(self, file_path, event_vocab, max_len=77, excluded_events=None, mode="train", debug=False):
        """
        Args:
            file_path (str): Path to the dataset CSV file.
            event_vocab (dict): Mapping of event names to unique IDs.
            max_len (int): Maximum length for padding sequences.
            excluded_events (list or set): Events to be excluded from the dataset.
            mode (str): Mode of dataset usage: "train", "val", or "test".
            debug (bool): Enable debug logging for dataset creation.
        """
        data = pd.read_csv(file_path)

        self.inputs = []
        self.targets = []
        self.event_vocab = event_vocab
        self.max_len = max_len
        self.debug = debug
        self.mode = mode  # Store mode

        # Convert excluded events to IDs
        excluded_ids = {event_vocab.get(event) for event in excluded_events if event in event_vocab}
        excluded_ids.discard(None)

        for cycle, group in data.groupby("cycle"):
            if self.debug:
                print(f"Processing cycle {cycle}, size: {len(group)}")

            # Map events to indices
            sequence = (
                group.drop(columns=["cycle"])  # Drop the "cycle" column
                .idxmax(axis=1)  # Get column names of max values
                .map(event_vocab)  # Map to vocabulary IDs
                .fillna(event_vocab.get("<PAD>", 0))  # Replace unmapped events with <PAD>
                .astype(int)
                .tolist()
            )

            if self.debug:
                print(f"Mapped sequence before exclusion: {sequence}")

            if self.mode == "train":
                # Replace excluded events with <EXCLUDED>
                filtered_sequence = [
                    event if event not in excluded_ids else event_vocab["<EXCLUDED>"] for event in sequence
                ]
            else:  # "val" or "test"
                # Remove excluded events
                filtered_sequence = [event for event in sequence if event not in excluded_ids]

            if self.debug:
                print(f"Filtered sequence (mode={self.mode}, excluded IDs {excluded_ids}): {filtered_sequence}")

            # Skip cycles with only padding or excluded events
            if not filtered_sequence:
                print(f"Skipping cycle {cycle} (all events are excluded or <PAD>).")
                continue

            # Ensure filtered sequence does not exceed maximum length
            filtered_sequence = filtered_sequence[:max_len]

            # Pad the filtered sequence
            padded_sequence = filtered_sequence + [0] * (max_len - len(filtered_sequence))

            # Input and target sequences
            self.inputs.append(padded_sequence[:-1])  # Remove last for input
            self.targets.append(padded_sequence[1:])  # Remove first for target

        # Convert to tensors
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

        if self.debug:
            print(f"Dataset initialized with {len(self.inputs)} sequences.")

        # Ensure dataset is not empty
        if len(self.inputs) == 0:
            raise ValueError(f"No valid sequences found in the dataset: {file_path}")

    def __len__(self):
        return len(self.inputs)  # Return the number of samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
