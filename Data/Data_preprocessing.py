import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom Dataset class for CPU usage data
class CPUDataset(Dataset):
    def __init__(self, data, sequence_length=10):
        """
        Args:
            data (DataFrame): The input data.
            sequence_length (int): The length of the input sequences for the Transformer.
        """
        self.data = data
        self.sequence_length = sequence_length
        
        # Normalize the data (excluding the timestamp)
        self.data.iloc[:, 1:] = (self.data.iloc[:, 1:] - self.data.iloc[:, 1:].mean()) / self.data.iloc[:, 1:].std()

        # Convert to numpy array
        self.data = self.data.values

    def __len__(self):
        # The length is reduced by the sequence length to ensure full sequences
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # Get the input sequence and the target
        input_seq = self.data[idx:idx + self.sequence_length, 1:]  # Exclude timestamp
        target_seq = self.data[idx + 1:idx + 1 + self.sequence_length, 1:]  # Shifted by 1 for prediction
        
        # Convert to torch tensors
        input_seq = torch.tensor(input_seq, dtype=torch.float32)
        target_seq = torch.tensor(target_seq, dtype=torch.float32)
        
        return input_seq, target_seq

# Load the data again
file_path = r'D:\ml-codespace\Event-Prediction\Data\Dummydata500.csv'

# Create the dataset and dataloaders
sequence_length = 10  # Define the sequence length
cpu_dataset = CPUDataset(data, sequence_length=sequence_length)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(cpu_dataset))
val_size = len(cpu_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(cpu_dataset, [train_size, val_size])

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# usage in a training loop
for epoch in range(num_epochs):
    for input_seq, target_seq in train_loader:
        # Move to GPU if available
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        
        # Forward pass through the model
        output = model(input_seq, target_seq)
        
        # Calculate loss, backpropagate, and optimize
        loss = criterion(output, target_seq)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
