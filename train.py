import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer_event_pred import TransformerModel
from Data.Data_preprocessing import CPUDataset
import pandas as pd

# Enabling nested tensor operations
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
# Load the dataset
file_path = r'D:\ml-codespace\Event-Prediction\Data\Dummydata500.csv'
data = pd.read_csv(file_path)

# Create dataset and dataloaders
sequence_length = 10
cpu_dataset = CPUDataset(data, sequence_length)
train_size = int(0.8 * len(cpu_dataset))
val_size = len(cpu_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(cpu_dataset, [train_size, val_size])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, criterion, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerModel(input_dim=20, output_dim=20).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for input_seq, target_seq in train_loader:
        input_seq, target_seq = input_seq.to(device), target_seq.to(device)
        
        # Forward pass through the model
        output = model(input_seq, target_seq)
        
        # Calculate loss, backpropagate, and optimize
        loss = criterion(output, target_seq)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Validation loop (optional)
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for input_seq, target_seq in val_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            output = model(input_seq, target_seq)
            val_loss += criterion(output, target_seq).item()
    
    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
