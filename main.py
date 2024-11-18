import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EventSequenceDataset, create_labeled_dataset_with_mask
from model import TransformerEventModel
from train import train_model, evaluate_with_mask

# Hyperparameters and paths
file_path = 'data/event_only_dataset.csv'  # Replace with your actual file path
vocab_size = 100  # Adjust to match actual event vocabulary size
embed_size = 128
num_heads = 4
num_layers = 2
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# Load data and create dataset
sequences, mask = create_labeled_dataset_with_mask(file_path)
labels = sequences  # Assuming a next-token prediction task
dataset = EventSequenceDataset(sequences, labels, mask)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model, criterion, and optimizer
model = TransformerEventModel(vocab_size=vocab_size, embed_size=embed_size, num_heads=num_heads, num_layers=num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train_model(model, train_loader, criterion, optimizer, num_epochs)

# Evaluate the model with the mask
evaluate_with_mask(model, train_loader)
