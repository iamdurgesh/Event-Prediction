import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import EventPredictionTransformer
from dataset import EventSequenceDataset, load_dataset

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:-1])  # Shift target for transformer
        loss = criterion(output.view(-1, model.num_events), tgt.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:-1])
            loss = criterion(output.view(-1, model.num_events), tgt.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    # Load dataset and create DataLoader
    dataset = load_dataset('data/final_global_context_dataset.json')
    unique_events = set(event for cycle in dataset for event in cycle)
    event_to_idx = {event: idx for idx, event in enumerate(unique_events)}
    event_to_idx['<PAD>'] = len(event_to_idx)
    
    train_dataset = EventSequenceDataset(dataset, event_to_idx)
    
    # Hyperparameters and model setup
    num_events = len(event_to_idx)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model parameters for tuning
    model = EventPredictionTransformer(
        num_events=num_events,
        d_model=256,                # Embedding size
        nhead=4,                    # Number of attention heads
        num_encoder_layers=3,       # Encoder layers
        num_decoder_layers=3,       # Decoder layers
        dim_feedforward=1024,       # Feedforward network dimension
        dropout=0.2,                # Dropout rate
        max_seq_length=78
    ).to(device)

    # Hyperparameters for optimization
    criterion = nn.CrossEntropyLoss(ignore_index=event_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.0005)  # Learning rate
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'model_checkpoint.pth')
    print("Model saved as model_checkpoint.pth")

if __name__ == "__main__":
    main()
