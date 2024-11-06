import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import EventPredictionTransformer  # Import your transformer model
from dataset import EventSequenceDataset, load_dataset  # Import your dataset utilities

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
    event_to_idx['<PAD>'] = len(event_to_idx)  # Add padding token
    
    train_dataset = EventSequenceDataset(dataset, event_to_idx)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Model setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_events = len(event_to_idx)
    model = EventPredictionTransformer(num_events).to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=event_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")

        # (Optional) Validation step
        # val_loss = evaluate(model, val_dataloader, criterion, device)
        # print(f"Validation Loss: {val_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'model_checkpoint.pth')
    print("Model saved as model_checkpoint.pth")

if __name__ == "__main__":
    main()
