import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
from model import EventPredictionTransformer  # Import model class
from dataset import EventSequenceDataset, load_dataset  # Import dataset utilities

def evaluate_model(model, dataloader, criterion, event_to_idx, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    all_preds, all_labels = [], []  # Lists to collect all predictions and labels
    
    with torch.no_grad():  # No gradients needed for evaluation
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:-1])  # Forward pass through model
            loss = criterion(output.view(-1, model.num_events), tgt.view(-1))  # Calculate loss
            total_loss += loss.item()
            
            # Collect predictions and labels for metric calculations
            preds = output.argmax(dim=-1)  # Get predicted classes by finding max logit index
            all_preds.extend(preds.view(-1).cpu().numpy())  # Store predictions as 1D array
            all_labels.extend(tgt.view(-1).cpu().numpy())   # Store labels as 1D array
    
    # Remove padding tokens from predictions and labels for accurate metrics
    pad_idx = event_to_idx['<PAD>']
    all_preds_filtered = [p for p, l in zip(all_preds, all_labels) if l != pad_idx]
    all_labels_filtered = [l for l in all_labels if l != pad_idx]
    
    # Calculate metrics using sklearn
    accuracy = accuracy_score(all_labels_filtered, all_preds_filtered)
    precision = precision_score(all_labels_filtered, all_preds_filtered, average='weighted')
    recall = recall_score(all_labels_filtered, all_preds_filtered, average='weighted')
    f1 = f1_score(all_labels_filtered, all_preds_filtered, average='weighted')
    
    avg_loss = total_loss / len(dataloader)  # Average loss across batches
    
    # Print results
    print("================ Evaluation Results ================")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("====================================================")
    
    return avg_loss, accuracy, precision, recall, f1

def main():
    # Load dataset and generate event-to-index mapping
    dataset = load_dataset('data/final_global_context_dataset.json')  # Load preprocessed dataset
    unique_events = set(event for cycle in dataset for event in cycle)  # Identify unique events
    event_to_idx = {event: idx for idx, event in enumerate(unique_events)}  # Create event-to-index map
    event_to_idx['<PAD>'] = len(event_to_idx)  # Ensure padding token is included
    
    # Initialize test dataset and DataLoader
    test_dataset = EventSequenceDataset(dataset, event_to_idx)  # Placeholder for actual test dataset
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # Ensure DataLoader for testing
    
    # Load trained model
    num_events = len(event_to_idx)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EventPredictionTransformer(num_events).to(device)
    model.load_state_dict(torch.load('model_checkpoint.pth'))  # Load trained weights

    # Define the loss function, ignoring padding
    criterion = nn.CrossEntropyLoss(ignore_index=event_to_idx['<PAD>'])
    
    # Evaluate model and output results
    evaluate_model(model, test_dataloader, criterion, event_to_idx, device)

if __name__ == "__main__":
    main()
