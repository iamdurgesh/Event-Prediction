import torch
import torch.nn as nn

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            sequences, labels, mask = batch
            optimizer.zero_grad()
            output = model(sequences)
            output = output.view(-1, output.shape[-1])
            labels = labels.view(-1)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader)}")

def evaluate_with_mask(model, data_loader):
    model.eval()
    total_accuracy = 0
    total_count = 0
    with torch.no_grad():
        for batch in data_loader:
            sequences, labels, mask = batch
            output = model(sequences)
            predicted = output.argmax(dim=-1)
            masked_predicted = predicted * mask
            masked_labels = labels * mask
            correct_predictions = (masked_predicted == masked_labels) & (mask == 1)
            accuracy = correct_predictions.sum().float() / mask.sum().float()
            total_accuracy += accuracy.item() * mask.sum().item()
            total_count += mask.sum().item()
    print(f"Masked Evaluation Accuracy: {total_accuracy / total_count}")
