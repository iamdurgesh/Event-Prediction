from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm


def train_lstm(model, train_loader, val_loader, config):
    """
    Train the LSTM model and validate after each epoch.

    Args:
        model: LSTM model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Configuration dictionary containing training parameters.

    Returns:
        Tuple: Lists of training and validation losses per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss()
    device = config["device"]

    training_losses = []
    validation_losses = []

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        print(f"Starting Epoch {epoch + 1}/{config['epochs']}")

        for inputs, targets, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['epochs']}"):
            inputs, targets = inputs.to(device), targets.to(device)

            # Zero out gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten for CrossEntropyLoss
            targets = targets.view(-1)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.4f}")

        # Validation after each epoch
        val_loss = validate_lstm(model, val_loader, criterion, device)
        validation_losses.append(val_loss)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

    return training_losses, validation_losses


def validate_lstm(model, val_loader, criterion, device):
    """
    Validate the LSTM model on the validation dataset.

    Args:
        model: LSTM model.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        device: Device (CPU or GPU).

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets, _ in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten for CrossEntropyLoss
            targets = targets.view(-1)

            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

    return total_loss / len(val_loader)
