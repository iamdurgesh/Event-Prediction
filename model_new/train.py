import torch
import torch.nn as nn
import torch.optim as optim  # Import optimizer module

# Define a multi-label loss using BCEWithLogitsLoss
def calculate_loss(outputs, targets, mask=None):
    """
    Calculate Cross-Entropy Loss with optional masking.

    Args:
        outputs (torch.Tensor): Model outputs of shape [batch_size, seq_len, vocab_size].
        targets (torch.Tensor): Target sequences of shape [batch_size, seq_len].
        mask (torch.Tensor, optional): Binary mask of shape [batch_size, seq_len] to ignore padded tokens.

    Returns:
        torch.Tensor: Loss value.
    """
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token (ID 0)
    
    # Clamp target values to avoid out-of-bounds errors
    targets = torch.clamp(targets, min=0, max=outputs.size(-1) - 1)

    # Reshape outputs and targets for CrossEntropyLoss
    outputs = outputs.view(-1, outputs.size(-1))  # [batch_size * seq_len, vocab_size]
    targets = targets.view(-1)  # [batch_size * seq_len]

    loss = criterion(outputs, targets)
    return loss


def train_model(model, train_loader, val_loader, config, start_epoch=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    device = config["device"]

    training_losses = []
    validation_losses = []

    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        total_loss = 0

        for inputs, targets, mask in train_loader:
            inputs, targets, mask = inputs.to(device), targets.to(device), mask.to(device)

            optimizer.zero_grad()

            src_key_padding_mask = mask == 0
            tgt_key_padding_mask = mask == 0

            outputs = model(
                inputs,
                inputs,
                src_mask=None,
                tgt_mask=None,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

            loss = calculate_loss(outputs, targets, mask)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        print(f"Epoch {epoch}, Training Loss: {avg_train_loss}")

        val_loss = validate_model(model, val_loader, config)
        validation_losses.append(val_loss)

    return training_losses, validation_losses


def validate_model(model, val_loader, config):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, targets, mask in val_loader:
            # Move tensors to device
            inputs, targets, mask = inputs.to(config["device"]), targets.to(config["device"]), mask.to(config["device"])

            # Generate masks
            src_key_padding_mask = mask == 0
            tgt_key_padding_mask = mask == 0

            # Forward pass
            outputs = model(
                inputs,
                inputs,
                src_mask=None,
                tgt_mask=None,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

            # Compute loss
            loss = calculate_loss(outputs, targets, mask)
            total_loss += loss.item()

    avg_val_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss}")
    return avg_val_loss
