import torch

def collate_fn(batch, max_len=77):
    """
    Custom collate function to pad or truncate sequences in a batch.

    Args:
        batch (list of tuples): Each tuple contains (input, target, mask).
        max_len (int): The fixed maximum length for sequences.

    Returns:
        tuple: Padded inputs, targets, and masks.
    """
    inputs, targets, masks = zip(*batch)

    # Initialize padded tensors with fixed max_len
    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_targets = torch.zeros(len(targets), max_len, dtype=torch.long)
    padded_masks = torch.zeros(len(masks), max_len, dtype=torch.bool)

    for i in range(len(inputs)):
        seq_len = min(len(inputs[i]), max_len)  # Truncate if sequence is longer than max_len
        padded_inputs[i, :seq_len] = inputs[i][:seq_len]
        padded_targets[i, :seq_len] = targets[i][:seq_len]
        padded_masks[i, :seq_len] = masks[i][:seq_len]

    return padded_inputs, padded_targets, padded_masks
