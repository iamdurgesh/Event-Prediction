import torch

def collate_fn(batch):
    """
    Custom collate function to pad sequences in a batch dynamically.
    """
    inputs, targets, masks = zip(*batch)

    # Find the maximum sequence length in the batch
    max_len = max(len(seq) for seq in inputs)

    # Initialize padded tensors
    padded_inputs = torch.zeros(len(inputs), max_len, dtype=torch.long)
    padded_targets = torch.zeros(len(targets), max_len, dtype=torch.long)
    padded_masks = torch.zeros(len(masks), max_len, dtype=torch.long)

    for i in range(len(inputs)):
        seq_len = len(inputs[i])
        padded_inputs[i, :seq_len] = inputs[i]
        padded_targets[i, :seq_len] = targets[i]
        padded_masks[i, :seq_len] = masks[i]

    return padded_inputs, padded_targets, padded_masks
