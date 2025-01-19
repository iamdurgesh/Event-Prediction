# utils1.py
import torch

#
# def collate_fn(batch):
#     inputs, targets = zip(*batch)

#     inputs = torch.stack(inputs)
#     targets = torch.stack(targets)

#     # Create padding mask (0 = padding token)
#     mask = (inputs != 0).float()

#     return inputs, targets, mask



# for Masked Sequenc
def collate_fn(batch, excluded_ids=None):
    """
    Custom collate function for DataLoader that handles excluded IDs and creates masks.

    Args:
        batch (list): List of (input, target) tuples.
        excluded_ids (set, optional): Set of IDs to exclude from targets.

    Returns:
        torch.Tensor: Stacked inputs.
        torch.Tensor: Stacked filtered targets.
        torch.Tensor: Padding mask.
    """
    inputs, targets = zip(*batch)

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    # Create padding mask (0 = padding token)
    mask = (inputs != 0).float()

    if excluded_ids:
        # Filter out excluded IDs in targets
        filtered_targets = torch.tensor(
            [[t if t not in excluded_ids else 0 for t in target_row] for target_row in targets],
            dtype=torch.long
        )

        print(f"Excluded IDs: {excluded_ids}")
        print(f"Batch Inputs Shape: {inputs.shape}")
        print(f"Batch Targets Shape (before filtering): {targets.shape}")
        print(f"Filtered Targets Shape: {filtered_targets.shape}")
        print(f"Batch Mask Shape: {mask.shape}")

        # Replace the original targets with filtered targets
        targets = filtered_targets

    # Ensure there are valid sequences in the targets
    if torch.all(targets == 0):
        print("Error: All targets are invalid or excluded.")
        raise ValueError("All targets are masked or invalid in this batch.")

    # Validate that inputs and targets do not exceed the vocabulary size
    vocab_size = inputs.size(-1) if len(inputs.size()) > 1 else torch.max(inputs) + 1
    if torch.max(inputs) >= vocab_size:
        print(f"Error: Input contains out-of-vocabulary indices (max: {torch.max(inputs)}, vocab size: {vocab_size}).")
        raise ValueError("Input indices are out of vocabulary range.")
    if torch.max(targets) >= vocab_size:
        print(f"Error: Target contains out-of-vocabulary indices (max: {torch.max(targets)}, vocab size: {vocab_size}).")
        raise ValueError("Target indices are out of vocabulary range.")

    return inputs, targets, mask
