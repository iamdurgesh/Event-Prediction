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
    inputs, targets = zip(*batch)

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    # Create padding mask (0 = padding token)
    mask = (inputs != 0).float()

    # Optionally filter out excluded IDs
    if excluded_ids:
        targets = torch.tensor(
            [[t if t not in excluded_ids else 0 for t in target_row] for target_row in targets]
        )

    # Ensure there are valid sequences
    if torch.all(targets == 0):
        raise ValueError("All targets are masked or invalid in this batch.")

    return inputs, targets, mask
