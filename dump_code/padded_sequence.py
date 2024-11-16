import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence

# Extended example sequences with varying lengths
sequences = [
    torch.tensor([0, 0, 1, 2, 3, 4, 5]),  # Sequence with first two columns padded
    torch.tensor([0, 0, 6, 7, 8]),        # Sequence with first two columns padded
    torch.tensor([0, 0, 9, 10, 11, 12, 13, 14]),  # Longer sequence
    torch.tensor([0, 0, 15, 16]),         # Short sequence
    torch.tensor([0, 0, 17, 18, 19, 20, 21]),    # New sequence
    torch.tensor([0, 0, 22, 23, 24, 25, 26, 27]),  # Another longer sequence
]

# Pad the sequences to the max sequence length
max_seq_len = 32
padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)

# Ensure sequences are padded to the full max_seq_len
if padded_sequences.shape[1] < max_seq_len:
    padding_size = max_seq_len - padded_sequences.shape[1]
    padded_sequences = torch.nn.functional.pad(
        padded_sequences, (0, padding_size), mode="constant", value=0
    )

# Plot the padded sequences as a heatmap
plt.figure(figsize=(12, 8))
plt.pcolormesh(
    padded_sequences.numpy(),
    cmap="Blues",
    edgecolor="k",
    linewidth=0.01,
    shading="nearest"
)
plt.xlabel("Sequence Length (Padded Dimension)")
plt.ylabel("Batch Index")
plt.title("Visualization of Padded Sequences with Max Length 32")
plt.colorbar(label="Value (0 indicates padding)")
plt.tight_layout()
plt.show()
