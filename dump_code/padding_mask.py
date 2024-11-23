import numpy as np
import matplotlib.pyplot as plt

# Example sequences with padding (1 = actual token, 0 = padding)
# Sequence 1: [event1, event2, event3]
# Sequence 2: [event4, event5, PAD]
padding_mask = np.array([
    [1, 1, 1],  # No padding in sequence 1
    [1, 1, 0]   # Padding at the end of sequence 2
])

# Plotting the padding mask
plt.figure(figsize=(8, 5))
plt.imshow(padding_mask, cmap="Greys", interpolation="nearest")
plt.colorbar(label="Attention Mask Value (1 = Attend, 0 = Ignore)")
plt.title("Padding Mask for a Batch of Sequences")
plt.xlabel("Token Position")
plt.ylabel("Sequence")
plt.xticks(range(3))
plt.yticks(range(2), labels=["Sequence 1", "Sequence 2"])
plt.show()
