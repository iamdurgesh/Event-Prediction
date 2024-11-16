import numpy as np
import matplotlib.pyplot as plt
# Example causal mask for a sequence of length 5
sequence_length = 32
causal_mask = np.tril(np.ones((sequence_length, sequence_length), dtype=int))

# Plotting the causal mask
plt.figure(figsize=(6, 5))
plt.imshow(causal_mask, cmap="Blues", interpolation="nearest")
plt.colorbar(label="Attention Mask Value (1 = Attend, 0 = Ignore)")
plt.title("Causal Mask for a Sequence of Length 5")
plt.xlabel("Token Position (Future Tokens Masked)")
plt.ylabel("Token Position (Current Position)")
plt.xticks(range(sequence_length))
plt.yticks(range(sequence_length))
plt.show()
