import numpy as np
import matplotlib.pyplot as plt

# Updated sequence length and embedding dimension
seq_len = 77  # Sequence length based on the project
embed_size = 128  # Embedding dimension

def get_positional_encodings(seq_len, embed_size):
    """
    Generates sinusoidal positional encodings for a given sequence length and embedding dimension.
    """
    # Initialize positional encodings matrix
    position = np.arange(seq_len)[:, np.newaxis]  # Shape (seq_len, 1)
    div_term = np.exp(np.arange(0, embed_size, 2) * -(np.log(10000.0) / embed_size))  # Shape (embed_size/2,)

    # Apply sine to even indices and cosine to odd indices
    pos_encodings = np.zeros((seq_len, embed_size))
    pos_encodings[:, 0::2] = np.sin(position * div_term)  # Apply sine to even indices
    pos_encodings[:, 1::2] = np.cos(position * div_term)  # Apply cosine to odd indices
    return pos_encodings

# Generate positional encodings with updated sequence length and embedding dimension
positional_encodings = get_positional_encodings(seq_len, embed_size)

# Plot the positional encodings
plt.figure(figsize=(14, 6))
plt.pcolormesh(positional_encodings, cmap='viridis', shading='auto')
plt.xlabel("Embedding Dimension")
plt.ylabel("Position in Sequence")
plt.title("Positional Encodings for Sequence Length of 77 and Embedding Dimension of 128")
plt.colorbar(label="Encoding Value")
plt.show()
