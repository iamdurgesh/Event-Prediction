import numpy as np
import matplotlib.pyplot as plt

# Define the total events (positions) based on your dataset
event_positions = [
    28, 30, 33, 36, 44, 57, 59, 68, 76, 77, 78, 82, 84, 90, 91, 97, 98, 102,
    103, 104, 105, 106, 107, 108, 110, 114, 115, 116, 117, 118, 119, 120
]

# Define embedding dimensions (128 as recommended)
embedding_dim = 128

# Function to compute positional encodings
def compute_positional_encodings(position, d_model):
    encodings = np.zeros(d_model)
    for i in range(d_model):
        if i % 2 == 0:  # Even index
            encodings[i] = np.sin(position / (10000 ** (2 * i / d_model)))
        else:  # Odd index
            encodings[i] = np.cos(position / (10000 ** (2 * i / d_model)))
    return encodings

# Compute positional encodings for all event positions
positional_encodings = [compute_positional_encodings(pos, embedding_dim) for pos in event_positions]

# Convert to a numpy array for easier plotting
positional_encodings = np.array(positional_encodings)

# Select a subset of embedding dimensions for visualization
selected_dims = [0, 1, 2, 3, 64, 65]  # Example: First few and middle dimensions

# Plot positional encodings for the selected dimensions
plt.figure(figsize=(14, 8))
for dim in selected_dims:
    plt.plot(event_positions, positional_encodings[:, dim], label=f'Dim {dim}')

plt.title("Sinusoidal Positional Encodings for Events (Selected Dimensions)")
plt.xlabel("Event Positions (Column_Index)")
plt.ylabel("Positional Encoding Value")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
