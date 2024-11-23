import matplotlib.pyplot as plt
import numpy as np

# Parameters for the illustration
input_dim = 128
vocab_size = 32

# Generate a random 2D array to simulate token representations
np.random.seed(42)
token_representations = np.random.rand(10, input_dim)  # Example: 10 tokens with 128-dimensional representation

# Simulate the output of the fully connected layer
output_predictions = np.random.rand(10, vocab_size)

# Create the figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot token representations (input to the fully connected layer)
im1 = axs[0].imshow(token_representations, aspect='auto', cmap='viridis')
axs[0].set_title("128-Dimensional Token Representations")
axs[0].set_xlabel("Features (128 dimensions)")
axs[0].set_ylabel("Tokens")
plt.colorbar(im1, ax=axs[0], orientation='vertical')

# Plot the output predictions (output of the fully connected layer)
im2 = axs[1].imshow(output_predictions, aspect='auto', cmap='coolwarm')
axs[1].set_title("Vocabulary Predictions (32 classes)")
axs[1].set_xlabel("Vocabulary Size (32)")
axs[1].set_ylabel("Tokens")
plt.colorbar(im2, ax=axs[1], orientation='vertical')

# Adjust layout and display
plt.tight_layout()
plt.show()
