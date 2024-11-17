import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Example: Assume your embedding layer is trained and available as model.embedding
class MockModel:
    def __init__(self, vocab_size, embedding_dim):
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)

# Define parameters (replace with actual project values)
vocab_size = 32  # Number of unique events
embedding_dim = 128  # Size of embedding vectors

# Mock model (replace with your trained model)
model = MockModel(vocab_size, embedding_dim)

# Extract embedding weights
embedding_weights = model.embedding.weight.detach().numpy()  # Shape: (vocab_size, embedding_dim)

# Dimensionality Reduction (PCA for 3D)
reducer = PCA(n_components=3)
reduced_embeddings = reducer.fit_transform(embedding_weights)  # Shape: (vocab_size, 3)

# Scale the embeddings to significantly increase spacing
scaling_factor = 10.0  # Significantly larger scaling factor
scaled_embeddings = reduced_embeddings * scaling_factor

# Event column indices for visualization
event_column_indices = [
    28, 30, 33, 36, 44, 57, 59, 68, 76, 77, 78, 82, 84, 90, 91, 97, 98, 102,
    103, 104, 105, 106, 107, 108, 110, 114, 115, 116, 117, 118, 119, 120
]

# Plot the scaled embeddings in 3D
fig = plt.figure(figsize=(18, 14))  # Larger plot size for magnification
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(scaled_embeddings[:, 0], scaled_embeddings[:, 1], scaled_embeddings[:, 2], 
                      s=200, c='blue', alpha=0.8)

# Annotate points with event column indices
for i, column_index in enumerate(event_column_indices):
    ax.text(scaled_embeddings[i, 0] + 1.5,  # Offset label slightly to avoid overlap
            scaled_embeddings[i, 1] + 1.5,
            scaled_embeddings[i, 2] + 1.5,
            str(column_index), fontsize=10, color='red')

# Set plot title and labels
ax.set_title("3D Visualization of Input Embeddings (Column Indices)", fontsize=16)
ax.set_xlabel("Principal Component 1 (Scaled)", fontsize=12)
ax.set_ylabel("Principal Component 2 (Scaled)", fontsize=12)
ax.set_zlabel("Principal Component 3 (Scaled)", fontsize=12)

# Adjust viewing angle for better visibility
ax.view_init(elev=25, azim=45)

plt.show()
