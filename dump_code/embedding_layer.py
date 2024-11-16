import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Define parameters
vocab_size = 32  # Total number of events (assuming 32 including duplicated)
embed_size = 128  # Embedding dimension matching the model

# Initialize an embedding layer for the vocabulary
embedding_layer = torch.nn.Embedding(vocab_size, embed_size)

# Generate embeddings for each event in the vocabulary
event_ids = torch.arange(vocab_size)  # Generate event IDs from 0 to vocab_size-1
embeddings = embedding_layer(event_ids).detach().numpy()  # Get embeddings as a numpy array

# Apply t-SNE to reduce embeddings to 2D for visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the embeddings in 2D space
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], marker='o', color='b')

# Annotate points with event IDs
for i in range(vocab_size):
    plt.text(embeddings_2d[i, 0], embeddings_2d[i, 1], str(i), fontsize=9, ha='right')

plt.title("2D Visualization of Embedding Layer for Event IDs")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()




# once the model is trained we can plot the train embeddings to find the relationship
# Assuming `model` is your trained Transformer model
# and model.embedding is the trained embedding layer


# trained_embeddings = model.embedding.weight.detach().numpy()  # Get trained embeddings as numpy array

# # Use t-SNE to reduce to 2D for visualization
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, random_state=42, perplexity=5)
# trained_embeddings_2d = tsne.fit_transform(trained_embeddings)

# # Plot the trained embeddings in 2D space
# plt.figure(figsize=(10, 10))
# plt.scatter(trained_embeddings_2d[:, 0], trained_embeddings_2d[:, 1], marker='o', color='b')

# # Annotate points with event IDs (0 to 31)
# for i in range(vocab_size):
#     plt.text(trained_embeddings_2d[i, 0], trained_embeddings_2d[i, 1], str(i), fontsize=9, ha='right')

# plt.title("2D Visualization of Trained Embedding Layer for Event IDs")
# plt.xlabel("t-SNE Dimension 1")
# plt.ylabel("t-SNE Dimension 2")
# plt.show()
