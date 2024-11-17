import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Example Transformer model with a single attention layer
class TransformerAttentionExample(torch.nn.Module):
    def __init__(self, embed_size, num_heads):
        super(TransformerAttentionExample, self).__init__()
        self.multihead_attn = torch.nn.MultiheadAttention(embed_size, num_heads)

    def forward(self, x):
        # Compute attention scores and output
        attn_output, attn_weights = self.multihead_attn(x, x, x, need_weights=True)
        return attn_output, attn_weights

# Parameters
seq_len = 77  # Sequence length
embed_size = 128  # Embedding size
num_heads = 4  # Number of attention heads

# Example input (random embedding matrix for 77 events)
input_embeddings = torch.rand(seq_len, 1, embed_size)  # Shape: (seq_len, batch_size, embed_size)

# Instantiate model
model = TransformerAttentionExample(embed_size, num_heads)

# Compute attention
_, attention_weights = model(input_embeddings)

# Attention weights shape: (num_heads, seq_len, seq_len)
# Average across heads to simplify visualization
average_attention = attention_weights.mean(dim=0).squeeze().detach().numpy()

# Plot the attention heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(average_attention, annot=False, cmap="viridis", xticklabels=False, yticklabels=False)
plt.title("Attention Scores Across Events")
plt.xlabel("Key Tokens (Events)")
plt.ylabel("Query Tokens (Events)")
plt.show()
