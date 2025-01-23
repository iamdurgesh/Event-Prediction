class TransformerWithCustomAttention(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_len):
        super(TransformerWithCustomAttention, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_size))
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(embed_size, num_heads) for _ in range(num_layers)
        ])
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
        )
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_size) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        # Embedding and Positional Encoding
        x = self.embed(x) + self.positional_encoding[:, :x.size(1), :]

        # Apply attention layers
        for attention_layer, norm_layer in zip(self.attention_layers, self.norm_layers):
            attention_output = attention_layer(x)
            x = norm_layer(x + attention_output)

        # Output layer
        return self.fc_out(x)







################

#model code


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by the number of heads."
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        # Define linear transformations for Query, Key, and Value
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        # Output linear layer
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch_size, seq_length, embed_size = x.shape

        # Linear transformations
        queries = self.query(x)  # [batch_size, seq_length, embed_size]
        keys = self.key(x)       # [batch_size, seq_length, embed_size]
        values = self.value(x)   # [batch_size, seq_length, embed_size]

        # Split into heads
        queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute scaled dot-product attention
        energy = torch.einsum("bnqd,bnkd->bnqk", queries, keys)  # [batch_size, num_heads, seq_length, seq_length]
        scaling_factor = self.head_dim ** 0.5
        attention = torch.softmax(energy / scaling_factor, dim=-1)

        # Compute the weighted sum of values
        out = torch.einsum("bnqk,bnvd->bnqd", attention, values)  # [batch_size, num_heads, seq_length, head_dim]

        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_size)

        # Final linear layer
        return self.fc_out(out)
