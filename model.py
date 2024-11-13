import torch.nn as nn
import torch

class TransformerEventModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, dropout=0.1):
        super(TransformerEventModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 77, embed_size))  # Adjust length if needed
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers, dropout=dropout)
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        x = self.embedding(x) + self.positional_encoding[:, :seq_len, :]
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, embed_size)
        output = self.transformer(x, x)
        output = self.fc_out(output.permute(1, 0, 2))  # Convert back to (batch, seq_len, vocab_size)
        return output
