import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        """
        Args:
            vocab_size (int): Vocabulary size.
            embed_size (int): Embedding size.
            hidden_size (int): Hidden size of the LSTM.
            num_layers (int): Number of LSTM layers.
        """
        super(LSTMModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)  # Embedding layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM
        self.fc_out = nn.Linear(hidden_size, vocab_size)  # Output layer

    def forward(self, x):
        """
        Forward pass of the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits for each token in the sequence.
        """
        x = self.embed(x)  # Shape: (batch_size, seq_len, embed_size)
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, hidden_size)
        logits = self.fc_out(lstm_out)  # Shape: (batch_size, seq_len, vocab_size)
        return logits
