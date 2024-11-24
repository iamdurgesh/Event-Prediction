import torch
import torch.nn as nn

class TransformerSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_len=77, dropout=0.1):
        super(TransformerSeq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = self.generate_positional_encoding(max_len, embed_size)
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True, 
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        Forward pass for sequence-to-sequence prediction.

        Args:
            src (torch.Tensor): Source sequence [batch_size, seq_len].
            tgt (torch.Tensor): Target sequence [batch_size, seq_len].
            src_mask (torch.Tensor, optional): Source mask for self-attention.
            tgt_mask (torch.Tensor, optional): Target mask for self-attention.
            src_key_padding_mask (torch.Tensor, optional): Padding mask for source.
            tgt_key_padding_mask (torch.Tensor, optional): Padding mask for target.

        Returns:
            torch.Tensor: Model output [batch_size, seq_len, vocab_size].
        """
        # Debugging shapes
        if src_key_padding_mask is not None:
            print(f"Source Key Padding Mask Shape: {src_key_padding_mask.shape}")
        if tgt_key_padding_mask is not None:
            print(f"Target Key Padding Mask Shape: {tgt_key_padding_mask.shape}")

        # Embed the source and target sequences
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :].to(src.device)
        tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)

        # Pass through the Transformer
        output = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Pass through the final output layer
        logits = self.fc_out(output)  # Shape: [batch_size, seq_len, vocab_size]

        return logits

    @staticmethod
    def generate_positional_encoding(max_len, embed_size):
        """
        Generate positional encoding for input sequences.

        Args:
            max_len (int): Maximum sequence length.
            embed_size (int): Embedding size.

        Returns:
            torch.Tensor: Positional encoding tensor [1, max_len, embed_size].
        """
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * -(torch.log(torch.tensor(10000.0)) / embed_size))
        pos_enc = torch.zeros(max_len, embed_size)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc.unsqueeze(0)


def load_trained_model(model_path, vocab_size, embed_size, num_heads, num_layers, max_len, device):
    """
    Load a pre-trained model from disk.

    Args:
        model_path (str): Path to the saved model weights.
        vocab_size (int): Vocabulary size.
        embed_size (int): Embedding size.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of Transformer layers.
        max_len (int): Maximum sequence length.
        device (str): Device to load the model on.

    Returns:
        TransformerSeq2SeqModel: The loaded model.
    """
    model = TransformerSeq2SeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Model loaded from {model_path}")
    return model


