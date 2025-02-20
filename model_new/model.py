import torch
import torch.nn as nn

class TransformerSeq2SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_len=77, dropout=0.1):
        super(TransformerSeq2SeqModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.register_buffer('positional_encoding', self.generate_positional_encoding(max_len, embed_size))
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

        # Validate that input indices are within range
        if torch.max(src) >= self.embedding.num_embeddings or torch.min(src) < 0:
            raise ValueError(f"Source input indices out of range: max={torch.max(src)}, min={torch.min(src)}")
        if torch.max(tgt) >= self.embedding.num_embeddings or torch.min(tgt) < 0:
            raise ValueError(f"Target input indices out of range: max={torch.max(tgt)}, min={torch.min(tgt)}")

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


# import torch
# import torch.nn as nn

# class TransformerSeq2SeqModel(nn.Module):
#     def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_len=77, dropout=0.1):
#         super(TransformerSeq2SeqModel, self).__init__()
#         self.embedding = nn.Embedding(vocab_size, embed_size)
#         self.register_buffer('positional_encoding', self.generate_positional_encoding(max_len, embed_size))
#         self.transformer = nn.Transformer(
#             d_model=embed_size,
#             nhead=num_heads,
#             num_encoder_layers=num_layers,
#             num_decoder_layers=num_layers,
#             dropout=dropout,
#             batch_first=True,
#         )
#         self.fc_out = nn.Linear(embed_size, vocab_size)
#         self.max_len = max_len

#     def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, mode="train"):
#         """
#         Forward pass for sequence-to-sequence prediction.

#         Args:
#             src (torch.Tensor): Source sequence [batch_size, seq_len].
#             tgt (torch.Tensor, optional): Target sequence [batch_size, seq_len].
#             src_mask (torch.Tensor, optional): Source mask for self-attention.
#             tgt_mask (torch.Tensor, optional): Target mask for self-attention.
#             src_key_padding_mask (torch.Tensor, optional): Padding mask for source.
#             tgt_key_padding_mask (torch.Tensor, optional): Padding mask for target.
#             mode (str): "train" for teacher forcing or "eval" for autoregressive decoding.

#         Returns:
#             torch.Tensor: Model output [batch_size, seq_len, vocab_size].
#         """
#         # Debugging shapes
#         if src_key_padding_mask is not None:
#             print(f"Source Key Padding Mask Shape: {src_key_padding_mask.shape}")
#         if tgt_key_padding_mask is not None:
#             print(f"Target Key Padding Mask Shape: {tgt_key_padding_mask.shape}")

#         # Validate input ranges
#         if torch.max(src) >= self.embedding.num_embeddings or torch.min(src) < 0:
#             raise ValueError(f"Source input indices out of range: max={torch.max(src)}, min={torch.min(src)}")
#         if tgt is not None and (torch.max(tgt) >= self.embedding.num_embeddings or torch.min(tgt) < 0):
#             raise ValueError(f"Target input indices out of range: max={torch.max(tgt)}, min={torch.min(tgt)}")

#         # Embed the source sequence
#         src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :].to(src.device)

#         if mode == "train":
#             # Teacher forcing: use target sequence during training
#             tgt_emb = self.embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
#             output = self.transformer(
#                 src_emb,
#                 tgt_emb,
#                 src_mask=src_mask,
#                 tgt_mask=tgt_mask,
#                 src_key_padding_mask=src_key_padding_mask,
#                 tgt_key_padding_mask=tgt_key_padding_mask,
#             )
#             logits = self.fc_out(output)  # Shape: [batch_size, seq_len, vocab_size]
#             return logits

#         elif mode == "eval":
#             # Autoregressive decoding during inference
#             batch_size = src.size(0)
#             tgt_emb = self.embedding(torch.zeros(batch_size, 1).long().to(src.device))  # Start token <SOS>
#             outputs = []

#             for t in range(self.max_len):
#                 # Add positional encoding
#                 tgt_emb += self.positional_encoding[:, :tgt_emb.size(1), :].to(tgt_emb.device)

#                 output = self.transformer(
#                     src_emb,
#                     tgt_emb,
#                     src_key_padding_mask=src_key_padding_mask,
#                 )

#                 logits = self.fc_out(output[:, -1:, :])  # Take the last token output
#                 outputs.append(logits)

#                 # Greedy decoding: take the token with the highest probability
#                 next_token = logits.argmax(dim=-1)
#                 tgt_emb = torch.cat([tgt_emb, self.embedding(next_token)], dim=1)

#             return torch.cat(outputs, dim=1)  # Concatenate outputs for the sequence

#     @staticmethod
#     def generate_positional_encoding(max_len, embed_size):
#         """
#         Generate positional encoding for input sequences.

#         Args:
#             max_len (int): Maximum sequence length.
#             embed_size (int): Embedding size.

#         Returns:
#             torch.Tensor: Positional encoding tensor [1, max_len, embed_size].
#         """
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, embed_size, 2) * -(torch.log(torch.tensor(10000.0)) / embed_size))
#         pos_enc = torch.zeros(max_len, embed_size)
#         pos_enc[:, 0::2] = torch.sin(position * div_term)
#         pos_enc[:, 1::2] = torch.cos(position * div_term)
#         return pos_enc.unsqueeze(0)


# def load_trained_model(model_path, vocab_size, embed_size, num_heads, num_layers, max_len, device):
#     """
#     Load a pre-trained model from disk.

#     Args:
#         model_path (str): Path to the saved model weights.
#         vocab_size (int): Vocabulary size.
#         embed_size (int): Embedding size.
#         num_heads (int): Number of attention heads.
#         num_layers (int): Number of Transformer layers.
#         max_len (int): Maximum sequence length.
#         device (str): Device to load the model on.

#     Returns:
#         TransformerSeq2SeqModel: The loaded model.
#     """
#     model = TransformerSeq2SeqModel(
#         vocab_size=vocab_size,
#         embed_size=embed_size,
#         num_heads=num_heads,
#         num_layers=num_layers,
#         max_len=max_len,
#     ).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     print(f"Model loaded from {model_path}")
#     return model


