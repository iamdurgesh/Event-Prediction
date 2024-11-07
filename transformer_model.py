import torch
import torch.nn as nn
import math

class EventPredictionTransformer(nn.Module):
    def __init__(self, num_events, d_model=512, nhead=8, num_encoder_layers=4, num_decoder_layers=4, dim_feedforward=2048, dropout=0.1, max_seq_length=78):
        super(EventPredictionTransformer, self).__init__()
        self.d_model = d_model
        self.num_events = num_events
        
        # Embedding layer for input events (num_events x d_model)
        self.event_embedding = nn.Embedding(num_events, d_model)
        
        # Positional encoding for sequence order awareness
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Transformer model with specified encoder and decoder layers
        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        
        # Linear layer to project the transformer output to event logits (num_events)
        self.output_layer = nn.Linear(d_model, num_events)
    
    def forward(self, src, tgt):
        """
        Forward pass for the transformer model.
        src: Source sequence input (e.g., previous events)
        tgt: Target sequence for prediction
        """
        # Embed source and target sequences, scale by sqrt(d_model)
        src = self.event_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        
        tgt = self.event_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        
        # Pass the embedded sequences through the transformer
        output = self.transformer(src, tgt)
        
        # Map transformer output to logits over the event vocabulary
        output = self.output_layer(output)
        
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Precompute positional encodings once for efficiency
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register as a buffer so it’s not updated during training
        self.register_buffer('pe', pe)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # """
        # Forward pass with optional masks for padding and causality.
        # """
        src = self.event_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        
        tgt = self.event_embedding(tgt) * math.sqrt(self.d_model)
        tgt = self.positional_encoding(tgt)
        
        # Pass through transformer with masks
        output = self.transformer(
            src, tgt,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_mask=self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)  # Causal mask
        )
        
        output = self.output_layer(output)
        return output

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a causal (future-blind) mask for the target sequence.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).type(torch.bool)
        return mask

