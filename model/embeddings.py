"""
Token and Positional Embeddings for the Mini Transformer.

Implements learnable token embeddings and sinusoidal positional encodings.
"""

import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """Learnable token embeddings."""
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Embedding dimension
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Token embeddings of shape (batch_size, seq_len, d_model)
        """
        # Scale embeddings by sqrt(d_model) as per "Attention is All You Need"
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional encodings."""
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model: Embedding dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encodings
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        
        # Register as buffer (not a parameter, but should be saved/loaded)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Positional encodings added to input
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """Combined token and positional embeddings."""
    
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        max_seq_len: int = 512, 
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Embedding dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbedding(d_model, max_seq_len, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Combined embeddings of shape (batch_size, seq_len, d_model)
        """
        token_emb = self.token_embedding(x)
        return self.positional_embedding(token_emb)
