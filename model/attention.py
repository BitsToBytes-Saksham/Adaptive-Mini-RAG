"""
Multi-Head Self-Attention implementation from scratch.

No use of nn.Transformer or nn.MultiheadAttention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism implemented from scratch.
    
    Includes:
    - Query, Key, Value linear projections
    - Scaled dot-product attention
    - Causal masking for autoregressive generation
    - Output projection
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            causal: Whether to apply causal masking (for decoder)
        
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Attention weights of shape (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.W_q(x)  # (batch_size, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch_size, n_heads, seq_len, d_k) @ (batch_size, n_heads, d_k, seq_len)
        # -> (batch_size, n_heads, seq_len, seq_len)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask (for decoder self-attention)
        if causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attention_scores = attention_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply additional mask if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, d_k)
        # -> (batch_size, n_heads, seq_len, d_k)
        context = torch.matmul(attention_weights, V)
        
        # Reshape back
        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(context)
        
        return output, attention_weights


class CrossAttention(nn.Module):
    """
    Cross-Attention for attending to retrieved documents.
    
    Similar to self-attention but Q comes from one source, K/V from another.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Query tensor of shape (batch_size, seq_len_q, d_model)
            context: Key/Value tensor of shape (batch_size, seq_len_kv, d_model)
            mask: Optional attention mask
        
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, seq_len_q, d_model)
                - Attention weights
        """
        batch_size, seq_len_q, _ = x.shape
        _, seq_len_kv, _ = context.shape
        
        Q = self.W_q(x)
        K = self.W_k(context)
        V = self.W_v(context)
        
        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.n_heads, self.d_k).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        out = self.W_o(out)
        
        return out, attention_weights
