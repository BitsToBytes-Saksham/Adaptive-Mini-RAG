"""
Mini Transformer Model - Decoder-Only Architecture.

Implements a complete transformer from scratch without nn.Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from .embeddings import TransformerEmbedding
from .attention import MultiHeadSelfAttention, CrossAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    
    Two linear transformations with GELU activation in between.
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward hidden dimension (typically 4 * d_model)
            dropout: Dropout probability
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer Block with:
    - Multi-head self-attention
    - Optional cross-attention (for RAG)
    - Feed-forward network
    - Residual connections
    - Layer normalization (pre-norm style)
    """
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int, 
        d_ff: int,
        dropout: float = 0.1,
        use_cross_attention: bool = False
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward hidden dimension
            dropout: Dropout probability
            use_cross_attention: Whether to include cross-attention layer
        """
        super().__init__()
        
        # Pre-norm layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) if use_cross_attention else None
        
        # Self-attention
        self.self_attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        # Cross-attention (optional, for RAG)
        self.cross_attention = CrossAttention(d_model, n_heads, dropout) if use_cross_attention else None
        
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        context: Optional[torch.Tensor] = None,
        causal: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            context: Optional context for cross-attention (batch_size, context_len, d_model)
            causal: Whether to apply causal masking
        
        Returns:
            Tuple of:
                - Output tensor of shape (batch_size, seq_len, d_model)
                - Dictionary of attention weights
        """
        attention_weights = {}
        
        # Self-attention with residual
        normed_x = self.norm1(x)
        attn_out, attn_w = self.self_attention(normed_x, causal=causal)
        x = x + self.dropout(attn_out)
        attention_weights['self'] = attn_w
        
        # Cross-attention with residual (if context provided)
        if self.cross_attention is not None and context is not None:
            normed_x = self.norm3(x)
            cross_out, cross_w = self.cross_attention(normed_x, context)
            x = x + self.dropout(cross_out)
            attention_weights['cross'] = cross_w
        
        # Feed-forward with residual
        normed_x = self.norm2(x)
        ff_out = self.feed_forward(normed_x)
        x = x + self.dropout(ff_out)
        
        return x, attention_weights


class MiniTransformer(nn.Module):
    """
    Mini Decoder-Only Transformer for language modeling.
    
    Features:
    - Token + positional embeddings
    - Multiple transformer blocks
    - Final layer norm
    - Language model head
    
    Returns both logits and final hidden state for confidence estimation.
    """
    
    def __init__(
        self,
        vocab_size: int = 256,  # Character-level by default
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_cross_attention: bool = False
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            d_ff: Feed-forward hidden dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            use_cross_attention: Whether to include cross-attention in blocks
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.embedding = TransformerEmbedding(
            vocab_size, d_model, max_seq_len, dropout
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, use_cross_attention)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between embedding and lm_head
        self.lm_head.weight = self.embedding.token_embedding.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_hidden: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Args:
            input_ids: Input token indices of shape (batch_size, seq_len)
            context: Optional context embeddings for cross-attention
            return_hidden: Whether to return final hidden states
        
        Returns:
            Tuple of:
                - Logits of shape (batch_size, seq_len, vocab_size)
                - Final hidden state of shape (batch_size, seq_len, d_model) if return_hidden
                - Dictionary of all attention weights if return_hidden
        """
        # Get embeddings
        x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        all_attention_weights = {}
        
        # Pass through transformer blocks
        for i, layer in enumerate(self.layers):
            x, attn_weights = layer(x, context)
            all_attention_weights[f'layer_{i}'] = attn_weights
        
        # Final layer norm
        hidden_states = self.final_norm(x)
        
        # Language model head
        logits = self.lm_head(hidden_states)
        
        if return_hidden:
            return logits, hidden_states, all_attention_weights
        return logits, None, None
    
    def get_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get mean-pooled embeddings for retrieval.
        
        Args:
            input_ids: Input token indices
        
        Returns:
            Mean-pooled hidden state of shape (batch_size, d_model)
        """
        with torch.no_grad():
            x = self.embedding(input_ids)
            for layer in self.layers:
                x, _ = layer(x, causal=False)  # No causal mask for embedding
            x = self.final_norm(x)
            # Mean pooling
            return x.mean(dim=1)
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Autoregressive text generation.
        
        Args:
            input_ids: Starting tokens of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top-k tokens
            context: Optional context for cross-attention
        
        Returns:
            Generated token sequence
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = input_ids[:, -self.max_seq_len:]
            
            # Get logits
            logits, _, _ = self.forward(idx_cond, context)
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MiniTransformerConfig:
    """Configuration class for MiniTransformer."""
    
    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_cross_attention: bool = False
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.use_cross_attention = use_cross_attention
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MiniTransformerConfig':
        return cls(**config_dict)


def create_model(config: Optional[MiniTransformerConfig] = None) -> MiniTransformer:
    """Create a MiniTransformer model from config."""
    if config is None:
        config = MiniTransformerConfig()
    return MiniTransformer(**config.to_dict())
