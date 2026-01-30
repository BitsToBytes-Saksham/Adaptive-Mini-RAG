"""
Confidence Estimation Module.

Computes entropy-based confidence scores from model output logits.
High entropy = model is uncertain = low confidence
Low entropy = model is confident = high confidence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class ConfidenceEstimator(nn.Module):
    """
    Estimates model confidence using entropy of output probability distribution.
    
    The entropy is computed as: H(p) = -sum(p * log(p))
    
    For a vocabulary of size V:
    - Maximum entropy = log(V) (uniform distribution)
    - Minimum entropy = 0 (one-hot distribution)
    
    Confidence score is normalized: confidence = 1 - (entropy / max_entropy)
    """
    
    def __init__(self, vocab_size: int = 256, eps: float = 1e-10):
        """
        Args:
            vocab_size: Size of vocabulary for entropy normalization
            eps: Small value for numerical stability
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.eps = eps
        self.max_entropy = math.log(vocab_size)
    
    def compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of probability distribution from logits.
        
        Args:
            logits: Logits of shape (batch_size, seq_len, vocab_size)
        
        Returns:
            Entropy per position of shape (batch_size, seq_len)
        """
        # Convert to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Clamp for numerical stability
        probs = torch.clamp(probs, min=self.eps)
        
        # Compute entropy: H = -sum(p * log(p))
        log_probs = torch.log(probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        
        return entropy
    
    def compute_confidence(
        self, 
        logits: torch.Tensor,
        aggregation: str = 'mean',
        last_n_tokens: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute confidence score from logits.
        
        Args:
            logits: Logits of shape (batch_size, seq_len, vocab_size)
            aggregation: How to aggregate across sequence ('mean', 'min', 'last')
            last_n_tokens: If set, only consider last N tokens for confidence
        
        Returns:
            Confidence score per batch item of shape (batch_size,)
        """
        # Compute entropy per position
        entropy = self.compute_entropy(logits)  # (batch_size, seq_len)
        
        # Optionally focus on last N tokens
        if last_n_tokens is not None:
            entropy = entropy[:, -last_n_tokens:]
        
        # Aggregate across sequence
        if aggregation == 'mean':
            seq_entropy = entropy.mean(dim=-1)
        elif aggregation == 'min':
            seq_entropy = entropy.min(dim=-1)[0]
        elif aggregation == 'last':
            seq_entropy = entropy[:, -1]
        elif aggregation == 'max':
            seq_entropy = entropy.max(dim=-1)[0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        # Normalize to confidence score [0, 1]
        # High entropy -> low confidence, so we invert
        confidence = 1.0 - (seq_entropy / self.max_entropy)
        
        # Clamp to valid range
        confidence = torch.clamp(confidence, 0.0, 1.0)
        
        return confidence
    
    def forward(
        self, 
        logits: torch.Tensor,
        aggregation: str = 'mean',
        last_n_tokens: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both entropy and confidence.
        
        Args:
            logits: Logits from model
            aggregation: Aggregation method
            last_n_tokens: Optional last N tokens to consider
        
        Returns:
            Tuple of (confidence, entropy)
        """
        entropy = self.compute_entropy(logits)
        
        if last_n_tokens is not None:
            entropy_agg = entropy[:, -last_n_tokens:]
        else:
            entropy_agg = entropy
        
        if aggregation == 'mean':
            seq_entropy = entropy_agg.mean(dim=-1)
        elif aggregation == 'min':
            seq_entropy = entropy_agg.min(dim=-1)[0]
        elif aggregation == 'last':
            seq_entropy = entropy_agg[:, -1]
        elif aggregation == 'max':
            seq_entropy = entropy_agg.max(dim=-1)[0]
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        confidence = 1.0 - (seq_entropy / self.max_entropy)
        confidence = torch.clamp(confidence, 0.0, 1.0)
        
        return confidence, seq_entropy


class HiddenStateConfidence(nn.Module):
    """
    Alternative confidence estimator using learned projection on hidden states.
    
    This can be trained alongside the model to predict retrieval necessity.
    """
    
    def __init__(self, d_model: int, hidden_dim: int = 64):
        """
        Args:
            d_model: Model dimension
            hidden_dim: Hidden dimension for confidence head
        """
        super().__init__()
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence from hidden states.
        
        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, d_model)
        
        Returns:
            Confidence score of shape (batch_size,)
        """
        # Use mean pooled representation
        pooled = hidden_states.mean(dim=1)  # (batch_size, d_model)
        confidence = self.confidence_head(pooled).squeeze(-1)
        return confidence
