"""
Retrieval Controller Module.

Decides whether to retrieve documents and how many based on confidence scores.
Implements the core intelligence of the adaptive RAG system.
"""

import torch
from typing import Tuple, Dict, List, Optional, NamedTuple
from dataclasses import dataclass, field
from enum import Enum


class RetrievalDecision(Enum):
    """Enum for retrieval decisions."""
    NO_RETRIEVAL = "no_retrieval"
    LOW_RETRIEVAL = "low_retrieval"      # k = 1-2
    MEDIUM_RETRIEVAL = "medium_retrieval" # k = 3
    HIGH_RETRIEVAL = "high_retrieval"     # k = 5


@dataclass
class RetrievalResult:
    """Container for retrieval decision results."""
    should_retrieve: bool
    num_docs: int
    confidence: float
    decision: RetrievalDecision
    reasoning: str


@dataclass
class RetrievalStats:
    """Statistics for retrieval operations."""
    total_queries: int = 0
    retrieval_calls: int = 0
    total_docs_retrieved: int = 0
    confidence_scores: List[float] = field(default_factory=list)
    decisions: List[RetrievalDecision] = field(default_factory=list)
    
    def log_query(
        self, 
        confidence: float, 
        decision: RetrievalDecision, 
        docs_retrieved: int
    ):
        """Log a query decision."""
        self.total_queries += 1
        self.confidence_scores.append(confidence)
        self.decisions.append(decision)
        
        if decision != RetrievalDecision.NO_RETRIEVAL:
            self.retrieval_calls += 1
            self.total_docs_retrieved += docs_retrieved
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if self.total_queries == 0:
            return {"error": "No queries logged"}
        
        retrieval_rate = self.retrieval_calls / self.total_queries
        avg_confidence = sum(self.confidence_scores) / len(self.confidence_scores)
        avg_docs = (
            self.total_docs_retrieved / self.retrieval_calls 
            if self.retrieval_calls > 0 else 0
        )
        
        decision_counts = {}
        for d in RetrievalDecision:
            decision_counts[d.value] = sum(1 for x in self.decisions if x == d)
        
        return {
            "total_queries": self.total_queries,
            "retrieval_calls": self.retrieval_calls,
            "retrieval_rate": retrieval_rate,
            "avg_confidence": avg_confidence,
            "avg_docs_per_retrieval": avg_docs,
            "total_docs_retrieved": self.total_docs_retrieved,
            "decision_distribution": decision_counts
        }
    
    def reset(self):
        """Reset all statistics."""
        self.total_queries = 0
        self.retrieval_calls = 0
        self.total_docs_retrieved = 0
        self.confidence_scores = []
        self.decisions = []


class RetrievalController:
    """
    Adaptive Retrieval Controller.
    
    Decides whether to retrieve documents and how many based on model confidence.
    
    Default policy:
    - confidence > 0.7: No retrieval (model is confident)
    - 0.4 < confidence <= 0.7: Retrieve k=3 documents
    - confidence <= 0.4: Retrieve k=5 documents (model is uncertain)
    
    The thresholds are configurable.
    """
    
    def __init__(
        self,
        high_confidence_threshold: float = 0.7,
        medium_confidence_threshold: float = 0.4,
        high_retrieval_k: int = 5,
        medium_retrieval_k: int = 3,
        low_retrieval_k: int = 1,
        enable_logging: bool = True
    ):
        """
        Args:
            high_confidence_threshold: Above this, no retrieval
            medium_confidence_threshold: Below this, maximum retrieval
            high_retrieval_k: Number of docs for low confidence
            medium_retrieval_k: Number of docs for medium confidence
            low_retrieval_k: Number of docs for borderline cases
            enable_logging: Whether to track statistics
        """
        self.high_threshold = high_confidence_threshold
        self.medium_threshold = medium_confidence_threshold
        self.high_k = high_retrieval_k
        self.medium_k = medium_retrieval_k
        self.low_k = low_retrieval_k
        self.enable_logging = enable_logging
        
        self.stats = RetrievalStats()
    
    def decide(self, confidence: float) -> RetrievalResult:
        """
        Make retrieval decision based on confidence score.
        
        Args:
            confidence: Confidence score in [0, 1]
        
        Returns:
            RetrievalResult with decision details
        """
        if confidence > self.high_threshold:
            result = RetrievalResult(
                should_retrieve=False,
                num_docs=0,
                confidence=confidence,
                decision=RetrievalDecision.NO_RETRIEVAL,
                reasoning=f"High confidence ({confidence:.3f} > {self.high_threshold}): Model is confident, no retrieval needed"
            )
        elif confidence > self.medium_threshold:
            result = RetrievalResult(
                should_retrieve=True,
                num_docs=self.medium_k,
                confidence=confidence,
                decision=RetrievalDecision.MEDIUM_RETRIEVAL,
                reasoning=f"Medium confidence ({self.medium_threshold} < {confidence:.3f} <= {self.high_threshold}): Retrieving {self.medium_k} documents"
            )
        else:
            result = RetrievalResult(
                should_retrieve=True,
                num_docs=self.high_k,
                confidence=confidence,
                decision=RetrievalDecision.HIGH_RETRIEVAL,
                reasoning=f"Low confidence ({confidence:.3f} <= {self.medium_threshold}): Model is uncertain, retrieving {self.high_k} documents"
            )
        
        if self.enable_logging:
            self.stats.log_query(confidence, result.decision, result.num_docs)
        
        return result
    
    def decide_batch(self, confidences: torch.Tensor) -> List[RetrievalResult]:
        """
        Make retrieval decisions for a batch of confidence scores.
        
        Args:
            confidences: Confidence scores of shape (batch_size,)
        
        Returns:
            List of RetrievalResult for each item
        """
        results = []
        for conf in confidences.tolist():
            results.append(self.decide(conf))
        return results
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return self.stats.get_summary()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats.reset()
    
    def update_thresholds(
        self, 
        high_threshold: Optional[float] = None,
        medium_threshold: Optional[float] = None
    ):
        """Update confidence thresholds."""
        if high_threshold is not None:
            self.high_threshold = high_threshold
        if medium_threshold is not None:
            self.medium_threshold = medium_threshold


class BaselineController:
    """
    Baseline retrieval controller that always retrieves.
    
    Used for comparison with adaptive controller.
    """
    
    def __init__(self, k: int = 5, enable_logging: bool = True):
        """
        Args:
            k: Always retrieve this many documents
            enable_logging: Whether to track statistics
        """
        self.k = k
        self.enable_logging = enable_logging
        self.stats = RetrievalStats()
    
    def decide(self, confidence: float = 0.0) -> RetrievalResult:
        """Always retrieve k documents."""
        result = RetrievalResult(
            should_retrieve=True,
            num_docs=self.k,
            confidence=confidence,
            decision=RetrievalDecision.HIGH_RETRIEVAL,
            reasoning=f"Baseline: Always retrieving {self.k} documents"
        )
        
        if self.enable_logging:
            self.stats.log_query(confidence, result.decision, result.num_docs)
        
        return result
    
    def decide_batch(self, confidences: torch.Tensor) -> List[RetrievalResult]:
        """Always retrieve k documents for each item."""
        results = []
        for conf in confidences.tolist():
            results.append(self.decide(conf))
        return results
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return self.stats.get_summary()
    
    def reset_stats(self):
        """Reset statistics."""
        self.stats.reset()
