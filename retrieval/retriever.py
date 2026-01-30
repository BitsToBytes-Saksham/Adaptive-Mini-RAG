"""
Retriever Module.

Wraps the vector store with logging and statistics tracking.
"""

import torch
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
import time

from .vector_store import VectorStore


@dataclass
class RetrievalLog:
    """Log entry for a single retrieval operation."""
    query_id: int
    k: int
    doc_ids: List[int]
    scores: List[float]
    latency_ms: float
    timestamp: float


@dataclass 
class RetrieverStats:
    """Statistics for retriever operations."""
    total_queries: int = 0
    total_docs_retrieved: int = 0
    total_latency_ms: float = 0.0
    retrieval_logs: List[RetrievalLog] = field(default_factory=list)
    
    def log_retrieval(
        self, 
        query_id: int, 
        k: int, 
        doc_ids: List[int], 
        scores: List[float],
        latency_ms: float
    ):
        """Log a retrieval operation."""
        self.total_queries += 1
        self.total_docs_retrieved += len(doc_ids)
        self.total_latency_ms += latency_ms
        
        self.retrieval_logs.append(RetrievalLog(
            query_id=query_id,
            k=k,
            doc_ids=doc_ids,
            scores=scores,
            latency_ms=latency_ms,
            timestamp=time.time()
        ))
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if self.total_queries == 0:
            return {"error": "No retrievals logged"}
        
        avg_docs = self.total_docs_retrieved / self.total_queries
        avg_latency = self.total_latency_ms / self.total_queries
        
        return {
            "total_queries": self.total_queries,
            "total_docs_retrieved": self.total_docs_retrieved,
            "avg_docs_per_query": avg_docs,
            "total_latency_ms": self.total_latency_ms,
            "avg_latency_ms": avg_latency
        }
    
    def reset(self):
        """Reset all statistics."""
        self.total_queries = 0
        self.total_docs_retrieved = 0
        self.total_latency_ms = 0.0
        self.retrieval_logs = []


class Retriever:
    """
    Document retriever with logging and statistics.
    
    Wraps VectorStore to provide:
    - Top-k document retrieval
    - Retrieval logging and statistics
    - Cost tracking for evaluation
    """
    
    def __init__(
        self, 
        vector_store: VectorStore,
        enable_logging: bool = True
    ):
        """
        Args:
            vector_store: Vector store containing documents
            enable_logging: Whether to track retrieval statistics
        """
        self.vector_store = vector_store
        self.enable_logging = enable_logging
        self.stats = RetrieverStats()
        self.query_counter = 0
    
    def retrieve(
        self, 
        query_embedding: torch.Tensor, 
        k: int = 5,
        return_scores: bool = False
    ) -> List[Tuple[int, str]] | List[Tuple[int, str, float]]:
        """
        Retrieve top-k most similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of documents to retrieve
            return_scores: Whether to include similarity scores
        
        Returns:
            List of (doc_id, document) or (doc_id, document, score) tuples
        """
        start_time = time.time()
        
        results = self.vector_store.search(query_embedding, k)
        
        latency_ms = (time.time() - start_time) * 1000
        
        if self.enable_logging:
            doc_ids = [r[0] for r in results]
            scores = [r[2] for r in results]
            self.stats.log_retrieval(
                query_id=self.query_counter,
                k=k,
                doc_ids=doc_ids,
                scores=scores,
                latency_ms=latency_ms
            )
            self.query_counter += 1
        
        if return_scores:
            return results
        else:
            return [(r[0], r[1]) for r in results]
    
    def retrieve_batch(
        self, 
        query_embeddings: torch.Tensor, 
        k: int = 5,
        return_scores: bool = False
    ) -> List[List[Tuple]]:
        """
        Retrieve documents for a batch of queries.
        
        Args:
            query_embeddings: Query embeddings of shape (batch_size, embed_dim)
            k: Number of documents per query
            return_scores: Whether to include similarity scores
        
        Returns:
            List of lists of retrieved documents
        """
        all_results = []
        
        for i in range(query_embeddings.shape[0]):
            results = self.retrieve(query_embeddings[i], k, return_scores)
            all_results.append(results)
        
        return all_results
    
    def retrieve_with_context(
        self, 
        query_embedding: torch.Tensor, 
        k: int = 5,
        separator: str = "\n\n"
    ) -> str:
        """
        Retrieve documents and concatenate into a context string.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of documents to retrieve
            separator: Separator between documents
        
        Returns:
            Concatenated document string
        """
        results = self.retrieve(query_embedding, k, return_scores=False)
        documents = [doc for _, doc in results]
        return separator.join(documents)
    
    def get_stats(self) -> Dict:
        """Get retrieval statistics."""
        return self.stats.get_summary()
    
    def reset_stats(self):
        """Reset retrieval statistics."""
        self.stats.reset()
        self.query_counter = 0
    
    def get_cost_estimate(self, cost_per_doc: float = 0.001) -> Dict:
        """
        Estimate retrieval cost.
        
        Args:
            cost_per_doc: Cost per document retrieved
        
        Returns:
            Cost estimation dictionary
        """
        summary = self.stats.get_summary()
        if "error" in summary:
            return summary
        
        total_cost = summary["total_docs_retrieved"] * cost_per_doc
        avg_cost = summary["avg_docs_per_query"] * cost_per_doc
        
        return {
            **summary,
            "cost_per_doc": cost_per_doc,
            "total_cost": total_cost,
            "avg_cost_per_query": avg_cost
        }


class MockRetriever:
    """
    Mock retriever for testing without actual vector store.
    """
    
    def __init__(self, documents: List[str]):
        """
        Args:
            documents: List of documents to return
        """
        self.documents = documents
        self.stats = RetrieverStats()
        self.query_counter = 0
    
    def retrieve(
        self, 
        query_embedding: torch.Tensor, 
        k: int = 5,
        return_scores: bool = False
    ) -> List[Tuple]:
        """Return first k documents."""
        k = min(k, len(self.documents))
        results = []
        for i in range(k):
            if return_scores:
                results.append((i, self.documents[i], 1.0 - i * 0.1))
            else:
                results.append((i, self.documents[i]))
        return results
    
    def get_stats(self) -> Dict:
        return self.stats.get_summary()
    
    def reset_stats(self):
        self.stats.reset()
