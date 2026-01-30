"""
Vector Store for document embeddings.

Simple in-memory vector database with cosine similarity search.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np


class VectorStore:
    """
    Simple in-memory vector store for document retrieval.
    
    Features:
    - Store documents with their embeddings
    - Cosine similarity search
    - Top-k retrieval
    """
    
    def __init__(self, embedding_dim: int):
        """
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.documents: List[str] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.doc_ids: List[int] = []
    
    def add_documents(
        self, 
        documents: List[str], 
        embeddings: torch.Tensor,
        doc_ids: Optional[List[int]] = None
    ):
        """
        Add documents to the store.
        
        Args:
            documents: List of document strings
            embeddings: Embeddings of shape (num_docs, embedding_dim)
            doc_ids: Optional list of document IDs
        """
        assert len(documents) == embeddings.shape[0], \
            f"Number of documents ({len(documents)}) must match embeddings ({embeddings.shape[0]})"
        assert embeddings.shape[1] == self.embedding_dim, \
            f"Embedding dim ({embeddings.shape[1]}) must match store ({self.embedding_dim})"
        
        self.documents.extend(documents)
        
        if doc_ids is None:
            start_id = len(self.doc_ids)
            doc_ids = list(range(start_id, start_id + len(documents)))
        self.doc_ids.extend(doc_ids)
        
        # Normalize embeddings for cosine similarity
        normalized = F.normalize(embeddings, p=2, dim=1)
        
        if self.embeddings is None:
            self.embeddings = normalized
        else:
            self.embeddings = torch.cat([self.embeddings, normalized], dim=0)
    
    def search(
        self, 
        query_embedding: torch.Tensor, 
        k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Search for most similar documents.
        
        Args:
            query_embedding: Query embedding of shape (embedding_dim,) or (1, embedding_dim)
            k: Number of results to return
        
        Returns:
            List of (doc_id, document, similarity_score) tuples
        """
        if self.embeddings is None or len(self.documents) == 0:
            return []
        
        # Ensure query is 2D
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        # Normalize query
        query_normalized = F.normalize(query_embedding, p=2, dim=1)
        
        # Compute cosine similarity
        similarities = torch.matmul(query_normalized, self.embeddings.T).squeeze(0)
        
        # Get top-k
        k = min(k, len(self.documents))
        top_scores, top_indices = torch.topk(similarities, k)
        
        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append((self.doc_ids[idx], self.documents[idx], score))
        
        return results
    
    def search_batch(
        self, 
        query_embeddings: torch.Tensor, 
        k: int = 5
    ) -> List[List[Tuple[int, str, float]]]:
        """
        Search for most similar documents for a batch of queries.
        
        Args:
            query_embeddings: Query embeddings of shape (batch_size, embedding_dim)
            k: Number of results per query
        
        Returns:
            List of lists of (doc_id, document, similarity_score) tuples
        """
        if self.embeddings is None or len(self.documents) == 0:
            return [[] for _ in range(query_embeddings.shape[0])]
        
        # Normalize queries
        queries_normalized = F.normalize(query_embeddings, p=2, dim=1)
        
        # Compute cosine similarities
        similarities = torch.matmul(queries_normalized, self.embeddings.T)
        
        # Get top-k for each query
        k = min(k, len(self.documents))
        all_results = []
        
        for i in range(similarities.shape[0]):
            top_scores, top_indices = torch.topk(similarities[i], k)
            results = []
            for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
                results.append((self.doc_ids[idx], self.documents[idx], score))
            all_results.append(results)
        
        return all_results
    
    def get_document(self, doc_id: int) -> Optional[str]:
        """Get document by ID."""
        try:
            idx = self.doc_ids.index(doc_id)
            return self.documents[idx]
        except ValueError:
            return None
    
    def get_all_documents(self) -> List[Tuple[int, str]]:
        """Get all documents with their IDs."""
        return list(zip(self.doc_ids, self.documents))
    
    def size(self) -> int:
        """Return number of documents in store."""
        return len(self.documents)
    
    def clear(self):
        """Clear all documents."""
        self.documents = []
        self.embeddings = None
        self.doc_ids = []
    
    def save(self, path: str):
        """Save vector store to disk."""
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'doc_ids': self.doc_ids,
            'embedding_dim': self.embedding_dim
        }
        torch.save(data, path)
    
    @classmethod
    def load(cls, path: str) -> 'VectorStore':
        """Load vector store from disk."""
        data = torch.load(path)
        store = cls(data['embedding_dim'])
        store.documents = data['documents']
        store.embeddings = data['embeddings']
        store.doc_ids = data['doc_ids']
        return store
