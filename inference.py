"""
Inference Pipeline for Adaptive Mini-RAG.

Implements the two-pass inference system:
1. Pass 1: Question only → confidence estimation
2. Retrieval decision based on confidence
3. Pass 2 (conditional): Question + retrieved docs → answer
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model.transformer import MiniTransformer, MiniTransformerConfig
from controller.confidence import ConfidenceEstimator
from controller.retrieval_controller import RetrievalController, BaselineController, RetrievalResult
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from train import CharTokenizer


@dataclass
class InferenceResult:
    """Container for inference results."""
    question: str
    answer: str
    confidence: float
    used_retrieval: bool
    num_docs_retrieved: int
    retrieved_docs: List[str]
    pass1_tokens: List[str]
    pass2_tokens: Optional[List[str]]
    reasoning: str


class AdaptiveRAGPipeline:
    """
    Adaptive RAG Pipeline with two-pass inference.
    
    Pass 1: Generate initial response without retrieval
    Pass 2: If confidence is low, retrieve documents and regenerate
    """
    
    def __init__(
        self,
        model: MiniTransformer,
        tokenizer: CharTokenizer,
        retriever: Retriever,
        controller: RetrievalController,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_gen_length: int = 100
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.controller = controller
        self.device = device
        self.max_gen_length = max_gen_length
        
        self.confidence_estimator = ConfidenceEstimator(
            vocab_size=len(tokenizer)
        )
        
        self.model.eval()
    
    def encode_question(self, question: str) -> torch.Tensor:
        """Encode question to tensor."""
        prompt = f"Question: {question}\nAnswer:"
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        return torch.tensor([ids], dtype=torch.long, device=self.device)
    
    def encode_with_context(self, question: str, context: str) -> torch.Tensor:
        """Encode question with retrieved context."""
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        return torch.tensor([ids], dtype=torch.long, device=self.device)
    
    def get_query_embedding(self, question: str) -> torch.Tensor:
        """Get embedding for retrieval query."""
        ids = self.tokenizer.encode(question, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        return self.model.get_embedding(input_ids)
    
    @torch.no_grad()
    def generate_pass1(self, question: str) -> Tuple[str, float, torch.Tensor]:
        """
        First pass: Generate response without retrieval.
        
        Returns:
            Tuple of (generated_text, confidence_score, logits)
        """
        input_ids = self.encode_question(question)
        
        # Forward pass to get logits
        logits, hidden_states, _ = self.model(input_ids)
        
        # Compute confidence from logits
        confidence, entropy = self.confidence_estimator(
            logits, 
            aggregation='mean',
            last_n_tokens=10
        )
        
        # Generate response
        generated = self.model.generate(
            input_ids,
            max_new_tokens=self.max_gen_length,
            temperature=0.7,
            top_k=50
        )
        
        # Decode
        generated_ids = generated[0].tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract answer part
        if "Answer:" in text:
            answer = text.split("Answer:")[-1].strip()
        else:
            answer = text
        
        return answer, confidence.item(), logits
    
    @torch.no_grad()
    def generate_pass2(
        self, 
        question: str, 
        retrieved_docs: List[str]
    ) -> Tuple[str, float]:
        """
        Second pass: Generate response with retrieved context.
        
        Args:
            question: The question
            retrieved_docs: List of retrieved documents
        
        Returns:
            Tuple of (generated_text, confidence_score)
        """
        # Combine retrieved docs as context
        context = "\n\n".join(retrieved_docs)
        
        input_ids = self.encode_with_context(question, context)
        
        # Truncate if too long
        if input_ids.shape[1] > self.model.max_seq_len - self.max_gen_length:
            input_ids = input_ids[:, -(self.model.max_seq_len - self.max_gen_length):]
        
        # Forward pass
        logits, _, _ = self.model(input_ids)
        confidence, _ = self.confidence_estimator(
            logits,
            aggregation='mean',
            last_n_tokens=10
        )
        
        # Generate
        generated = self.model.generate(
            input_ids,
            max_new_tokens=self.max_gen_length,
            temperature=0.7,
            top_k=50
        )
        
        generated_ids = generated[0].tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if "Answer:" in text:
            answer = text.split("Answer:")[-1].strip()
        else:
            answer = text
        
        return answer, confidence.item()
    
    def run(self, question: str) -> InferenceResult:
        """
        Run the full adaptive RAG pipeline.
        
        Args:
            question: The input question
        
        Returns:
            InferenceResult with all details
        """
        # Pass 1: Without retrieval
        pass1_answer, confidence, logits = self.generate_pass1(question)
        
        # Get retrieval decision
        decision = self.controller.decide(confidence)
        
        if not decision.should_retrieve:
            # High confidence - return pass 1 answer
            return InferenceResult(
                question=question,
                answer=pass1_answer,
                confidence=confidence,
                used_retrieval=False,
                num_docs_retrieved=0,
                retrieved_docs=[],
                pass1_tokens=pass1_answer.split()[:20],
                pass2_tokens=None,
                reasoning=decision.reasoning
            )
        
        # Low confidence - retrieve and re-generate
        query_embedding = self.get_query_embedding(question)
        
        retrieved = self.retriever.retrieve(
            query_embedding, 
            k=decision.num_docs,
            return_scores=False
        )
        retrieved_docs = [doc for _, doc in retrieved]
        
        # Pass 2: With retrieved context
        pass2_answer, pass2_confidence = self.generate_pass2(question, retrieved_docs)
        
        return InferenceResult(
            question=question,
            answer=pass2_answer,
            confidence=pass2_confidence,
            used_retrieval=True,
            num_docs_retrieved=len(retrieved_docs),
            retrieved_docs=retrieved_docs,
            pass1_tokens=pass1_answer.split()[:20],
            pass2_tokens=pass2_answer.split()[:20],
            reasoning=decision.reasoning
        )
    
    def run_batch(self, questions: List[str]) -> List[InferenceResult]:
        """Run pipeline on batch of questions."""
        return [self.run(q) for q in questions]


class BaselineRAGPipeline:
    """
    Baseline RAG Pipeline that always retrieves.
    
    Used for comparison with adaptive pipeline.
    """
    
    def __init__(
        self,
        model: MiniTransformer,
        tokenizer: CharTokenizer,
        retriever: Retriever,
        k: int = 5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_gen_length: int = 100
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.k = k
        self.device = device
        self.max_gen_length = max_gen_length
        self.controller = BaselineController(k=k)
        
        self.model.eval()
    
    def encode_with_context(self, question: str, context: str) -> torch.Tensor:
        """Encode question with retrieved context."""
        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        return torch.tensor([ids], dtype=torch.long, device=self.device)
    
    def get_query_embedding(self, question: str) -> torch.Tensor:
        """Get embedding for retrieval query."""
        ids = self.tokenizer.encode(question, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        return self.model.get_embedding(input_ids)
    
    @torch.no_grad()
    def run(self, question: str) -> InferenceResult:
        """Always retrieve and generate."""
        # Always retrieve
        query_embedding = self.get_query_embedding(question)
        
        retrieved = self.retriever.retrieve(
            query_embedding,
            k=self.k,
            return_scores=False
        )
        retrieved_docs = [doc for _, doc in retrieved]
        
        # Generate with context
        context = "\n\n".join(retrieved_docs)
        input_ids = self.encode_with_context(question, context)
        
        if input_ids.shape[1] > self.model.max_seq_len - self.max_gen_length:
            input_ids = input_ids[:, -(self.model.max_seq_len - self.max_gen_length):]
        
        generated = self.model.generate(
            input_ids,
            max_new_tokens=self.max_gen_length,
            temperature=0.7,
            top_k=50
        )
        
        generated_ids = generated[0].tolist()
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        if "Answer:" in text:
            answer = text.split("Answer:")[-1].strip()
        else:
            answer = text
        
        return InferenceResult(
            question=question,
            answer=answer,
            confidence=0.0,  # Not computed for baseline
            used_retrieval=True,
            num_docs_retrieved=self.k,
            retrieved_docs=retrieved_docs,
            pass1_tokens=[],
            pass2_tokens=answer.split()[:20],
            reasoning=f"Baseline: Always retrieving {self.k} documents"
        )
    
    def run_batch(self, questions: List[str]) -> List[InferenceResult]:
        """Run pipeline on batch of questions."""
        return [self.run(q) for q in questions]


def load_pipeline(
    checkpoint_dir: str,
    data_dir: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[AdaptiveRAGPipeline, BaselineRAGPipeline]:
    """
    Load trained model and create pipelines.
    
    Args:
        checkpoint_dir: Directory with model checkpoints
        data_dir: Directory with data files
        device: Device to use
    
    Returns:
        Tuple of (adaptive_pipeline, baseline_pipeline)
    """
    # Load config
    config_path = os.path.join(checkpoint_dir, 'config.json')
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = MiniTransformerConfig.from_dict(config_dict)
    
    # Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer.json')
    tokenizer = CharTokenizer.load(tokenizer_path)
    
    # Load model
    model = MiniTransformer(**config.to_dict())
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load documents and create vector store
    doc_path = os.path.join(data_dir, 'documents.txt')
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    documents = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    # Create embeddings for documents
    model = model.to(device)
    model.eval()
    
    doc_embeddings = []
    with torch.no_grad():
        for doc in documents:
            ids = tokenizer.encode(doc, add_special_tokens=False)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            if input_ids.shape[1] > config.max_seq_len:
                input_ids = input_ids[:, :config.max_seq_len]
            emb = model.get_embedding(input_ids)
            doc_embeddings.append(emb.cpu())
    
    doc_embeddings = torch.cat(doc_embeddings, dim=0)
    
    # Create vector store
    vector_store = VectorStore(config.d_model)
    vector_store.add_documents(documents, doc_embeddings)
    
    # Create retriever
    retriever = Retriever(vector_store)
    
    # Create controllers
    adaptive_controller = RetrievalController()
    
    # Create pipelines
    adaptive_pipeline = AdaptiveRAGPipeline(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        controller=adaptive_controller,
        device=device
    )
    
    baseline_pipeline = BaselineRAGPipeline(
        model=model,
        tokenizer=tokenizer,
        retriever=Retriever(vector_store),  # Separate retriever for stats
        k=5,
        device=device
    )
    
    return adaptive_pipeline, baseline_pipeline


def main():
    parser = argparse.ArgumentParser(description='Run Adaptive RAG Inference')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--question', type=str, default=None)
    parser.add_argument('--test', action='store_true', help='Run test inference')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if args.test:
        # Test mode - create minimal pipeline
        print("Running test inference...")
        
        # Create minimal model
        config = MiniTransformerConfig(vocab_size=256, d_model=64, n_heads=2, n_layers=2)
        model = MiniTransformer(**config.to_dict())
        
        # Create minimal tokenizer
        tokenizer = CharTokenizer(256)
        tokenizer.fit(["Hello world", "Test question"])
        
        # Create mock vector store
        vector_store = VectorStore(64)
        docs = ["This is document 1.", "This is document 2."]
        embeddings = torch.randn(2, 64)
        vector_store.add_documents(docs, embeddings)
        
        retriever = Retriever(vector_store)
        controller = RetrievalController()
        
        pipeline = AdaptiveRAGPipeline(
            model=model,
            tokenizer=tokenizer,
            retriever=retriever,
            controller=controller,
            device=device,
            max_gen_length=20
        )
        
        result = pipeline.run("What is 2+2?")
        print(f"Question: {result.question}")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Used retrieval: {result.used_retrieval}")
        print(f"Docs retrieved: {result.num_docs_retrieved}")
        print("Test passed!")
        return
    
    print("Loading pipeline...")
    adaptive_pipeline, baseline_pipeline = load_pipeline(
        args.checkpoint_dir, 
        args.data_dir,
        device
    )
    
    if args.question:
        print(f"\nQuestion: {args.question}")
        result = adaptive_pipeline.run(args.question)
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence:.4f}")
        print(f"Used retrieval: {result.used_retrieval}")
        if result.used_retrieval:
            print(f"Docs retrieved: {result.num_docs_retrieved}")
        print(f"Reasoning: {result.reasoning}")
    else:
        # Interactive mode
        print("\nAdaptive RAG Pipeline ready. Enter questions (or 'quit' to exit):")
        while True:
            question = input("\nQ: ").strip()
            if question.lower() in ['quit', 'exit', 'q']:
                break
            if not question:
                continue
            
            result = adaptive_pipeline.run(question)
            print(f"A: {result.answer}")
            print(f"   [Confidence: {result.confidence:.3f}, "
                  f"Retrieval: {'Yes' if result.used_retrieval else 'No'}"
                  f"{f', Docs: {result.num_docs_retrieved}' if result.used_retrieval else ''}]")


if __name__ == '__main__':
    main()
