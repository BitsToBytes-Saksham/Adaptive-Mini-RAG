"""
Evaluation Script for Adaptive Mini-RAG.

Compares adaptive RAG vs baseline RAG and generates metrics and visualizations.
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.transformer import MiniTransformer, MiniTransformerConfig
from controller.confidence import ConfidenceEstimator
from controller.retrieval_controller import RetrievalController, BaselineController
from retrieval.vector_store import VectorStore
from retrieval.retriever import Retriever
from train import CharTokenizer
from inference import AdaptiveRAGPipeline, BaselineRAGPipeline, load_pipeline, InferenceResult


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    retrieval_calls: int
    total_queries: int
    avg_docs_retrieved: float
    total_docs_retrieved: int
    retrieval_rate: float
    avg_confidence: float
    confidences: List[float]
    entropies: List[float]
    retrieval_decisions: List[bool]
    correct_predictions: List[bool]
    latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "retrieval_calls": self.retrieval_calls,
            "total_queries": self.total_queries,
            "avg_docs_retrieved": self.avg_docs_retrieved,
            "total_docs_retrieved": self.total_docs_retrieved,
            "retrieval_rate": self.retrieval_rate,
            "avg_confidence": self.avg_confidence,
            "latency_ms": self.latency_ms
        }


def simple_answer_match(predicted: str, ground_truth: str, threshold: float = 0.5) -> bool:
    """
    Simple answer matching using word overlap.
    
    Args:
        predicted: Predicted answer
        ground_truth: Ground truth answer
        threshold: Minimum overlap ratio for match
    
    Returns:
        True if answers match sufficiently
    """
    # Normalize
    pred_words = set(predicted.lower().split())
    truth_words = set(ground_truth.lower().split())
    
    if len(truth_words) == 0:
        return len(pred_words) == 0
    
    # Check for exact substring match
    if ground_truth.lower() in predicted.lower():
        return True
    
    # Check word overlap
    overlap = len(pred_words & truth_words)
    overlap_ratio = overlap / len(truth_words)
    
    return overlap_ratio >= threshold


def evaluate_pipeline(
    pipeline: AdaptiveRAGPipeline | BaselineRAGPipeline,
    questions: List[str],
    answers: List[str],
    name: str = "Pipeline"
) -> EvaluationMetrics:
    """
    Evaluate a RAG pipeline on QA dataset.
    
    Args:
        pipeline: The pipeline to evaluate
        questions: List of questions
        answers: List of ground truth answers
        name: Name for progress bar
    
    Returns:
        EvaluationMetrics with all computed metrics
    """
    results: List[InferenceResult] = []
    correct = 0
    confidences = []
    entropies = []
    retrieval_decisions = []
    correct_predictions = []
    total_docs = 0
    retrieval_calls = 0
    
    start_time = time.time()
    
    for q, a in tqdm(zip(questions, answers), total=len(questions), desc=f"Evaluating {name}"):
        result = pipeline.run(q)
        results.append(result)
        
        # Check correctness
        is_correct = simple_answer_match(result.answer, a)
        if is_correct:
            correct += 1
        correct_predictions.append(is_correct)
        
        # Collect metrics
        confidences.append(result.confidence)
        retrieval_decisions.append(result.used_retrieval)
        
        if result.used_retrieval:
            retrieval_calls += 1
            total_docs += result.num_docs_retrieved
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    accuracy = correct / len(questions) if questions else 0
    avg_docs = total_docs / retrieval_calls if retrieval_calls > 0 else 0
    retrieval_rate = retrieval_calls / len(questions) if questions else 0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return EvaluationMetrics(
        accuracy=accuracy,
        retrieval_calls=retrieval_calls,
        total_queries=len(questions),
        avg_docs_retrieved=avg_docs,
        total_docs_retrieved=total_docs,
        retrieval_rate=retrieval_rate,
        avg_confidence=avg_confidence,
        confidences=confidences,
        entropies=entropies,
        retrieval_decisions=retrieval_decisions,
        correct_predictions=correct_predictions,
        latency_ms=elapsed_ms
    )


def compute_cost_reduction(
    adaptive_metrics: EvaluationMetrics,
    baseline_metrics: EvaluationMetrics
) -> Dict:
    """
    Compute cost reduction of adaptive vs baseline.
    
    Args:
        adaptive_metrics: Metrics from adaptive pipeline
        baseline_metrics: Metrics from baseline pipeline
    
    Returns:
        Dictionary with cost reduction metrics
    """
    baseline_docs = baseline_metrics.total_docs_retrieved
    adaptive_docs = adaptive_metrics.total_docs_retrieved
    
    if baseline_docs == 0:
        reduction = 0.0
    else:
        reduction = (baseline_docs - adaptive_docs) / baseline_docs * 100
    
    return {
        "baseline_total_docs": baseline_docs,
        "adaptive_total_docs": adaptive_docs,
        "docs_saved": baseline_docs - adaptive_docs,
        "cost_reduction_percent": reduction,
        "baseline_retrieval_rate": baseline_metrics.retrieval_rate,
        "adaptive_retrieval_rate": adaptive_metrics.retrieval_rate
    }


def plot_accuracy_vs_retrieval(
    adaptive_metrics: EvaluationMetrics,
    baseline_metrics: EvaluationMetrics,
    output_path: str
):
    """Plot accuracy vs retrieval calls comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Adaptive RAG', 'Baseline RAG']
    accuracies = [adaptive_metrics.accuracy * 100, baseline_metrics.accuracy * 100]
    retrieval_calls = [adaptive_metrics.retrieval_calls, baseline_metrics.retrieval_calls]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='#2ecc71')
    bars2 = ax.bar(x + width/2, retrieval_calls, width, label='Retrieval Calls', color='#3498db')
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Accuracy vs Retrieval Calls: Adaptive vs Baseline RAG', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confidence_entropy(
    metrics: EvaluationMetrics,
    output_path: str
):
    """Plot confidence score distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confidence distribution
    ax1 = axes[0]
    colors = ['#e74c3c' if r else '#2ecc71' for r in metrics.retrieval_decisions]
    ax1.scatter(range(len(metrics.confidences)), metrics.confidences, c=colors, alpha=0.7)
    ax1.axhline(y=0.7, color='#f39c12', linestyle='--', label='High threshold (0.7)')
    ax1.axhline(y=0.4, color='#9b59b6', linestyle='--', label='Low threshold (0.4)')
    ax1.set_xlabel('Query Index', fontsize=12)
    ax1.set_ylabel('Confidence Score', fontsize=12)
    ax1.set_title('Confidence Scores by Query\n(Red = Retrieved, Green = No Retrieval)', fontsize=12)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # Confidence histogram
    ax2 = axes[1]
    retrieved_conf = [c for c, r in zip(metrics.confidences, metrics.retrieval_decisions) if r]
    no_retrieval_conf = [c for c, r in zip(metrics.confidences, metrics.retrieval_decisions) if not r]
    
    ax2.hist(retrieved_conf, bins=20, alpha=0.7, label='Retrieved', color='#e74c3c')
    ax2.hist(no_retrieval_conf, bins=20, alpha=0.7, label='No Retrieval', color='#2ecc71')
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Confidence Distribution by Retrieval Decision', fontsize=12)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_retrieval_distribution(
    adaptive_metrics: EvaluationMetrics,
    baseline_metrics: EvaluationMetrics,
    output_path: str
):
    """Plot retrieval count distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count retrieval decisions
    adaptive_no_retrieval = adaptive_metrics.total_queries - adaptive_metrics.retrieval_calls
    
    categories = ['Retrieved', 'No Retrieval']
    adaptive_counts = [adaptive_metrics.retrieval_calls, adaptive_no_retrieval]
    baseline_counts = [baseline_metrics.retrieval_calls, 0]  # Baseline always retrieves
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, adaptive_counts, width, label='Adaptive RAG', color='#2ecc71')
    bars2 = ax.bar(x + width/2, baseline_counts, width, label='Baseline RAG', color='#e74c3c')
    
    ax.set_xlabel('Retrieval Decision', fontsize=12)
    ax.set_ylabel('Number of Queries', fontsize=12)
    ax.set_title('Retrieval Decision Distribution', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    for bar in bars1 + bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_cost_reduction(
    cost_reduction: Dict,
    output_path: str
):
    """Plot cost reduction comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Baseline RAG', 'Adaptive RAG']
    docs_retrieved = [cost_reduction['baseline_total_docs'], cost_reduction['adaptive_total_docs']]
    
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(categories, docs_retrieved, color=colors)
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Total Documents Retrieved', fontsize=12)
    ax.set_title(f'Cost Reduction: {cost_reduction["cost_reduction_percent"]:.1f}% fewer retrievals', fontsize=14)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=14)
    
    # Add arrow showing savings
    if cost_reduction['docs_saved'] > 0:
        ax.annotate('',
                    xy=(1, docs_retrieved[1]),
                    xytext=(0, docs_retrieved[0]),
                    arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
        ax.text(0.5, (docs_retrieved[0] + docs_retrieved[1]) / 2,
                f'-{cost_reduction["docs_saved"]} docs',
                ha='center', va='center', fontsize=12, color='#3498db')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_evaluation(
    checkpoint_dir: str,
    data_dir: str,
    output_dir: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Dict:
    """
    Run full evaluation comparing adaptive vs baseline RAG.
    
    Args:
        checkpoint_dir: Directory with model checkpoints
        data_dir: Directory with data files
        output_dir: Directory for output files
        device: Device to use
    
    Returns:
        Dictionary with all evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    qa_path = os.path.join(data_dir, 'qa.json')
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)['questions']
    
    questions = [q['question'] for q in qa_data]
    answers = [q['answer'] for q in qa_data]
    needs_retrieval = [q['needs_retrieval'] for q in qa_data]
    
    print(f"Loaded {len(questions)} questions")
    print(f"  - Need retrieval: {sum(needs_retrieval)}")
    print(f"  - Don't need retrieval: {len(questions) - sum(needs_retrieval)}")
    
    print("\nLoading pipelines...")
    try:
        adaptive_pipeline, baseline_pipeline = load_pipeline(checkpoint_dir, data_dir, device)
    except Exception as e:
        print(f"Could not load trained model: {e}")
        print("Running with untrained model for demonstration...")
        
        # Create minimal model for demo
        config = MiniTransformerConfig(vocab_size=256, d_model=128, n_heads=4, n_layers=2)
        model = MiniTransformer(**config.to_dict())
        
        tokenizer = CharTokenizer(256)
        with open(os.path.join(data_dir, 'documents.txt'), 'r', encoding='utf-8') as f:
            docs_text = f.read()
        documents = [p.strip() for p in docs_text.split('\n\n') if p.strip()]
        tokenizer.fit(documents + questions)
        
        # Create embeddings
        model = model.to(device)
        model.eval()
        
        doc_embeddings = []
        with torch.no_grad():
            for doc in documents:
                ids = tokenizer.encode(doc, add_special_tokens=False)
                input_ids = torch.tensor([ids[:256]], dtype=torch.long, device=device)
                emb = model.get_embedding(input_ids)
                doc_embeddings.append(emb.cpu())
        doc_embeddings = torch.cat(doc_embeddings, dim=0)
        
        vector_store = VectorStore(config.d_model)
        vector_store.add_documents(documents, doc_embeddings)
        
        retriever = Retriever(vector_store)
        controller = RetrievalController()
        
        adaptive_pipeline = AdaptiveRAGPipeline(
            model=model,
            tokenizer=tokenizer,
            retriever=retriever,
            controller=controller,
            device=device,
            max_gen_length=50
        )
        
        baseline_pipeline = BaselineRAGPipeline(
            model=model,
            tokenizer=tokenizer,
            retriever=Retriever(vector_store),
            k=5,
            device=device,
            max_gen_length=50
        )
    
    print("\nEvaluating Adaptive RAG...")
    adaptive_metrics = evaluate_pipeline(adaptive_pipeline, questions, answers, "Adaptive")
    
    print("\nEvaluating Baseline RAG...")
    baseline_metrics = evaluate_pipeline(baseline_pipeline, questions, answers, "Baseline")
    
    print("\nComputing cost reduction...")
    cost_reduction = compute_cost_reduction(adaptive_metrics, baseline_metrics)
    
    print("\nGenerating visualizations...")
    plot_accuracy_vs_retrieval(
        adaptive_metrics, baseline_metrics,
        os.path.join(output_dir, 'accuracy_vs_retrieval.png')
    )
    
    plot_confidence_entropy(
        adaptive_metrics,
        os.path.join(output_dir, 'confidence_entropy.png')
    )
    
    plot_retrieval_distribution(
        adaptive_metrics, baseline_metrics,
        os.path.join(output_dir, 'retrieval_distribution.png')
    )
    
    plot_cost_reduction(
        cost_reduction,
        os.path.join(output_dir, 'cost_reduction.png')
    )
    
    # Summary results
    results = {
        "adaptive_rag": adaptive_metrics.to_dict(),
        "baseline_rag": baseline_metrics.to_dict(),
        "cost_reduction": cost_reduction,
        "comparison": {
            "accuracy_diff": adaptive_metrics.accuracy - baseline_metrics.accuracy,
            "retrieval_reduction": baseline_metrics.retrieval_calls - adaptive_metrics.retrieval_calls,
            "docs_saved": cost_reduction['docs_saved'],
            "cost_reduction_percent": cost_reduction['cost_reduction_percent']
        }
    }
    
    # Save results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<30} {'Adaptive':<15} {'Baseline':<15}")
    print("-" * 60)
    print(f"{'Accuracy':<30} {adaptive_metrics.accuracy*100:.1f}%{'':<10} {baseline_metrics.accuracy*100:.1f}%")
    print(f"{'Retrieval Calls':<30} {adaptive_metrics.retrieval_calls:<15} {baseline_metrics.retrieval_calls:<15}")
    print(f"{'Avg Docs Retrieved':<30} {adaptive_metrics.avg_docs_retrieved:.2f}{'':<10} {baseline_metrics.avg_docs_retrieved:.2f}")
    print(f"{'Retrieval Rate':<30} {adaptive_metrics.retrieval_rate*100:.1f}%{'':<10} {baseline_metrics.retrieval_rate*100:.1f}%")
    print(f"{'Avg Confidence':<30} {adaptive_metrics.avg_confidence:.3f}{'':<10} N/A")
    print("-" * 60)
    print(f"\n📊 Cost Reduction: {cost_reduction['cost_reduction_percent']:.1f}%")
    print(f"📄 Documents Saved: {cost_reduction['docs_saved']}")
    print(f"\n✅ Results saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Adaptive Mini-RAG')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='outputs')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = run_evaluation(
        args.checkpoint_dir,
        args.data_dir,
        args.output_dir,
        device
    )


if __name__ == '__main__':
    main()
