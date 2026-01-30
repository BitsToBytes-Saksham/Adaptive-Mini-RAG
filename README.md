# Adaptive Mini-RAG with Transformer-Based Retrieval Control

A mini Retrieval-Augmented Generation (RAG) system where a custom transformer model dynamically decides whether retrieval is needed and how many documents to retrieve based on model confidence.

## 🎯 Motivation

Traditional RAG systems **always retrieve documents**, even when the model already knows the answer. This leads to:
- Unnecessary latency from retrieval operations
- Increased API costs for vector store queries
- Wasted computational resources

This project introduces a **retrieval controller driven by model uncertainty**, allowing the system to:
- ✅ Skip retrieval for easy or confident questions
- ✅ Retrieve more documents only for complex or uncertain queries
- ✅ Reduce costs while maintaining answer accuracy

## 🏗️ Architecture

```
User Question
      ↓
┌─────────────────────────────┐
│  Transformer (Pass 1)        │
│  - No retrieval              │
│  - Generate initial response │
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│  Confidence Estimator        │
│  - Compute entropy from      │
│    output logits             │
│  - High entropy = uncertain  │
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│  Retrieval Controller        │
│  ┌─────────────────────────┐ │
│  │ If confident (>0.7)     │─┼──→ Answer directly
│  │ If medium (0.4-0.7)     │─┼──→ Retrieve k=3 docs
│  │ If uncertain (<0.4)     │─┼──→ Retrieve k=5 docs
│  └─────────────────────────┘ │
└─────────────────────────────┘
      ↓ (if retrieval needed)
┌─────────────────────────────┐
│  Adaptive Retriever          │
│  - Vector similarity search  │
│  - Top-k document retrieval  │
└─────────────────────────────┘
      ↓
┌─────────────────────────────┐
│  Transformer (Pass 2)        │
│  - Question + retrieved docs │
│  - Final answer generation   │
└─────────────────────────────┘
      ↓
    Final Answer
```

## 📁 Project Structure

```
mini-adaptive-rag/
├── data/
│   ├── documents.txt          # Document corpus (50 paragraphs)
│   └── qa.json                # Q&A dataset (100 questions)
├── model/
│   ├── __init__.py
│   ├── embeddings.py          # Token + Positional embeddings
│   ├── attention.py           # Multi-head self-attention
│   └── transformer.py         # Full decoder-only transformer
├── controller/
│   ├── __init__.py
│   ├── confidence.py          # Entropy-based confidence estimation
│   └── retrieval_controller.py # Adaptive retrieval logic
├── retrieval/
│   ├── __init__.py
│   ├── vector_store.py        # In-memory vector database
│   └── retriever.py           # Top-k retriever with logging
├── train.py                   # Training script
├── inference.py               # Two-pass inference pipeline
├── evaluate.py                # Evaluation and visualization
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🔧 Installation

```bash
# Clone or navigate to the project
cd Mini-RAG

# Create virtual environment (optional)
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Train the Model

```bash
python train.py --epochs 10 --batch_size 32 --d_model 256 --n_layers 4
```

**Options:**
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--d_model`: Model dimension (default: 256)
- `--n_heads`: Attention heads (default: 4)
- `--n_layers`: Transformer layers (default: 4)
- `--output_dir`: Checkpoint directory (default: checkpoints)

### 2. Run Inference

```bash
# Single question
python inference.py --question "What is the speed of light?"

# Interactive mode
python inference.py

# Test mode (no trained model required)
python inference.py --test
```

### 3. Evaluate Performance

```bash
python evaluate.py --output_dir outputs
```

This generates:
- `accuracy_vs_retrieval.png` - Comparison chart
- `confidence_entropy.png` - Confidence distribution
- `retrieval_distribution.png` - Decision breakdown
- `cost_reduction.png` - Cost savings visualization
- `evaluation_results.json` - Full metrics

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Answer Accuracy** | Correctness of final answers |
| **Retrieval Calls** | Number of retrievals triggered |
| **Avg Docs Retrieved** | Mean documents per retrieval |
| **Cost Reduction** | % reduction vs always-retrieve baseline |
| **Retrieval Rate** | % of queries that triggered retrieval |

## 🧠 Confidence-Based Retrieval Logic

The system uses **entropy** as a proxy for uncertainty:

```
Entropy = -Σ(p × log(p))  over vocabulary

Confidence = 1 - (entropy / max_entropy)
```

**Decision thresholds:**
- `confidence > 0.7` → **No retrieval** (model is confident)
- `0.4 < confidence ≤ 0.7` → **Retrieve 3 docs** (medium uncertainty)
- `confidence ≤ 0.4` → **Retrieve 5 docs** (high uncertainty)

## 🔬 Technical Details

### Custom Transformer (No nn.Transformer)

The transformer is implemented entirely from scratch:

- **Token Embeddings**: Learnable embeddings with √d scaling
- **Positional Embeddings**: Sinusoidal encodings
- **Multi-Head Attention**: Q/K/V projections, scaled dot-product, causal masking
- **Feed-Forward Network**: Two-layer MLP with GELU activation
- **Layer Normalization**: Pre-norm style
- **Residual Connections**: For stable training

### Vector Store

Simple in-memory store with:
- Cosine similarity search
- Mean-pooled embeddings
- Top-k retrieval

## 📈 Expected Results

With proper training, expect:

| Metric | Adaptive RAG | Baseline RAG |
|--------|--------------|--------------|
| Accuracy | ~85-90% | ~85-90% |
| Retrieval Rate | ~50-60% | 100% |
| Cost Reduction | 40-50% | 0% |

The adaptive system maintains similar accuracy while significantly reducing retrieval costs.

## ⚠️ Limitations

1. **Small model**: The mini transformer has limited capacity for complex reasoning
2. **Character-level tokenization**: Could be improved with BPE/WordPiece
3. **Simple dataset**: Real-world performance would require larger corpora
4. **Heuristic thresholds**: Confidence thresholds could be learned

## 🔮 Future Improvements

- [ ] Add learned retrieval decision head (auxiliary loss)
- [ ] Implement BPE tokenization
- [ ] Add cross-attention for retrieved documents
- [ ] Integrate with external embedding models
- [ ] Add latency measurements and optimization
- [ ] Support for streaming generation

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)
- [Self-RAG](https://arxiv.org/abs/2310.11511)

## 📝 License

MIT License - feel free to use and modify for your projects.
