"""
Training Script for Mini Transformer.

Trains the transformer on document corpus using language modeling objective.
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model.transformer import MiniTransformer, MiniTransformerConfig


class CharTokenizer:
    """
    Simple character-level tokenizer.
    
    For simplicity, we use character-level tokenization.
    This can be upgraded to BPE for better performance.
    """
    
    def __init__(self, vocab_size: int = 256):
        """
        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        
        # Special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.bos_token = '<BOS>'
        self.eos_token = '<EOS>'
        
        # Initialize with special tokens
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        for i, token in enumerate(self.special_tokens):
            self.char_to_id[token] = i
            self.id_to_char[i] = token
        
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self.next_id = len(self.special_tokens)
    
    def fit(self, texts: List[str]):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
        """
        char_counts = {}
        for text in texts:
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1
        
        # Sort by frequency and take top vocab_size - special_tokens
        sorted_chars = sorted(char_counts.items(), key=lambda x: -x[1])
        max_chars = self.vocab_size - len(self.special_tokens)
        
        for char, _ in sorted_chars[:max_chars]:
            if char not in self.char_to_id:
                self.char_to_id[char] = self.next_id
                self.id_to_char[self.next_id] = char
                self.next_id += 1
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
        
        Returns:
            List of token IDs
        """
        ids = []
        if add_special_tokens:
            ids.append(self.bos_id)
        
        for char in text:
            ids.append(self.char_to_id.get(char, self.unk_id))
        
        if add_special_tokens:
            ids.append(self.eos_id)
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
        
        Returns:
            Decoded text
        """
        chars = []
        for id in ids:
            if id in self.id_to_char:
                token = self.id_to_char[id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                chars.append(token)
        return ''.join(chars)
    
    def save(self, path: str):
        """Save tokenizer to file."""
        data = {
            'char_to_id': self.char_to_id,
            'id_to_char': {int(k): v for k, v in self.id_to_char.items()},
            'vocab_size': self.vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'CharTokenizer':
        """Load tokenizer from file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tokenizer = cls(data['vocab_size'])
        tokenizer.char_to_id = data['char_to_id']
        tokenizer.id_to_char = {int(k): v for k, v in data['id_to_char'].items()}
        tokenizer.next_id = max(int(k) for k in data['id_to_char'].keys()) + 1
        return tokenizer
    
    def __len__(self) -> int:
        return len(self.char_to_id)


class TextDataset(Dataset):
    """Dataset for language modeling."""
    
    def __init__(
        self, 
        texts: List[str], 
        tokenizer: CharTokenizer, 
        max_length: int = 256
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Tokenize all texts
        for text in texts:
            ids = tokenizer.encode(text)
            # Create sliding windows
            for i in range(0, len(ids) - 1, max_length // 2):
                chunk = ids[i:i + max_length]
                if len(chunk) >= 10:  # Minimum sequence length
                    self.samples.append(chunk)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get input and target for language modeling.
        
        Returns:
            Tuple of (input_ids, target_ids)
        """
        ids = self.samples[idx]
        
        # Pad to max_length
        if len(ids) < self.max_length:
            ids = ids + [self.tokenizer.pad_id] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]
        
        # Input: all tokens except last
        # Target: all tokens except first (shifted by 1)
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        
        return input_ids, target_ids


def load_training_data(data_dir: str) -> Tuple[List[str], List[Dict]]:
    """
    Load training data from files.
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        Tuple of (documents, qa_data)
    """
    # Load documents
    doc_path = os.path.join(data_dir, 'documents.txt')
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    documents = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    # Load Q&A data
    qa_path = os.path.join(data_dir, 'qa.json')
    with open(qa_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)['questions']
    
    return documents, qa_data


def prepare_training_texts(documents: List[str], qa_data: List[Dict]) -> List[str]:
    """
    Prepare training texts combining documents and Q&A.
    
    Args:
        documents: List of document texts
        qa_data: List of Q&A dictionaries
    
    Returns:
        List of training texts
    """
    texts = documents.copy()
    
    # Add Q&A as training text
    for qa in qa_data:
        q = qa['question']
        a = qa['answer']
        texts.append(f"Question: {q}\nAnswer: {a}")
    
    return texts


class Trainer:
    """Trainer class for Mini Transformer."""
    
    def __init__(
        self,
        model: MiniTransformer,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=len(train_loader) * 10
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for batch_idx, (input_ids, target_ids) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, _, _ = self.model(input_ids)
            
            # Flatten for cross entropy
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0  # Ignore padding
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate the model."""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0.0
        
        for input_ids, target_ids in self.val_loader:
            input_ids = input_ids.to(self.device)
            target_ids = target_ids.to(self.device)
            
            logits, _, _ = self.model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_ids.view(-1),
                ignore_index=0
            )
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        return checkpoint['epoch']


def main():
    parser = argparse.ArgumentParser(description='Train Mini Transformer')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256, help='Max sequence length')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading training data...")
    documents, qa_data = load_training_data(args.data_dir)
    texts = prepare_training_texts(documents, qa_data)
    print(f"Loaded {len(documents)} documents and {len(qa_data)} Q&A pairs")
    
    print("Building tokenizer...")
    tokenizer = CharTokenizer(vocab_size=256)
    tokenizer.fit(texts)
    tokenizer.save(os.path.join(args.output_dir, 'tokenizer.json'))
    print(f"Vocabulary size: {len(tokenizer)}")
    
    print("Creating dataset...")
    dataset = TextDataset(texts, tokenizer, max_length=args.max_length)
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    print("Creating model...")
    config = MiniTransformerConfig(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_length
    )
    model = MiniTransformer(**config.to_dict())
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Save config
    import json
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate
    )
    
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume) + 1
    
    print(f"Training on {trainer.device}...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        train_loss = trainer.train_epoch(epoch)
        val_loss = trainer.validate()
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save checkpoint
        trainer.save_checkpoint(
            os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt'),
            epoch
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_checkpoint(
                os.path.join(args.output_dir, 'best_model.pt'),
                epoch
            )
            print(f"New best model saved (val_loss: {val_loss:.4f})")
    
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
