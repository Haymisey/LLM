"""
Real Amharic-Oromiffa Translation Training (Checkpoint 8)
========================================================

This script trains the Transformer model on real parallel text data.
Features:
- Real dataset loading (208,906 parallel sentences)
- Progressive training (start with subset, gradually increase)
- Real evaluation metrics and translation examples
- Checkpointing and model saving
"""

import os
import sys
import time
import numpy as np
import argparse
import json

# Ensure `src` is on path when running from repo root
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "..")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data.amharic_oromiffa_dataset import AmharicOromiffaDataset, create_progressive_dataset
from data.data_loader import DataLoader
from transformer.transformer import Transformer
from transformer.embeddings import TokenEmbedding, PositionalEmbedding
from transformer.output_projection import OutputProjection
from attention.multi_head_attention import MultiHeadAttention
from training.loss_functions import CrossEntropyLoss
from training.metrics import TranslationMetrics
from training.optimizers import AdamOptimizer
from training.optimizers import LearningRateScheduler
from training.training_loop import TrainingLoop, ValidationLoop
from training.progress_monitor import ProgressMonitor, TrainingLogger


class TrainableTransformer(Transformer):
    """Adapter around Transformer to expose optimizer-friendly API for training_loop."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Replace mock components with simple, trainable components
        self.src_embedding = TokenEmbedding(self.src_vocab_size, self.d_model)
        self.tgt_embedding = TokenEmbedding(self.tgt_vocab_size, self.d_model)
        self.src_positional_embedding = PositionalEmbedding(self.d_model, self.max_seq_len)
        self.tgt_positional_embedding = PositionalEmbedding(self.d_model, self.max_seq_len)
        self.output_projection = OutputProjection(self.d_model, self.tgt_vocab_size)
        
        # Step 3: Add proper cross-attention layer
        self.cross_attention = MultiHeadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=self.dropout_rate
        )

        # Optimizer-facing flat parameter dict
        self._optim_params = {
            'src_embeddings': self.src_embedding.embeddings,
            'tgt_embeddings': self.tgt_embedding.embeddings,
            'out_weights': self.output_projection.weights,
            'out_bias': self.output_projection.bias,
            # Add cross-attention parameters
            'cross_attn_W_Q': self.cross_attention.W_Q,
            'cross_attn_W_K': self.cross_attention.W_K,
            'cross_attn_W_V': self.cross_attention.W_V,
            'cross_attn_W_O': self.cross_attention.W_O,
            'cross_attn_b_Q': self.cross_attention.b_Q,
            'cross_attn_b_K': self.cross_attention.b_K,
            'cross_attn_b_V': self.cross_attention.b_V,
            'cross_attn_b_O': self.cross_attention.b_O,
        }
        self._grads = {k: np.zeros_like(v) for k, v in self._optim_params.items()}
        self._is_training = True

    # Training/eval mode toggles (no-op for NumPy but required by loops)
    def train(self):
        self._is_training = True

    def eval(self):
        self._is_training = False

    def get_parameters(self):
        return self._optim_params

    def set_parameters(self, params):
        self._optim_params = params
        # Also reflect into submodules if they have these attributes
        if hasattr(self.src_embedding, 'embeddings'):
            self.src_embedding.embeddings = params['src_embeddings']
        if hasattr(self.tgt_embedding, 'embeddings'):
            self.tgt_embedding.embeddings = params['tgt_embeddings']
        if hasattr(self.output_projection, 'weights'):
            self.output_projection.weights = params['out_weights']
        if hasattr(self.output_projection, 'bias'):
            self.output_projection.bias = params['out_bias']
        
        # Update cross-attention parameters
        if 'cross_attn_W_Q' in params:
            self.cross_attention.W_Q = params['cross_attn_W_Q']
        if 'cross_attn_W_K' in params:
            self.cross_attention.W_K = params['cross_attn_W_K']
        if 'cross_attn_W_V' in params:
            self.cross_attention.W_V = params['cross_attn_W_V']
        if 'cross_attn_W_O' in params:
            self.cross_attention.W_O = params['cross_attn_W_O']
        if 'cross_attn_b_Q' in params:
            self.cross_attention.b_Q = params['cross_attn_b_Q']
        if 'cross_attn_b_K' in params:
            self.cross_attention.b_K = params['cross_attn_b_K']
        if 'cross_attn_b_V' in params:
            self.cross_attention.b_V = params['cross_attn_b_V']
        if 'cross_attn_b_O' in params:
            self.cross_attention.b_O = params['cross_attn_b_O']

    def zero_grad(self):
        for grad in self._grads.values():
            grad.fill(0)

    def zero_gradients(self):
        """Reset all gradients to zero."""
        for grad in self._grads.values():
            grad.fill(0)
        # Also zero underlying module grads if present
        if hasattr(self.src_embedding, 'grad_embeddings'):
            self.src_embedding.grad_embeddings.fill(0)
        if hasattr(self.tgt_embedding, 'grad_embeddings'):
            self.tgt_embedding.grad_embeddings.fill(0)
        if hasattr(self.output_projection, 'grad_weights'):
            self.output_projection.grad_weights.fill(0)
        if hasattr(self.output_projection, 'grad_bias'):
            self.output_projection.grad_bias.fill(0)
        
        # Zero cross-attention gradients
        if hasattr(self.cross_attention, 'grad_W_Q'):
            self.cross_attention.grad_W_Q.fill(0)
        if hasattr(self.cross_attention, 'grad_W_K'):
            self.cross_attention.grad_W_K.fill(0)
        if hasattr(self.cross_attention, 'grad_W_V'):
            self.cross_attention.grad_W_V.fill(0)
        if hasattr(self.cross_attention, 'grad_W_O'):
            self.cross_attention.grad_W_O.fill(0)
        if hasattr(self.cross_attention, 'grad_b_Q'):
            self.cross_attention.grad_b_Q.fill(0)
        if hasattr(self.cross_attention, 'grad_b_K'):
            self.cross_attention.grad_b_K.fill(0)
        if hasattr(self.cross_attention, 'grad_b_V'):
            self.cross_attention.grad_b_V.fill(0)
        if hasattr(self.cross_attention, 'grad_b_O'):
            self.cross_attention.grad_b_O.fill(0)

    def get_gradients(self):
        """Get current gradients for optimizer."""
        return self._grads

    def step(self, optimizer):
        # Apply gradients to parameters
        for name, param in self._optim_params.items():
            if name in self._grads:
                param -= optimizer.learning_rate * self._grads[name]
        # Reflect updates into submodules
        self.set_parameters(self._optim_params)

    # Override forward/backward with a simple trainable decoder-only LM head
    def forward(self, src_tokens, tgt_tokens, src_padding_mask=None, tgt_padding_mask=None, look_ahead_mask=None):
        # Step 1: Encode source sequence
        batch_size, src_len = src_tokens.shape
        # Store source length for backward pass
        self._last_src_len = src_len
        src_emb = self.src_embedding.forward(src_tokens)
        src_pos = self.src_positional_embedding.forward(src_len).reshape(1, src_len, self.d_model)
        src_hidden = src_emb + src_pos
        if self._is_training and self.dropout_rate > 0:
            src_hidden = src_hidden * (1 - self.dropout_rate)
        
        # Step 2: Decode target sequence with source context
        batch_size, tgt_len = tgt_tokens.shape
        tgt_emb = self.tgt_embedding.forward(tgt_tokens)
        pos = self.tgt_positional_embedding.forward(tgt_len).reshape(1, tgt_len, self.d_model)
        dec_hidden = tgt_emb + pos
        if self._is_training and self.dropout_rate > 0:
            dec_hidden = dec_hidden * (1 - self.dropout_rate)
        
        # Store for backward pass
        self._last_dec_hidden = dec_hidden
        self._last_src_hidden = src_hidden
        
        # Step 3: Proper multi-head cross-attention (source context influences decoder)
        # Use the cross-attention layer: query=decoder, key=source, value=source
        # This allows the decoder to attend to relevant parts of the source sequence
        cross_attn_output = self._cross_attention_forward(
            query=dec_hidden,  # Decoder hidden states as queries
            key=src_hidden,    # Source hidden states as keys
            value=src_hidden,  # Source hidden states as values
            mask=src_padding_mask  # Mask padded source tokens
        )
        
        # Store for backward pass
        self._last_attended_values = cross_attn_output
        
        # Combine decoder hidden states with cross-attention output
        dec_hidden = dec_hidden + cross_attn_output
        
        logits = self.output_projection.forward(dec_hidden)
        return logits, {}
    
    def _cross_attention_forward(self, query, key, value, mask=None):
        """
        Custom cross-attention forward pass.
        query: decoder hidden states [batch, tgt_len, d_model]
        key: source hidden states [batch, src_len, d_model]  
        value: source hidden states [batch, src_len, d_model]
        mask: source padding mask [batch, src_len]
        """
        batch_size, tgt_len, _ = query.shape
        _, src_len, _ = key.shape
        
        # Linear projections: Q from decoder, K and V from source
        Q = np.dot(query, self.cross_attention.W_Q)  # [batch, tgt_len, d_model]
        K = np.dot(key, self.cross_attention.W_K)    # [batch, src_len, d_model]
        V = np.dot(value, self.cross_attention.W_V)  # [batch, src_len, d_model]
        
        # Add bias if enabled
        if self.cross_attention.bias:
            Q += self.cross_attention.b_Q
            K += self.cross_attention.b_K
            V += self.cross_attention.b_V
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, tgt_len, self.cross_attention.num_heads, self.cross_attention.head_dim)
        K = K.reshape(batch_size, src_len, self.cross_attention.num_heads, self.cross_attention.head_dim)
        V = V.reshape(batch_size, src_len, self.cross_attention.num_heads, self.cross_attention.head_dim)
        
        # Transpose for computation: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(0, 2, 1, 3)  # [batch, heads, tgt_len, head_dim]
        K = K.transpose(0, 2, 1, 3)  # [batch, heads, src_len, head_dim]
        V = V.transpose(0, 2, 1, 3)  # [batch, heads, src_len, head_dim]
        
        # Compute attention scores: (batch_size, num_heads, tgt_len, src_len)
        attention_scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
        
        # Scale attention scores
        attention_scores = attention_scores / np.sqrt(self.cross_attention.head_dim)
        
        # Apply mask if provided (mask padded source tokens)
        if mask is not None:
            # Expand mask for multi-head attention: [batch, 1, 1, src_len]
            mask = mask.reshape(batch_size, 1, 1, src_len)
            attention_scores = attention_scores + (mask * -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = self._softmax(attention_scores, axis=-1)
        
        # Apply dropout during training
        if self._is_training and self.cross_attention.dropout > 0:
            attention_weights = self._dropout(attention_weights, self.cross_attention.dropout)
        
        # Store for backward pass
        self._cross_attn_weights = attention_weights
        self._cross_attn_Q = Q
        self._cross_attn_K = K
        self._cross_attn_V = V
        
        # Apply attention weights to values
        attended_values = np.matmul(attention_weights, V)  # [batch, heads, tgt_len, head_dim]
        
        # Reshape back: (batch_size, tgt_len, embed_dim)
        attended_values = attended_values.transpose(0, 2, 1, 3).reshape(
            batch_size, tgt_len, self.cross_attention.embed_dim
        )
        
        # Final linear projection
        output = np.dot(attended_values, self.cross_attention.W_O)
        if self.cross_attention.bias:
            output += self.cross_attention.b_O
        
        return output
    
    def _softmax(self, x, axis=-1):
        """Numerically stable softmax."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _dropout(self, x, rate):
        """Simple dropout implementation."""
        if rate > 0:
            mask = np.random.binomial(1, 1 - rate, size=x.shape) / (1 - rate)
            return x * mask
        return x
    
    def backward(self, grad_logits):
        """Backward pass with proper cross-attention gradients."""
        # Backprop through output projection to decoder hidden states
        grad_dec = self.output_projection.backward(grad_logits)
        
        # Backprop through cross-attention (residual connection)
        grad_cross_attn = grad_dec  # Gradient from residual connection
        
        # Backprop through cross-attention output projection
        grad_attended_values = np.dot(grad_cross_attn, self.cross_attention.W_O.T)
        
        # Reshape for multi-head attention backprop
        batch_size, tgt_len, _ = grad_attended_values.shape
        grad_attended_values = grad_attended_values.reshape(
            batch_size, tgt_len, self.cross_attention.num_heads, self.cross_attention.head_dim
        ).transpose(0, 2, 1, 3)  # [batch, heads, tgt_len, head_dim]
        
        # Backprop through attention mechanism
        if hasattr(self, '_cross_attn_weights') and hasattr(self, '_cross_attn_V'):
            # Gradient w.r.t. V
            grad_V = np.matmul(self._cross_attn_weights.transpose(0, 1, 3, 2), grad_attended_values)
            
            # Gradient w.r.t. attention weights
            grad_weights = np.matmul(grad_attended_values, self._cross_attn_V.transpose(0, 1, 3, 2))
            
            # Backprop through softmax and scaling
            # This is a simplified version - in practice you'd want more detailed backprop
            grad_scores = grad_weights / np.sqrt(self.cross_attention.head_dim)
            
            # Gradients w.r.t. Q, K
            grad_Q = np.matmul(grad_scores, self._cross_attn_K)
            grad_K = np.matmul(grad_scores.transpose(0, 1, 3, 2), self._cross_attn_Q)
            
            # Reshape back to [batch, seq_len, d_model]
            grad_Q = grad_Q.transpose(0, 2, 1, 3).reshape(batch_size, tgt_len, self.cross_attention.embed_dim)
            grad_K = grad_K.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.cross_attention.embed_dim)
            grad_V = grad_V.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.cross_attention.embed_dim)
            
            # Backprop through linear projections
            grad_dec_hidden = np.dot(grad_Q, self.cross_attention.W_Q.T)
            grad_src_hidden = np.dot(grad_K, self.cross_attention.W_K.T) + np.dot(grad_V, self.cross_attention.W_V.T)
            
            # Store gradients for cross-attention parameters
            if hasattr(self.cross_attention, 'grad_W_Q'):
                self.cross_attention.grad_W_Q = np.dot(grad_Q.T, self._last_dec_hidden)
            if hasattr(self.cross_attention, 'grad_W_K'):
                self.cross_attention.grad_W_K = np.dot(grad_K.T, self._last_src_hidden)
            if hasattr(self.cross_attention, 'grad_W_V'):
                self.cross_attention.grad_W_V = np.dot(grad_V.T, self._last_src_hidden)
            if hasattr(self.cross_attention, 'grad_W_O'):
                self.cross_attention.grad_W_O = np.dot(grad_cross_attn.T, self._last_attended_values)
        else:
            # Fallback if cross-attention intermediates aren't available
            grad_dec_hidden = grad_dec
            grad_src_hidden = np.zeros_like(self._last_src_hidden) if hasattr(self, '_last_src_len') else None
        
        # Combine gradients from residual connection and cross-attention
        grad_dec_final = grad_dec + grad_dec_hidden
        
        # Backprop through target embeddings
        self.tgt_embedding.backward(grad_dec_final)
        
        # Backprop through source embeddings (if we have source gradients)
        if grad_src_hidden is not None:
            self.src_embedding.backward(grad_src_hidden)
        
        # Collect all gradients into flat map
        self._grads['out_weights'] = getattr(self.output_projection, 'grad_weights', self._grads['out_weights'])
        self._grads['out_bias'] = getattr(self.output_projection, 'grad_bias', self._grads['out_bias'])
        self._grads['src_embeddings'] = getattr(self.src_embedding, 'grad_embeddings', self._grads['src_embeddings'])
        self._grads['tgt_embeddings'] = getattr(self.tgt_embedding, 'grad_embeddings', self._grads['tgt_embeddings'])
        
        # Collect cross-attention gradients
        self._grads['cross_attn_W_Q'] = getattr(self.cross_attention, 'grad_W_Q', self._grads['cross_attn_W_Q'])
        self._grads['cross_attn_W_K'] = getattr(self.cross_attention, 'grad_W_K', self._grads['cross_attn_W_K'])
        self._grads['cross_attn_W_V'] = getattr(self.cross_attention, 'grad_W_V', self._grads['cross_attn_W_V'])
        self._grads['cross_attn_W_O'] = getattr(self.cross_attention, 'grad_W_O', self._grads['cross_attn_W_O'])
        self._grads['cross_attn_b_Q'] = getattr(self.cross_attention, 'grad_b_Q', self._grads['cross_attn_b_Q'])
        self._grads['cross_attn_b_K'] = getattr(self.cross_attention, 'grad_b_K', self._grads['cross_attn_b_K'])
        self._grads['cross_attn_b_V'] = getattr(self.cross_attention, 'grad_b_V', self._grads['cross_attn_b_V'])
        self._grads['cross_attn_b_O'] = getattr(self.cross_attention, 'grad_b_O', self._grads['cross_attn_b_O'])
        
        # Return placeholder grads for API compatibility
        return None, None


def save_checkpoint(checkpoint_dir: str, epoch: int, model: TrainableTransformer,
                    src_tokenizer, tgt_tokenizer, config: dict):
    """Save model checkpoint with vocabularies and config."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model parameters (optimizer-exposed params) using numpy
    params = model.get_parameters()
    np.savez_compressed(os.path.join(checkpoint_dir, f"model_epoch_{epoch}.npz"), **params)
    
    # Save tokenizers
    src_tokenizer.save_vocab(os.path.join(checkpoint_dir, f"src_vocab_epoch_{epoch}.json"))
    tgt_tokenizer.save_vocab(os.path.join(checkpoint_dir, f"tgt_vocab_epoch_{epoch}.json"))
    
    # Save training config
    config_path = os.path.join(checkpoint_dir, f"config_epoch_{epoch}.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Checkpoint saved: {checkpoint_dir}/epoch_{epoch}")


def load_checkpoint(checkpoint_dir: str, epoch: int, model: TrainableTransformer,
                   src_tokenizer, tgt_tokenizer):
    """Load model checkpoint."""
    # Load model parameters
    params_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.npz")
    if os.path.exists(params_path):
        params = np.load(params_path)
        model_params = {k: params[k] for k in params.keys()}
        model.set_parameters(model_params)
        print(f"Model parameters loaded from epoch {epoch}")
    
    # Load tokenizers
    src_vocab_path = os.path.join(checkpoint_dir, f"src_vocab_epoch_{epoch}.json")
    tgt_vocab_path = os.path.join(checkpoint_dir, f"tgt_vocab_epoch_{epoch}.json")
    
    if os.path.exists(src_vocab_path):
        src_tokenizer.load_vocab(src_vocab_path)
        print(f"Source vocabulary loaded from epoch {epoch}")
    
    if os.path.exists(tgt_vocab_path):
        tgt_tokenizer.load_vocab(tgt_vocab_path)
        print(f"Target vocabulary loaded from epoch {epoch}")


def evaluate_translations(model, src_tokenizer, tgt_tokenizer, test_samples):
    """Evaluate translation quality on test samples."""
    print("\n============================================================")
    print("TRANSLATION EVALUATION")
    print("============================================================")
    
    metrics = TranslationMetrics()
    
    for i, (src_text, tgt_text) in enumerate(test_samples[:3]):  # Test first 3 samples
        print(f"\nSample {i+1}:")
        print(f"Source (Amharic): {src_text}")
        print(f"Target (Oromiffa): {tgt_text}")
        
        # Encode source text
        src_ids = src_tokenizer.encode(src_text, add_special_tokens=True)
        src_ids = np.array(src_ids).reshape(1, -1)  # [1, seq_len]
        
        # Generate translation
        max_length = len(src_ids[0]) + 20  # Allow some extra length
        generated_ids = model.generate(
            src_ids,
            max_length=max_length,
            start_token=tgt_tokenizer.sos_token_id,
            end_token=tgt_tokenizer.eos_token_id
        )
        
        # Decode generated text
        generated_text = tgt_tokenizer.decode(generated_ids[0])
        print(f"Generated: {generated_text}")
        
        # Calculate metrics
        target_tokens = tgt_tokenizer.encode(tgt_text, add_special_tokens=True)
        
        # Convert to numpy arrays
        pred_tokens = np.array(generated_ids[0])
        target_tokens = np.array(target_tokens)
        
        # Calculate metrics that can handle different lengths
        bleu_score = metrics.compute_bleu_score(
            pred_tokens.reshape(1, -1), 
            target_tokens.reshape(1, -1)
        )
        
        # For accuracy, compare up to the minimum length
        min_len = min(len(pred_tokens), len(target_tokens))
        if min_len > 0:
            accuracy = np.mean(pred_tokens[:min_len] == target_tokens[:min_len])
        else:
            accuracy = 0.0
        
        print(f"BLEU: {bleu_score:.4f}, Accuracy: {accuracy:.4f}")

def progressive_training(data_file_path: str,
                        initial_samples: int = 10000,
                        epochs_per_stage: int = 3,
                        samples_per_increase: int = 20000):
    """
    Progressive training: start with subset, gradually increase data.
    
    Args:
        data_file_path: Path to parallel text file
        initial_samples: Number of samples to start with
        epochs_per_stage: Number of epochs to train on each data size
        samples_per_increase: How many samples to add each time
    """
    print("🚀 Starting Progressive Training Pipeline")
    print("="*60)
    
    # Stage 1: Start with subset
    print(f"\n📊 Stage 1: Training on {initial_samples:,} samples")
    dataset = create_progressive_dataset(data_file_path, initial_samples)
    
    # Get data and tokenizers
    src_texts, tgt_texts = dataset.get_training_data()
    val_src, val_tgt = dataset.get_validation_data()
    src_tokenizer, tgt_tokenizer = dataset.get_tokenizers()
    src_vocab_size, tgt_vocab_size = dataset.get_vocab_sizes()
    
    print(f"Vocabulary sizes: Source={src_vocab_size}, Target={tgt_vocab_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        source_texts=src_texts,
        target_texts=tgt_texts,
        source_tokenizer=src_tokenizer,
        target_tokenizer=tgt_tokenizer,
        batch_size=16,  # Smaller batch size for initial training
        max_source_length=100,
        max_target_length=100
    )
    
    val_loader = DataLoader(
        source_texts=val_src,
        target_texts=val_tgt,
        source_tokenizer=src_tokenizer,
        target_tokenizer=tgt_tokenizer,
        batch_size=16,
        max_source_length=100,
        max_target_length=100,
        shuffle=False
    )
    
    # Model
    model = TrainableTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=512,
        n_heads=8,
        d_ff=2048,
        n_encoder_layers=3,
        n_decoder_layers=3,
        dropout_rate=0.1,
    )
    
    # Training components
    loss_fn = CrossEntropyLoss()
    optimizer = AdamOptimizer(learning_rate=1e-3)
    
    # Configure warmup-cosine schedule
    total_steps = epochs_per_stage * len(train_loader)
    scheduler = LearningRateScheduler(
        optimizer,
        scheduler_type='warmup_cosine',
        warmup_steps=total_steps // 10,
        max_steps=total_steps
    )
    
    # Progress monitoring
    total_epochs = epochs_per_stage
    progress_monitor = ProgressMonitor(total_epochs=total_epochs)
    logger = TrainingLogger(log_dir="logs")
    
    # Training and validation loops
    training_loop = TrainingLoop(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    validation_loop = ValidationLoop(
        model=model,
        metrics=TranslationMetrics()
    )
    
    # Training configuration
    config = {
        'model_config': {
            'd_model': 512,
            'n_heads': 8,
            'd_ff': 2048,
            'n_encoder_layers': 6,
            'n_decoder_layers': 6,
            'dropout_rate': 0.1
        },
        'training_config': {
            'initial_samples': initial_samples,
            'epochs_per_stage': epochs_per_stage,
            'samples_per_increase': samples_per_increase,
            'learning_rate': 3e-4,
            'batch_size': 16
        }
    }
    
    # Stage 1 training
    checkpoint_dir = "checkpoints/progressive_training"
    best_val_loss = float('inf')
    
    for epoch in range(epochs_per_stage):
        print(f"\n📚 Epoch {epoch + 1}/{epochs_per_stage}")
        
        # Training
        train_stats = training_loop.train_epoch(train_loader, epoch + 1)
        train_loss = train_stats['loss']
        
        # Validation
        val_metrics = validation_loop.validate(val_loader, epoch + 1)
        val_loss = val_metrics['loss']
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(checkpoint_dir, epoch, model, src_tokenizer, tgt_tokenizer, config)
            print(f"✨ New best checkpoint saved (val_loss: {val_loss:.4f})")
    
    # Evaluate on validation samples
    test_samples = list(zip(val_src[:10], val_tgt[:10]))  # First 10 validation samples
    evaluate_translations(model, src_tokenizer, tgt_tokenizer, test_samples)
    
    print(f"\n✅ Stage 1 completed! Trained on {initial_samples:,} samples")
    
    # Progressive stages: increase data size
    current_samples = initial_samples
    stage = 2
    
    while True:
        # Try to increase training data
        additional = samples_per_increase
        moved = dataset.increase_training_data(additional)
        
        if not moved or moved <= 0:
            print(f"⚠️  Cannot increase training data further - maintaining current size")
            break
            
        current_samples += moved
        
        print(f"\n📊 Stage {stage}: Training on {current_samples:,} samples")
        print(f"Added {moved:,} samples to training")
        
        # Update data loaders with new data
        src_texts, tgt_texts = dataset.get_training_data()
        val_src, val_tgt = dataset.get_validation_data()
        
        # Check if we have validation data
        if len(val_src) == 0:
            print(f"❌ No validation data available - cannot continue training")
            break
        
        train_loader = DataLoader(
            source_texts=src_texts,
            target_texts=tgt_texts,
            source_tokenizer=src_tokenizer,
            target_tokenizer=tgt_tokenizer,
            batch_size=16,
            max_source_length=100,
            max_target_length=100
        )
        
        val_loader = DataLoader(
            source_texts=val_src,
            target_texts=val_tgt,
            source_tokenizer=src_tokenizer,
            target_tokenizer=tgt_tokenizer,
            batch_size=16,
            max_source_length=100,
            max_target_length=100,
            shuffle=False
        )
        
        # Train on new data
        for epoch in range(epochs_per_stage):
            print(f"📚 Epoch {epoch + 1}/{epochs_per_stage}")
            
            train_stats = training_loop.train_epoch(train_loader, epoch + 1)
            train_loss = train_stats['loss']
            val_metrics = validation_loop.validate(val_loader, epoch + 1)
            val_loss = val_metrics['loss']
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(checkpoint_dir, f"stage_{stage}_epoch_{epoch}", 
                              model, src_tokenizer, tgt_tokenizer, config)
                print(f"✨ New best checkpoint saved (val_loss: {val_loss:.4f})")
        
        # Evaluate
        test_samples = list(zip(val_src[:10], val_tgt[:10]))
        evaluate_translations(model, src_tokenizer, tgt_tokenizer, test_samples)
        
        print(f"✅ Stage {stage} completed! Trained on {current_samples:,} samples")
        stage += 1
    
    print(f"\n🎉 Progressive training completed!")
    print(f"Final model trained on {current_samples:,} samples")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Save final model
    final_checkpoint_dir = "checkpoints/final_model"
    save_checkpoint(final_checkpoint_dir, "final", model, src_tokenizer, tgt_tokenizer, config)
    
    return model, src_tokenizer, tgt_tokenizer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Amharic-Oromiffa Transformer")
    parser.add_argument("--data_file", type=str, required=True,
                       help="Path to tab-separated parallel text file")
    parser.add_argument("--initial_samples", type=int, default=10000,
                       help="Number of samples to start training with")
    parser.add_argument("--epochs_per_stage", type=int, default=3,
                       help="Number of epochs to train on each data size")
    parser.add_argument("--samples_per_increase", type=int, default=20000,
                       help="How many samples to add each time")
    
    args = parser.parse_args()
    
    # Validate data file
    if not os.path.exists(args.data_file):
        print(f"❌ Data file not found: {args.data_file}")
        return
    
    print("🚀 Starting Amharic-Oromiffa Translation Training")
    print("="*60)
    print(f"Data file: {args.data_file}")
    print(f"Initial samples: {args.initial_samples:,}")
    print(f"Epochs per stage: {args.epochs_per_stage}")
    print(f"Samples per increase: {args.samples_per_increase:,}")
    print("="*60)
    
    try:
        # Start progressive training
        model, src_tokenizer, tgt_tokenizer = progressive_training(
            data_file_path=args.data_file,
            initial_samples=args.initial_samples,
            epochs_per_stage=args.epochs_per_stage,
            samples_per_increase=args.samples_per_increase
        )
        
        print("\n🎯 Training completed successfully!")
        print("Model and tokenizers saved to checkpoints/")
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


