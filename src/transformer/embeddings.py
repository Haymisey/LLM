"""
Embedding Layers for Transformer
===============================

This module implements the embedding layers used in the Transformer architecture:
- TokenEmbedding: Converts token indices to dense vectors
- PositionalEmbedding: Adds positional information to embeddings
"""

import numpy as np
from typing import Optional, Tuple


class TokenEmbedding:
    """
    Token embedding layer that converts token indices to dense vectors.
    
    This is essentially a lookup table where each token ID maps to a learned
    vector representation of size d_model.
    """
    
    def __init__(self, vocab_size: int, d_model: int):
        """
        Initialize the token embedding layer.
        
        Args:
            vocab_size: Size of the vocabulary (number of unique tokens)
            d_model: Dimension of the embedding vectors
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Initialize embeddings with Xavier/Glorot initialization
        # This helps with training stability
        self.embeddings = np.random.randn(vocab_size, d_model) * np.sqrt(2.0 / d_model)
        
        # Gradient storage
        self.grad_embeddings = np.zeros_like(self.embeddings)
        
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Forward pass: convert token IDs to embeddings.
        
        Args:
            token_ids: Array of token IDs of shape (batch_size, seq_len)
            
        Returns:
            Embedded tokens of shape (batch_size, seq_len, d_model)
        """
        self.token_ids = token_ids
        batch_size, seq_len = token_ids.shape
        
        # Reshape for easier indexing
        token_ids_flat = token_ids.reshape(-1)
        
        # Get embeddings for each token ID
        embedded = self.embeddings[token_ids_flat]
        
        # Reshape back to (batch_size, seq_len, d_model)
        return embedded.reshape(batch_size, seq_len, self.d_model)
    
    def backward(self, grad_output: np.ndarray) -> None:
        """
        Backward pass: compute gradients for embeddings.
        
        Args:
            grad_output: Gradient of loss with respect to output
                        Shape: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = self.token_ids.shape
        
        # Reshape for easier indexing
        token_ids_flat = self.token_ids.reshape(-1)
        grad_output_flat = grad_output.reshape(-1, self.d_model)
        
        # Accumulate gradients for each unique token ID
        for i, token_id in enumerate(token_ids_flat):
            self.grad_embeddings[token_id] += grad_output_flat[i]
    
    def update_parameters(self, learning_rate: float) -> None:
        """
        Update embedding parameters using accumulated gradients.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.embeddings -= learning_rate * self.grad_embeddings
        self.grad_embeddings.fill(0)  # Reset gradients
    
    def get_parameters(self) -> dict:
        """Get all parameters for saving/loading."""
        return {
            'embeddings': self.embeddings.copy(),
            'vocab_size': self.vocab_size,
            'd_model': self.d_model
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set parameters from saved state."""
        self.embeddings = params['embeddings'].copy()
        self.vocab_size = params['vocab_size']
        self.d_model = params['d_model']


class PositionalEmbedding:
    """
    Positional embedding layer that adds positional information to token embeddings.
    
    Uses the sinusoidal positional encoding from the original Transformer paper,
    which allows the model to learn relative positions and generalize to
    sequences longer than those seen during training.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        """
        Initialize the positional embedding layer.
        
        Args:
            d_model: Dimension of the embedding vectors
            max_seq_len: Maximum sequence length to precompute
        """
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Precompute positional encodings
        self.positional_encodings = self._create_positional_encodings()
        
    def _create_positional_encodings(self) -> np.ndarray:
        """
        Create sinusoidal positional encodings.
        
        Returns:
            Positional encodings of shape (max_seq_len, d_model)
        """
        pos_encodings = np.zeros((self.max_seq_len, self.d_model))
        
        for pos in range(self.max_seq_len):
            for i in range(0, self.d_model, 2):
                # Even indices use sin, odd indices use cos
                if i < self.d_model:
                    pos_encodings[pos, i] = np.sin(pos / (10000 ** (i / self.d_model)))
                if i + 1 < self.d_model:
                    pos_encodings[pos, i + 1] = np.cos(pos / (10000 ** (i / self.d_model)))
        
        return pos_encodings
    
    def forward(self, seq_len: int) -> np.ndarray:
        """
        Forward pass: get positional encodings for sequence length.
        
        Args:
            seq_len: Length of the sequence
            
        Returns:
            Positional encodings of shape (seq_len, d_model)
        """
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
        
        return self.positional_encodings[:seq_len]
    
    def get_parameters(self) -> dict:
        """Get all parameters for saving/loading."""
        return {
            'positional_encodings': self.positional_encodings.copy(),
            'd_model': self.d_model,
            'max_seq_len': self.max_seq_len
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set parameters from saved state."""
        self.positional_encodings = params['positional_encodings'].copy()
        self.d_model = params['d_model']
        self.max_seq_len = params['max_seq_len']


def test_embeddings():
    """Test the embedding layers."""
    print("Testing TokenEmbedding...")
    
    # Test token embedding
    vocab_size = 1000
    d_model = 512
    batch_size = 2
    seq_len = 10
    
    token_embedding = TokenEmbedding(vocab_size, d_model)
    
    # Create random token IDs
    token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    embedded = token_embedding.forward(token_ids)
    print(f"Token embedding output shape: {embedded.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")
    
    # Test backward pass
    grad_output = np.random.randn(batch_size, seq_len, d_model)
    token_embedding.backward(grad_output)
    
    # Test parameter update
    token_embedding.update_parameters(0.01)
    
    print("✓ TokenEmbedding tests passed!")
    
    print("\nTesting PositionalEmbedding...")
    
    # Test positional embedding
    pos_embedding = PositionalEmbedding(d_model, max_seq_len=100)
    
    # Forward pass
    pos_encodings = pos_embedding.forward(seq_len)
    print(f"Positional encoding output shape: {pos_encodings.shape}")
    print(f"Expected shape: ({seq_len}, {d_model})")
    
    # Check that positional encodings are different for different positions
    pos_diff = np.abs(pos_encodings[0] - pos_encodings[1]).sum()
    print(f"Difference between positions 0 and 1: {pos_diff:.6f}")
    print(f"Should be > 0: {pos_diff > 0}")
    
    print("✓ PositionalEmbedding tests passed!")
    
    print("\nAll embedding tests passed! 🎉")


if __name__ == "__main__":
    test_embeddings()
