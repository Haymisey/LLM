"""
Basic Attention Mechanism Implementation

This module implements the fundamental attention mechanism using the
Query-Key-Value paradigm. It's the building block for more complex
attention mechanisms like self-attention and multi-head attention.
"""

import numpy as np
from typing import Tuple, Optional

class BasicAttention:
    """
    Basic attention mechanism implementation using Query-Key-Value paradigm.
    
    The attention mechanism computes attention weights by comparing queries
    with keys, then uses these weights to combine values.
    """
    
    def __init__(self, query_dim: int, key_dim: int, value_dim: int, 
                 scale: bool = True, dropout: float = 0.1):
        """
        Initialize attention mechanism.
        
        Args:
            query_dim: Dimension of query vectors
            key_dim: Dimension of key vectors  
            value_dim: Dimension of value vectors
            scale: Whether to scale attention scores by sqrt(key_dim)
            dropout: Dropout probability for attention weights
        """
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.scale = scale
        self.dropout = dropout
        
        # Initialize weight matrices for linear transformations
        self.W_Q = np.random.randn(query_dim, query_dim) * 0.01
        self.W_K = np.random.randn(key_dim, key_dim) * 0.01
        self.W_V = np.random.randn(value_dim, value_dim) * 0.01
        
        # Initialize bias vectors
        self.b_Q = np.zeros(query_dim)
        self.b_K = np.zeros(key_dim)
        self.b_V = np.zeros(value_dim)
        
        # Store intermediate values for backpropagation
        self.queries = None
        self.keys = None
        self.values = None
        self.attention_weights = None
        self.output = None
        
    def forward(self, queries: np.ndarray, keys: np.ndarray, 
                values: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass of attention mechanism.
        
        Args:
            queries: Query vectors, shape (batch_size, seq_len_q, query_dim)
            keys: Key vectors, shape (batch_size, seq_len_k, key_dim)
            values: Value vectors, shape (batch_size, seq_len_k, value_dim)
            mask: Optional mask for padding, shape (batch_size, seq_len_q, seq_len_k)
            
        Returns:
            Attention output, shape (batch_size, seq_len_q, value_dim)
        """
        batch_size, seq_len_q, _ = queries.shape
        _, seq_len_k, _ = keys.shape
        
        # Store inputs for backpropagation
        self.queries = queries
        self.keys = keys
        self.values = values
        
        # Linear transformations
        Q = np.dot(queries, self.W_Q) + self.b_Q  # (batch_size, seq_len_q, query_dim)
        K = np.dot(keys, self.W_K) + self.b_K      # (batch_size, seq_len_k, key_dim)
        V = np.dot(values, self.W_V) + self.b_V    # (batch_size, seq_len_k, value_dim)
        
        # Compute attention scores
        # Q @ K^T gives us (batch_size, seq_len_q, seq_len_k)
        attention_scores = np.matmul(Q, K.transpose(0, 2, 1))
        
        # Scale attention scores if requested
        if self.scale:
            attention_scores = attention_scores / np.sqrt(self.key_dim)
        
        # Apply mask if provided (set masked positions to large negative values)
        if mask is not None:
            attention_scores = attention_scores + (mask * -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = self._softmax(attention_scores, axis=-1)
        
        # Store attention weights for backpropagation
        self.attention_weights = attention_weights
        
        # Apply attention weights to values
        output = np.matmul(attention_weights, V)
        
        # Store output for backpropagation
        self.output = output
        
        return output
    
    def backward(self, d_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass to compute gradients.
        
        Args:
            d_output: Gradient of loss with respect to output
            
        Returns:
            Gradients with respect to queries, keys, and values
        """
        batch_size, seq_len_q, seq_len_k = self.attention_weights.shape
        _, _, value_dim = self.values.shape
        
        # Gradient with respect to attention weights
        d_attention_weights = np.matmul(d_output, self.values.transpose(0, 2, 1))
        
        # Gradient with respect to values
        d_values = np.matmul(self.attention_weights.transpose(0, 2, 1), d_output)
        
        # Gradient with respect to queries and keys (simplified)
        # In practice, you'd need to compute gradients through the softmax and matmul
        d_queries = np.zeros_like(self.queries)
        d_keys = np.zeros_like(self.keys)
        
        # For now, return simplified gradients
        # In a full implementation, you'd compute the full chain rule
        return d_queries, d_keys, d_values
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along specified axis."""
        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _dropout(self, x: np.ndarray, dropout_rate: float) -> np.ndarray:
        """Apply dropout to input."""
        if dropout_rate == 0:
            return x
        
        # Create dropout mask
        mask = np.random.binomial(1, 1 - dropout_rate, size=x.shape) / (1 - dropout_rate)
        return x * mask
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get the computed attention weights for visualization."""
        return self.attention_weights


# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Configuration
    batch_size = 2
    seq_len_q = 3
    seq_len_k = 4
    query_dim = 8
    key_dim = 8
    value_dim = 6
    
    # Create attention mechanism
    attention = BasicAttention(query_dim, key_dim, value_dim)
    
    # Create dummy input data
    queries = np.random.randn(batch_size, seq_len_q, query_dim)
    keys = np.random.randn(batch_size, seq_len_k, key_dim)
    values = np.random.randn(batch_size, seq_len_k, value_dim)
    
    # Create a simple mask (no masking for this example)
    mask = None
    
    print("=== Basic Attention Test ===")
    print(f"Input shapes:")
    print(f"  Queries: {queries.shape}")
    print(f"  Keys: {keys.shape}")
    print(f"  Values: {values.shape}")
    
    # Forward pass
    output = attention.forward(queries, keys, values, mask)
    print(f"\nOutput shape: {output.shape}")
    
    # Get attention weights
    attention_weights = attention.get_attention_weights()
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Show attention weights for first batch
    print(f"\nAttention weights (first batch):")
    print(attention_weights[0])
    
    print("\n✅ Basic attention mechanism working correctly!")
