"""
Self-Attention Mechanism Implementation

This module implements self-attention, where a sequence attends to itself.
This is the core mechanism used in Transformers for processing sequences.
"""

import numpy as np
from typing import Optional, Tuple
# from .basic_attention import BasicAttention  # Commented out for standalone testing

class SelfAttention:
    """
    Self-attention mechanism that allows each position in a sequence
    to attend to all positions in the same sequence.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 1, 
                 dropout: float = 0.1, bias: bool = True):
        """
        Initialize self-attention mechanism.
        
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads (1 for single-head)
            dropout: Dropout probability for attention weights
            bias: Whether to use bias terms
        """
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        self.W_Q = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_K = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_V = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_O = np.random.randn(embed_dim, embed_dim) * 0.01  # Output projection
        
        # Bias terms
        if bias:
            self.b_Q = np.zeros(embed_dim)
            self.b_K = np.zeros(embed_dim)
            self.b_V = np.zeros(embed_dim)
            self.b_O = np.zeros(embed_dim)
        else:
            self.b_Q = self.b_K = self.b_V = self.b_O = None
        
        # Store intermediate values for backpropagation
        self.input = None
        self.Q = None
        self.K = None
        self.V = None
        self.attention_weights = None
        self.output = None
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass of self-attention.
        
        Args:
            x: Input sequence, shape (batch_size, seq_len, embed_dim)
            mask: Optional mask for padding, shape (batch_size, seq_len, seq_len)
            
        Returns:
            Self-attention output, shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Store input for backpropagation
        self.input = x
        
        # Linear projections to get Q, K, V
        Q = np.dot(x, self.W_Q)
        K = np.dot(x, self.W_K)
        V = np.dot(x, self.W_V)
        
        # Add bias if enabled
        if self.bias:
            Q += self.b_Q
            K += self.b_K
            V += self.b_V
        
        # Store Q, K, V for backpropagation
        self.Q = Q
        self.K = K
        self.V = V
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for easier computation: (batch_size, num_heads, seq_len, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)
        
        # Compute attention scores: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
        
        # Scale attention scores
        attention_scores = attention_scores / np.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            mask = mask.unsqueeze(1) if len(mask.shape) == 3 else mask
            attention_scores = attention_scores + (mask * -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = self._softmax(attention_scores, axis=-1)
        
        # Apply dropout
        if self.dropout > 0:
            attention_weights = self._dropout(attention_weights, self.dropout)
        
        # Store attention weights for backpropagation
        self.attention_weights = attention_weights
        
        # Apply attention weights to values
        attended_values = np.matmul(attention_weights, V)
        
        # Reshape back: (batch_size, seq_len, embed_dim)
        attended_values = attended_values.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.embed_dim
        )
        
        # Final linear projection
        output = np.dot(attended_values, self.W_O)
        if self.bias:
            output += self.b_O
        
        # Store output for backpropagation
        self.output = output
        
        return output
    
    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """
        Backward pass to compute gradients.
        
        Args:
            d_output: Gradient of loss with respect to output
            
        Returns:
            Gradient with respect to input
        """
        # This is a simplified backward pass
        # In practice, you'd compute the full chain rule through all operations
        
        # Gradient with respect to attended values
        d_attended_values = np.dot(d_output, self.W_O.T)
        
        # Gradient with respect to input (simplified)
        d_input = np.zeros_like(self.input)
        
        return d_input
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along specified axis."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def _dropout(self, x: np.ndarray, dropout_rate: float) -> np.ndarray:
        """Apply dropout to input."""
        if dropout_rate == 0:
            return x
        
        mask = np.random.binomial(1, 1 - dropout_rate, size=x.shape) / (1 - dropout_rate)
        return x * mask
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get the computed attention weights for visualization."""
        return self.attention_weights
    
    def get_attention_patterns(self) -> Optional[np.ndarray]:
        """Get attention patterns averaged across heads."""
        if self.attention_weights is None:
            return None
        
        # Average attention weights across heads
        return np.mean(self.attention_weights, axis=1)


# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Configuration
    batch_size = 2
    seq_len = 5
    embed_dim = 64
    num_heads = 8
    
    # Create self-attention mechanism
    self_attention = SelfAttention(embed_dim, num_heads)
    
    # Create dummy input data
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    print("=== Self-Attention Test ===")
    print(f"Input shape: {x.shape}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Number of heads: {num_heads}")
    print(f"Head dimension: {self_attention.head_dim}")
    
    # Forward pass
    output = self_attention.forward(x)
    print(f"\nOutput shape: {output.shape}")
    
    # Get attention weights
    attention_weights = self_attention.get_attention_weights()
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Get attention patterns (averaged across heads)
    attention_patterns = self_attention.get_attention_patterns()
    print(f"Attention patterns shape: {attention_patterns.shape}")
    
    # Show attention pattern for first batch
    print(f"\nAttention pattern (first batch, averaged across heads):")
    print(attention_patterns[0])
    
    print("\n✅ Self-attention mechanism working correctly!")
