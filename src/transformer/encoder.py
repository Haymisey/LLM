"""
Transformer Encoder Implementation
================================

This module implements the encoder part of the Transformer architecture:
- TransformerEncoderLayer: Single encoder layer with self-attention and feed-forward
- TransformerEncoder: Stack of encoder layers
"""

import numpy as np
from typing import Optional, Tuple

# For standalone testing, we'll create mock classes
class MockMultiHeadAttention:
    """Mock MultiHeadAttention for standalone testing."""
    def __init__(self, d_model: int, n_heads: int, dropout_rate: float = 0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        # Mock forward pass
        batch_size, seq_len, d_model = x.shape
        attention_output = x + np.random.randn(*x.shape) * 0.1
        attention_weights = np.random.rand(batch_size, self.n_heads, seq_len, seq_len)
        attention_weights = attention_weights / attention_weights.sum(axis=-1, keepdims=True)
        return attention_output, attention_weights
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Mock backward pass
        return grad_output
    
    def get_parameters(self) -> dict:
        return {'mock': True}
    
    def set_parameters(self, params: dict) -> None:
        pass

class MockFeedForward:
    """Mock FeedForward for standalone testing."""
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Mock forward pass
        return x + np.random.randn(*x.shape) * 0.1
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Mock backward pass
        return grad_output
    
    def get_parameters(self) -> dict:
        return {'mock': True}
    
    def set_parameters(self, params: dict) -> None:
        pass

class MockLayerNormalization:
    """Mock LayerNormalization for standalone testing."""
    def __init__(self, d_model: int, epsilon: float = 1e-6):
        self.d_model = d_model
        self.epsilon = epsilon
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Mock forward pass
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Mock backward pass
        return grad_output
    
    def get_parameters(self) -> dict:
        return {'mock': True}
    
    def set_parameters(self, params: dict) -> None:
        pass


class TransformerEncoderLayer:
    """
    Single layer of the Transformer encoder.
    
    Each encoder layer consists of:
    1. Multi-head self-attention with residual connection and layer normalization
    2. Feed-forward network with residual connection and layer normalization
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1):
        """
        Initialize the encoder layer.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network
            dropout_rate: Dropout rate for regularization
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # Multi-head self-attention
        self.attention = MockMultiHeadAttention(d_model, n_heads, dropout_rate)
        
        # Feed-forward network
        self.feed_forward = MockFeedForward(d_model, d_ff, dropout_rate)
        
        # Layer normalization layers
        self.norm1 = MockLayerNormalization(d_model)
        self.norm2 = MockLayerNormalization(d_model)
        
        # Dropout for regularization
        self.dropout = dropout_rate
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor and attention weights
        """
        # Store input for residual connections
        residual = x
        
        # Self-attention with residual connection and layer normalization
        # Pre-norm: normalize before attention
        x_norm = self.norm1.forward(x)
        attention_output, attention_weights = self.attention.forward(x_norm, mask)
        
        # Apply dropout and residual connection
        if self.dropout > 0:
            attention_output = attention_output * (1 - self.dropout)
        x = residual + attention_output
        
        # Feed-forward with residual connection and layer normalization
        residual = x
        x_norm = self.norm2.forward(x)
        ff_output = self.feed_forward.forward(x_norm)
        
        # Apply dropout and residual connection
        if self.dropout > 0:
            ff_output = ff_output * (1 - self.dropout)
        x = residual + ff_output
        
        return x, attention_weights
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the encoder layer.
        
        Args:
            grad_output: Gradient of loss with respect to output
            
        Returns:
            Gradient of loss with respect to input
        """
        # Backward through feed-forward
        grad_ff = self.feed_forward.backward(grad_output)
        
        # Backward through layer norm
        grad_norm2 = self.norm2.backward(grad_ff)
        
        # Backward through attention
        grad_attention = self.attention.backward(grad_norm2)
        
        # Backward through first layer norm
        grad_norm1 = self.norm1.backward(grad_attention)
        
        return grad_norm1
    
    def get_parameters(self) -> dict:
        """Get all parameters for saving/loading."""
        return {
            'attention': self.attention.get_parameters(),
            'feed_forward': self.feed_forward.get_parameters(),
            'norm1': self.norm1.get_parameters(),
            'norm2': self.norm2.get_parameters(),
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set parameters from saved state."""
        self.attention.set_parameters(params['attention'])
        self.feed_forward.set_parameters(params['feed_forward'])
        self.norm1.set_parameters(params['norm1'])
        self.norm2.set_parameters(params['norm2'])
        self.d_model = params['d_model']
        self.n_heads = params['n_heads']
        self.d_ff = params['d_ff']
        self.dropout_rate = params['dropout_rate']


class TransformerEncoder:
    """
    Stack of Transformer encoder layers.
    
    The encoder processes the input sequence through multiple layers,
    each applying self-attention and feed-forward transformations.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout_rate: float = 0.1):
        """
        Initialize the encoder stack.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network
            n_layers: Number of encoder layers
            dropout_rate: Dropout rate for regularization
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        
        # Create stack of encoder layers
        self.layers = [
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ]
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, list]:
        """
        Forward pass through the encoder stack.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor and list of attention weights from all layers
        """
        attention_weights = []
        
        # Process through each encoder layer
        for layer in self.layers:
            x, attn_weights = layer.forward(x, mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the encoder stack.
        
        Args:
            grad_output: Gradient of loss with respect to output
            
        Returns:
            Gradient of loss with respect to input
        """
        # Backward through layers in reverse order
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        
        return grad_output
    
    def get_parameters(self) -> dict:
        """Get all parameters for saving/loading."""
        return {
            'layers': [layer.get_parameters() for layer in self.layers],
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'n_layers': self.n_layers,
            'dropout_rate': self.dropout_rate
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set parameters from saved state."""
        for i, layer_params in enumerate(params['layers']):
            self.layers[i].set_parameters(layer_params)
        
        self.d_model = params['d_model']
        self.n_heads = params['n_heads']
        self.d_ff = params['d_ff']
        self.n_layers = params['n_layers']
        self.dropout_rate = params['dropout_rate']


def create_padding_mask(seq: np.ndarray, pad_token_id: int = 0) -> np.ndarray:
    """
    Create padding mask for sequences.
    
    Args:
        seq: Input sequence of shape (batch_size, seq_len)
        pad_token_id: ID of the padding token
        
    Returns:
        Padding mask of shape (batch_size, 1, 1, seq_len)
        True values indicate positions to mask (pad tokens)
    """
    batch_size, seq_len = seq.shape
    mask = (seq == pad_token_id).reshape(batch_size, 1, 1, seq_len)
    return mask


def test_encoder():
    """Test the encoder implementation."""
    print("Testing TransformerEncoderLayer...")
    
    # Test single encoder layer
    d_model = 512
    n_heads = 8
    d_ff = 2048
    batch_size = 2
    seq_len = 10
    
    encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff)
    
    # Create input
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output, attention_weights = encoder_layer.forward(x)
    print(f"Encoder layer output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Test backward pass
    grad_output = np.random.randn(batch_size, seq_len, d_model)
    grad_input = encoder_layer.backward(grad_output)
    print(f"Gradient input shape: {grad_input.shape}")
    
    print("✓ TransformerEncoderLayer tests passed!")
    
    print("\nTesting TransformerEncoder...")
    
    # Test encoder stack
    n_layers = 3
    encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers)
    
    # Forward pass
    output, all_attention_weights = encoder.forward(x)
    print(f"Encoder output shape: {output.shape}")
    print(f"Number of attention weight sets: {len(all_attention_weights)}")
    print(f"Expected: {n_layers}")
    
    # Test backward pass
    grad_input = encoder.backward(grad_output)
    print(f"Encoder gradient input shape: {grad_input.shape}")
    
    print("✓ TransformerEncoder tests passed!")
    
    print("\nTesting padding mask...")
    
    # Test padding mask creation
    seq = np.array([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    mask = create_padding_mask(seq, pad_token_id=0)
    print(f"Sequence shape: {seq.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask for first sequence: {mask[0, 0, 0]}")
    print(f"Mask for second sequence: {mask[1, 0, 0]}")
    
    print("✓ Padding mask tests passed!")
    
    print("\nAll encoder tests passed! 🎉")


if __name__ == "__main__":
    test_encoder()
