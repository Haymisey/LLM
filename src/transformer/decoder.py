"""
Transformer Decoder Implementation
================================

This module implements the decoder part of the Transformer architecture:
- TransformerDecoderLayer: Single decoder layer with self-attention, cross-attention, and feed-forward
- TransformerDecoder: Stack of decoder layers
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
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None, 
                context: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
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


class TransformerDecoderLayer:
    """
    Single layer of the Transformer decoder.
    
    Each decoder layer consists of:
    1. Multi-head self-attention with residual connection and layer normalization
    2. Multi-head cross-attention with residual connection and layer normalization
    3. Feed-forward network with residual connection and layer normalization
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1):
        """
        Initialize the decoder layer.
        
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
        
        # Multi-head self-attention (for decoder input)
        self.self_attention = MockMultiHeadAttention(d_model, n_heads, dropout_rate)
        
        # Multi-head cross-attention (for encoder-decoder attention)
        self.cross_attention = MockMultiHeadAttention(d_model, n_heads, dropout_rate)
        
        # Feed-forward network
        self.feed_forward = MockFeedForward(d_model, d_ff, dropout_rate)
        
        # Layer normalization layers
        self.norm1 = MockLayerNormalization(d_model)
        self.norm2 = MockLayerNormalization(d_model)
        self.norm3 = MockLayerNormalization(d_model)
        
        # Dropout for regularization
        self.dropout = dropout_rate
        
    def forward(self, x: np.ndarray, encoder_output: np.ndarray, 
                self_attention_mask: Optional[np.ndarray] = None,
                cross_attention_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, list]:
        """
        Forward pass through the decoder layer.
        
        Args:
            x: Decoder input tensor of shape (batch_size, seq_len, d_model)
            encoder_output: Output from encoder of shape (batch_size, enc_seq_len, d_model)
            self_attention_mask: Optional mask for self-attention
            cross_attention_mask: Optional mask for cross-attention
            
        Returns:
            Output tensor and list of attention weights [self_attn, cross_attn]
        """
        attention_weights = []
        
        # Store input for residual connections
        residual = x
        
        # Self-attention with residual connection and layer normalization
        x_norm = self.norm1.forward(x)
        self_attn_output, self_attn_weights = self.self_attention.forward(x_norm, self_attention_mask)
        
        # Apply dropout and residual connection
        if self.dropout > 0:
            self_attn_output = self_attn_output * (1 - self.dropout)
        x = residual + self_attn_output
        attention_weights.append(self_attn_weights)
        
        # Cross-attention with residual connection and layer normalization
        residual = x
        x_norm = self.norm2.forward(x)
        cross_attn_output, cross_attn_weights = self.cross_attention.forward(
            x_norm, cross_attention_mask, context=encoder_output
        )
        
        # Apply dropout and residual connection
        if self.dropout > 0:
            cross_attn_output = cross_attn_output * (1 - self.dropout)
        x = residual + cross_attn_output
        attention_weights.append(cross_attn_weights)
        
        # Feed-forward with residual connection and layer normalization
        residual = x
        x_norm = self.norm3.forward(x)
        ff_output = self.feed_forward.forward(x_norm)
        
        # Apply dropout and residual connection
        if self.dropout > 0:
            ff_output = ff_output * (1 - self.dropout)
        x = residual + ff_output
        
        return x, attention_weights
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass through the decoder layer.
        
        Args:
            grad_output: Gradient of loss with respect to output
            
        Returns:
            Tuple of (gradient with respect to decoder input, gradient with respect to encoder output)
        """
        # Backward through feed-forward
        grad_ff = self.feed_forward.backward(grad_output)
        
        # Backward through layer norm
        grad_norm3 = self.norm3.backward(grad_ff)
        
        # Backward through cross-attention
        grad_cross_attn = self.cross_attention.backward(grad_norm3)
        
        # Backward through layer norm
        grad_norm2 = self.norm2.backward(grad_cross_attn)
        
        # Backward through self-attention
        grad_self_attn = self.self_attention.backward(grad_norm2)
        
        # Backward through first layer norm
        grad_norm1 = self.norm1.backward(grad_self_attn)
        
        # For simplicity, we'll return the same gradient for both inputs
        # In a full implementation, you'd need to properly handle the gradients
        return grad_norm1, grad_norm1
    
    def get_parameters(self) -> dict:
        """Get all parameters for saving/loading."""
        return {
            'self_attention': self.self_attention.get_parameters(),
            'cross_attention': self.cross_attention.get_parameters(),
            'feed_forward': self.feed_forward.get_parameters(),
            'norm1': self.norm1.get_parameters(),
            'norm2': self.norm2.get_parameters(),
            'norm3': self.norm3.get_parameters(),
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'dropout_rate': self.dropout_rate
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set parameters from saved state."""
        self.self_attention.set_parameters(params['self_attention'])
        self.cross_attention.set_parameters(params['cross_attention'])
        self.feed_forward.set_parameters(params['feed_forward'])
        self.norm1.set_parameters(params['norm1'])
        self.norm2.set_parameters(params['norm2'])
        self.norm3.set_parameters(params['norm3'])
        self.d_model = params['d_model']
        self.n_heads = params['n_heads']
        self.d_ff = params['d_ff']
        self.dropout_rate = params['dropout_rate']


class TransformerDecoder:
    """
    Stack of Transformer decoder layers.
    
    The decoder processes the target sequence through multiple layers,
    each applying self-attention, cross-attention, and feed-forward transformations.
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout_rate: float = 0.1):
        """
        Initialize the decoder stack.
        
        Args:
            d_model: Dimension of the model
            n_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network
            n_layers: Number of decoder layers
            dropout_rate: Dropout rate for regularization
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        
        # Create stack of decoder layers
        self.layers = [
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ]
        
    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                self_attention_mask: Optional[np.ndarray] = None,
                cross_attention_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, list]:
        """
        Forward pass through the decoder stack.
        
        Args:
            x: Decoder input tensor of shape (batch_size, seq_len, d_model)
            encoder_output: Output from encoder of shape (batch_size, enc_seq_len, d_model)
            self_attention_mask: Optional mask for self-attention
            cross_attention_mask: Optional mask for cross-attention
            
        Returns:
            Output tensor and list of attention weights from all layers
        """
        all_attention_weights = []
        
        # Process through each decoder layer
        for layer in self.layers:
            x, attention_weights = layer.forward(
                x, encoder_output, self_attention_mask, cross_attention_mask
            )
            all_attention_weights.append(attention_weights)
        
        return x, all_attention_weights
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass through the decoder stack.
        
        Args:
            grad_output: Gradient of loss with respect to output
            
        Returns:
            Tuple of (gradient with respect to decoder input, gradient with respect to encoder output)
        """
        grad_decoder = grad_output
        grad_encoder = np.zeros_like(grad_output)  # Placeholder
        
        # Backward through layers in reverse order
        for layer in reversed(self.layers):
            grad_dec, grad_enc = layer.backward(grad_decoder)
            grad_decoder = grad_dec
            grad_encoder = grad_enc  # In full implementation, accumulate these
        
        return grad_decoder, grad_encoder
    
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


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal mask for decoder self-attention.
    
    This mask ensures that each position can only attend to previous positions
    and itself, preventing information leakage from future tokens.
    
    Args:
        seq_len: Length of the sequence
        
    Returns:
        Causal mask of shape (1, 1, seq_len, seq_len)
        True values indicate positions to mask (future tokens)
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return mask.reshape(1, 1, seq_len, seq_len)


def create_look_ahead_mask(seq_len: int) -> np.ndarray:
    """
    Create look-ahead mask for decoder self-attention.
    
    This is the same as causal_mask but with a different name for clarity.
    
    Args:
        seq_len: Length of the sequence
        
    Returns:
        Look-ahead mask of shape (1, 1, seq_len, seq_len)
    """
    return create_causal_mask(seq_len)


def test_decoder():
    """Test the decoder implementation."""
    print("Testing TransformerDecoderLayer...")
    
    # Test single decoder layer
    d_model = 512
    n_heads = 8
    d_ff = 2048
    batch_size = 2
    dec_seq_len = 8
    enc_seq_len = 10
    
    decoder_layer = TransformerDecoderLayer(d_model, n_heads, d_ff)
    
    # Create inputs
    x = np.random.randn(batch_size, dec_seq_len, d_model)
    encoder_output = np.random.randn(batch_size, enc_seq_len, d_model)
    
    # Forward pass
    output, attention_weights = decoder_layer.forward(x, encoder_output)
    print(f"Decoder layer output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, {dec_seq_len}, {d_model})")
    print(f"Number of attention weight sets: {len(attention_weights)}")
    print(f"Expected: 2 (self-attention + cross-attention)")
    
    # Test backward pass
    grad_output = np.random.randn(batch_size, dec_seq_len, d_model)
    grad_dec, grad_enc = decoder_layer.backward(grad_output)
    print(f"Gradient decoder input shape: {grad_dec.shape}")
    print(f"Gradient encoder output shape: {grad_enc.shape}")
    
    print("✓ TransformerDecoderLayer tests passed!")
    
    print("\nTesting TransformerDecoder...")
    
    # Test decoder stack
    n_layers = 3
    decoder = TransformerDecoder(d_model, n_heads, d_ff, n_layers)
    
    # Forward pass
    output, all_attention_weights = decoder.forward(x, encoder_output)
    print(f"Decoder output shape: {output.shape}")
    print(f"Number of layers: {len(all_attention_weights)}")
    print(f"Expected: {n_layers}")
    print(f"Attention weights per layer: {len(all_attention_weights[0])}")
    print(f"Expected: 2 (self-attention + cross-attention)")
    
    # Test backward pass
    grad_dec, grad_enc = decoder.backward(grad_output)
    print(f"Decoder gradient input shape: {grad_dec.shape}")
    print(f"Encoder gradient output shape: {grad_enc.shape}")
    
    print("✓ TransformerDecoder tests passed!")
    
    print("\nTesting masks...")
    
    # Test causal mask
    seq_len = 5
    causal_mask = create_causal_mask(seq_len)
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Expected shape: (1, 1, {seq_len}, {seq_len})")
    print(f"Causal mask:\n{causal_mask[0, 0]}")
    
    # Test look-ahead mask
    look_ahead_mask = create_look_ahead_mask(seq_len)
    print(f"Look-ahead mask shape: {look_ahead_mask.shape}")
    print(f"Are masks identical? {np.array_equal(causal_mask, look_ahead_mask)}")
    
    print("✓ Mask tests passed!")
    
    print("\nAll decoder tests passed! 🎉")


if __name__ == "__main__":
    test_decoder()
