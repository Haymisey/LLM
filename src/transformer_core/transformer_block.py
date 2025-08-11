"""
Transformer Block for Transformer Architecture

This module implements the core Transformer block, which combines multi-head
attention, feed-forward networks, and layer normalization with residual connections.
"""

import numpy as np
# from ..attention.multi_head_attention import MultiHeadAttention
# from .feed_forward import FeedForward
# from .layer_norm import LayerNormalization

# For standalone testing, we'll create mock classes
class MockMultiHeadAttention:
    def __init__(self, d_model, n_heads, dropout_rate):
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        # Initialize with random weights for testing
        self.W_Q = np.random.randn(d_model, d_model) * 0.1
        self.W_K = np.random.randn(d_model, d_model) * 0.1
        self.W_V = np.random.randn(d_model, d_model) * 0.1
        self.W_O = np.random.randn(d_model, d_model) * 0.1
    
    def forward(self, x, q, k, mask=None, training=True):
        # Simple mock implementation
        batch_size, seq_len, d_model = x.shape
        return np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    def backward(self, grad_output):
        batch_size, seq_len, d_model = grad_output.shape
        return np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    def get_parameters(self):
        return {'W_Q': self.W_Q, 'W_K': self.W_K, 'W_V': self.W_V, 'W_O': self.W_O}
    
    def set_parameters(self, params):
        for key, value in params.items():
            setattr(self, key, value)

class MockFeedForward:
    def __init__(self, d_model, d_ff, dropout_rate):
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        # Initialize with random weights for testing
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x, training=True):
        # Simple mock implementation
        batch_size, seq_len, d_model = x.shape
        return np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    def backward(self, grad_output):
        batch_size, seq_len, d_model = grad_output.shape
        return np.random.randn(batch_size, seq_len, d_model) * 0.1
    
    def get_parameters(self):
        return {'W1': self.W1, 'W2': self.W2, 'b1': self.b1, 'b2': self.b2}
    
    def set_parameters(self, params):
        for key, value in params.items():
            setattr(self, key, value)

class MockLayerNormalization:
    def __init__(self, d_model, epsilon=1e-6):
        self.d_model = d_model
        self.epsilon = epsilon
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x, training=True):
        # Simple mock implementation
        return x * self.gamma + self.beta
    
    def backward(self, grad_output):
        return grad_output
    
    def get_parameters(self):
        return {'gamma': self.gamma, 'beta': self.beta}
    
    def set_parameters(self, params):
        for key, value in params.items():
            setattr(self, key, value)


class TransformerBlock:
    """
    Transformer Block - the core building block of Transformer models.
    
    Each block consists of:
    1. Multi-head self-attention with residual connection and layer norm
    2. Feed-forward network with residual connection and layer norm
    
    Attributes:
        d_model (int): The dimension of the model embeddings
        n_heads (int): Number of attention heads
        d_ff (int): Dimension of the feed-forward network
        dropout_rate (float): Dropout probability
        attention (MultiHeadAttention): Multi-head attention mechanism
        feed_forward (FeedForward): Feed-forward network
        norm1 (LayerNormalization): First layer normalization
        norm2 (LayerNormalization): Second layer normalization
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout_rate: float = 0.1):
        """
        Initialize the Transformer block.
        
        Args:
            d_model (int): The dimension of the model embeddings
            n_heads (int): Number of attention heads
            d_ff (int): Dimension of the feed-forward network
            dropout_rate (float): Dropout probability for regularization
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # Multi-head attention
        self.attention = MockMultiHeadAttention(d_model, n_heads, dropout_rate)
        
        # Feed-forward network
        self.feed_forward = MockFeedForward(d_model, d_ff, dropout_rate)
        
        # Layer normalizations
        self.norm1 = MockLayerNormalization(d_model)
        self.norm2 = MockLayerNormalization(d_model)
        
        # Store intermediate values for backpropagation
        self.cache = {}
    
    def forward(self, x: np.ndarray, mask: np.ndarray = None, training: bool = True) -> np.ndarray:
        """
        Forward pass through the Transformer block.
        
        Args:
            x (np.ndarray): Input tensor of shape (batch_size, seq_len, d_model)
            mask (np.ndarray): Attention mask of shape (batch_size, seq_len, seq_len)
            training (bool): Whether in training mode
            
        Returns:
            np.ndarray: Output tensor of same shape as input
        """
        batch_size, seq_len, d_model = x.shape
        
        # Store input for backpropagation
        self.cache['x'] = x.copy()
        
        # First sub-layer: Multi-head attention with residual connection
        # Apply layer normalization to input
        x_norm1 = self.norm1.forward(x, training)
        
        # Apply multi-head attention
        attn_output = self.attention.forward(x_norm1, x_norm1, x_norm1, mask, training)
        
        # Add residual connection
        x_attn = x + attn_output
        
        # Store for backpropagation
        self.cache['x_attn'] = x_attn.copy()
        self.cache['x_norm1'] = x_norm1.copy()
        self.cache['attn_output'] = attn_output.copy()
        
        # Second sub-layer: Feed-forward network with residual connection
        # Apply layer normalization to the output of attention
        x_norm2 = self.norm2.forward(x_attn, training)
        
        # Apply feed-forward network
        ff_output = self.feed_forward.forward(x_norm2, training)
        
        # Add residual connection
        output = x_attn + ff_output
        
        # Store for backpropagation
        self.cache['x_norm2'] = x_norm2.copy()
        self.cache['ff_output'] = ff_output.copy()
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the Transformer block.
        
        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output
            
        Returns:
            np.ndarray: Gradient of the loss with respect to the input
        """
        # Retrieve cached values
        x = self.cache['x']
        x_attn = self.cache['x_attn']
        x_norm1 = self.cache['x_norm1']
        attn_output = self.cache['attn_output']
        x_norm2 = self.cache['x_norm2']
        ff_output = self.cache['ff_output']
        
        # Gradient with respect to feed-forward output
        grad_ff_output = grad_output
        
        # Gradient with respect to x_attn (from feed-forward residual)
        grad_x_attn_ff = grad_ff_output
        
        # Gradient with respect to x_norm2
        grad_x_norm2 = self.feed_forward.backward(grad_ff_output)
        
        # Gradient with respect to x_attn (from layer norm)
        grad_x_attn_norm = self.norm2.backward(grad_x_norm2)
        
        # Total gradient with respect to x_attn
        grad_x_attn = grad_x_attn_ff + grad_x_attn_norm
        
        # Gradient with respect to attention output
        grad_attn_output = grad_x_attn
        
        # Gradient with respect to x (from attention residual)
        grad_x_attn_res = grad_x_attn
        
        # Gradient with respect to x_norm1
        grad_x_norm1 = self.attention.backward(grad_attn_output)
        
        # Gradient with respect to x (from layer norm)
        grad_x_norm = self.norm1.backward(grad_x_norm1)
        
        # Total gradient with respect to x
        grad_x = grad_x_attn_res + grad_x_norm
        
        return grad_x
    
    def get_parameters(self):
        """
        Get all trainable parameters from all sub-components.
        
        Returns:
            dict: Dictionary containing all parameters
        """
        params = {}
        
        # Attention parameters
        attn_params = self.attention.get_parameters()
        for key, value in attn_params.items():
            params[f'attention_{key}'] = value
        
        # Feed-forward parameters
        ff_params = self.feed_forward.get_parameters()
        for key, value in ff_params.items():
            params[f'feed_forward_{key}'] = value
        
        # Layer normalization parameters
        norm1_params = self.norm1.get_parameters()
        for key, value in norm1_params.items():
            params[f'norm1_{key}'] = value
        
        norm2_params = self.norm2.get_parameters()
        for key, value in norm2_params.items():
            params[f'norm2_{key}'] = value
        
        return params
    
    def set_parameters(self, params: dict):
        """
        Set all trainable parameters for all sub-components.
        
        Args:
            params (dict): Dictionary containing all parameters
        """
        # Set attention parameters
        attn_params = {}
        for key in ['W_Q', 'W_K', 'W_V', 'W_O']:
            if f'attention_{key}' in params:
                attn_params[key] = params[f'attention_{key}']
        if attn_params:
            self.attention.set_parameters(attn_params)
        
        # Set feed-forward parameters
        ff_params = {}
        for key in ['W1', 'W2', 'b1', 'b2']:
            if f'feed_forward_{key}' in params:
                ff_params[key] = params[f'feed_forward_{key}']
        if ff_params:
            self.feed_forward.set_parameters(ff_params)
        
        # Set layer normalization parameters
        norm1_params = {}
        for key in ['gamma', 'beta']:
            if f'norm1_{key}' in params:
                norm1_params[key] = params[f'norm1_{key}']
        if norm1_params:
            self.norm1.set_parameters(norm1_params)
        
        norm2_params = {}
        for key in ['gamma', 'beta']:
            if f'norm2_{key}' in params:
                norm2_params[key] = params[f'norm2_{key}']
        if norm2_params:
            self.norm2.set_parameters(norm2_params)
    
    def test_transformer_block(self):
        """
        Test the Transformer block functionality.
        """
        print("Testing Transformer Block...")
        
        # Test parameters
        d_model = 64
        n_heads = 8
        d_ff = 128
        batch_size = 2
        seq_len = 10
        
        # Create Transformer block
        block = TransformerBlock(d_model, n_heads, d_ff, dropout_rate=0.1)
        
        # Create dummy input
        x = np.random.randn(batch_size, seq_len, d_model)
        
        # Create dummy mask (no masking for testing)
        mask = None
        
        # Forward pass
        output = block.forward(x, mask, training=True)
        
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Transformer block forward pass successful")
        
        # Test that output has the same shape as input
        assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
        print("✓ Output shape matches input shape")
        
        # Test backward pass
        grad_output = np.random.randn(batch_size, seq_len, d_model)
        grad_input = block.backward(grad_output)
        
        print(f"✓ Backward pass successful")
        print(f"✓ Gradient input shape: {grad_input.shape}")
        
        # Test that gradient input has the same shape as original input
        assert grad_input.shape == x.shape, f"Gradient input shape {grad_input.shape} != input shape {x.shape}"
        print("✓ Gradient input shape matches input shape")
        
        # Test parameters
        params = block.get_parameters()
        print(f"✓ Parameters retrieved: {len(params)} total parameters")
        
        # Check that we have parameters from all components
        param_keys = list(params.keys())
        assert any('attention_' in key for key in param_keys), "Missing attention parameters"
        assert any('feed_forward_' in key for key in param_keys), "Missing feed-forward parameters"
        assert any('norm1_' in key for key in param_keys), "Missing norm1 parameters"
        assert any('norm2_' in key for key in param_keys), "Missing norm2 parameters"
        print("✓ All component parameters are present")
        
        print("✓ All tests passed!")
        return True


if __name__ == "__main__":
    # Run tests
    block = TransformerBlock(d_model=64, n_heads=8, d_ff=128, dropout_rate=0.1)
    block.test_transformer_block()
