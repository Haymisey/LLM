"""
Feed-Forward Network for Transformer Architecture

This module implements the feed-forward network used in Transformer blocks.
It consists of two linear transformations with a ReLU activation in between.
"""

import numpy as np


class FeedForward:
    """
    Feed-Forward Network for Transformer models.
    
    Implements a two-layer feed-forward network with the following structure:
    Linear -> ReLU -> Dropout -> Linear
    
    This is applied to each position separately and identically.
    
    Attributes:
        d_model (int): The dimension of the model embeddings
        d_ff (int): The dimension of the feed-forward network
        dropout_rate (float): Dropout probability
        W1 (np.ndarray): First linear layer weights
        b1 (np.ndarray): First linear layer bias
        W2 (np.ndarray): Second linear layer weights
        b2 (np.ndarray): Second linear layer bias
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout_rate: float = 0.1):
        """
        Initialize the feed-forward network.
        
        Args:
            d_model (int): The dimension of the model embeddings
            d_ff (int): The dimension of the feed-forward network
            dropout_rate (float): Dropout probability for regularization
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # Initialize weights using Xavier/Glorot initialization
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
        
        # Store intermediate activations for backpropagation
        self.cache = {}
    
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """
        ReLU activation function.
        
        Args:
            x (np.ndarray): Input tensor
            
        Returns:
            np.ndarray: Output after ReLU activation
        """
        return np.maximum(0, x)
    
    def _relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Derivative of ReLU activation function.
        
        Args:
            x (np.ndarray): Input tensor
            
        Returns:
            np.ndarray: Derivative of ReLU
        """
        return (x > 0).astype(np.float64)
    
    def _dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Apply dropout during training.
        
        Args:
            x (np.ndarray): Input tensor
            training (bool): Whether in training mode
            
        Returns:
            np.ndarray: Output after dropout
        """
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape) / (1 - self.dropout_rate)
            return x * mask
        return x
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through the feed-forward network.
        
        Args:
            x (np.ndarray): Input tensor of shape (batch_size, seq_len, d_model)
            training (bool): Whether in training mode
            
        Returns:
            np.ndarray: Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Store input for backpropagation
        self.cache['x'] = x.copy()
        
        # First linear transformation
        # Reshape to (batch_size * seq_len, d_model) for matrix multiplication
        x_reshaped = x.reshape(-1, d_model)
        
        # Linear layer 1: (batch_size * seq_len, d_model) @ (d_model, d_ff)
        z1 = x_reshaped @ self.W1 + self.b1
        
        # Store for backpropagation
        self.cache['z1'] = z1.copy()
        
        # ReLU activation
        a1 = self._relu(z1)
        
        # Store for backpropagation
        self.cache['a1'] = a1.copy()
        
        # Dropout
        a1_dropout = self._dropout(a1, training)
        
        # Store for backpropagation
        self.cache['a1_dropout'] = a1_dropout.copy()
        
        # Linear layer 2: (batch_size * seq_len, d_ff) @ (d_ff, d_model)
        z2 = a1_dropout @ self.W2 + self.b2
        
        # Store for backpropagation
        self.cache['z2'] = z2.copy()
        
        # Reshape back to (batch_size, seq_len, d_model)
        output = z2.reshape(batch_size, seq_len, d_model)
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the feed-forward network.
        
        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output
            
        Returns:
            np.ndarray: Gradient of the loss with respect to the input
        """
        batch_size, seq_len, d_model = grad_output.shape
        
        # Reshape gradients
        grad_output_reshaped = grad_output.reshape(-1, d_model)
        
        # Gradient with respect to z2
        grad_z2 = grad_output_reshaped
        
        # Gradient with respect to W2
        grad_W2 = self.cache['a1_dropout'].T @ grad_z2
        grad_b2 = np.sum(grad_z2, axis=0)
        
        # Gradient with respect to a1_dropout
        grad_a1_dropout = grad_z2 @ self.W2.T
        
        # Gradient with respect to a1 (accounting for dropout)
        grad_a1 = grad_a1_dropout * (self.cache['a1_dropout'] > 0).astype(np.float64)
        
        # Gradient with respect to z1
        grad_z1 = grad_a1 * self._relu_derivative(self.cache['z1'])
        
        # Gradient with respect to W1
        grad_W1 = self.cache['x'].reshape(-1, d_model).T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)
        
        # Gradient with respect to input
        grad_input = grad_z1 @ self.W1.T
        grad_input = grad_input.reshape(batch_size, seq_len, d_model)
        
        # Update weights (in a real implementation, you'd use an optimizer)
        # For now, we'll just store the gradients
        self.gradients = {
            'W1': grad_W1,
            'b1': grad_b1,
            'W2': grad_W2,
            'b2': grad_b2
        }
        
        return grad_input
    
    def get_parameters(self):
        """
        Get all trainable parameters.
        
        Returns:
            dict: Dictionary containing all parameters
        """
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
    
    def set_parameters(self, params: dict):
        """
        Set all trainable parameters.
        
        Args:
            params (dict): Dictionary containing all parameters
        """
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
    
    def test_feed_forward(self):
        """
        Test the feed-forward network functionality.
        """
        print("Testing Feed-Forward Network...")
        
        # Test parameters
        d_model = 64
        d_ff = 128
        batch_size = 2
        seq_len = 10
        
        # Create feed-forward network
        ff = FeedForward(d_model, d_ff, dropout_rate=0.1)
        
        # Create dummy input
        x = np.random.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = ff.forward(x, training=True)
        
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Feed-forward network forward pass successful")
        
        # Test that output has the same shape as input
        assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
        print("✓ Output shape matches input shape")
        
        # Test backward pass
        grad_output = np.random.randn(batch_size, seq_len, d_model)
        grad_input = ff.backward(grad_output)
        
        print(f"✓ Backward pass successful")
        print(f"✓ Gradient input shape: {grad_input.shape}")
        
        # Test that gradient input has the same shape as original input
        assert grad_input.shape == x.shape, f"Gradient input shape {grad_input.shape} != input shape {x.shape}"
        print("✓ Gradient input shape matches input shape")
        
        # Test parameters
        params = ff.get_parameters()
        print(f"✓ Parameters retrieved: {list(params.keys())}")
        
        # Test that weights have correct shapes
        assert params['W1'].shape == (d_model, d_ff), f"W1 shape {params['W1'].shape} != ({d_model}, {d_ff})"
        assert params['W2'].shape == (d_ff, d_model), f"W2 shape {params['W2'].shape} != ({d_ff}, {d_model})"
        print("✓ Weight shapes are correct")
        
        print("✓ All tests passed!")
        return True


if __name__ == "__main__":
    # Run tests
    ff = FeedForward(d_model=64, d_ff=128, dropout_rate=0.1)
    ff.test_feed_forward()
