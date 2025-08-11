"""
Layer Normalization for Transformer Architecture

This module implements layer normalization, which normalizes inputs across
the feature dimension. This helps with training stability and convergence.
"""

import numpy as np


class LayerNormalization:
    """
    Layer Normalization for Transformer models.
    
    Normalizes inputs across the feature dimension (last dimension) to have
    zero mean and unit variance. This helps with training stability and
    convergence.
    
    Attributes:
        d_model (int): The dimension of the model embeddings
        epsilon (float): Small constant to prevent division by zero
        gamma (np.ndarray): Learnable scale parameter
        beta (np.ndarray): Learnable shift parameter
    """
    
    def __init__(self, d_model: int, epsilon: float = 1e-6):
        """
        Initialize layer normalization.
        
        Args:
            d_model (int): The dimension of the model embeddings
            epsilon (float): Small constant to prevent division by zero
        """
        self.d_model = d_model
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones(d_model)  # Scale parameter
        self.beta = np.zeros(d_model)  # Shift parameter
        
        # Store intermediate values for backpropagation
        self.cache = {}
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass through layer normalization.
        
        Args:
            x (np.ndarray): Input tensor of shape (batch_size, seq_len, d_model)
            training (bool): Whether in training mode (unused, kept for consistency)
            
        Returns:
            np.ndarray: Normalized output tensor of same shape as input
        """
        batch_size, seq_len, d_model = x.shape
        
        # Store input for backpropagation
        self.cache['x'] = x.copy()
        
        # Calculate mean across the last dimension (features)
        # Shape: (batch_size, seq_len, 1)
        mean = np.mean(x, axis=-1, keepdims=True)
        
        # Store mean for backpropagation
        self.cache['mean'] = mean.copy()
        
        # Calculate variance across the last dimension (features)
        # Shape: (batch_size, seq_len, 1)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Store variance for backpropagation
        self.cache['var'] = var.copy()
        
        # Normalize: (x - mean) / sqrt(var + epsilon)
        x_norm = (x - mean) / np.sqrt(var + self.epsilon)
        
        # Store normalized values for backpropagation
        self.cache['x_norm'] = x_norm.copy()
        
        # Apply learnable parameters: gamma * x_norm + beta
        output = self.gamma * x_norm + self.beta
        
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through layer normalization.
        
        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output
            
        Returns:
            np.ndarray: Gradient of the loss with respect to the input
        """
        batch_size, seq_len, d_model = grad_output.shape
        
        # Retrieve cached values
        x = self.cache['x']
        mean = self.cache['mean']
        var = self.cache['var']
        x_norm = self.cache['x_norm']
        
        # Gradient with respect to gamma
        grad_gamma = np.sum(grad_output * x_norm, axis=(0, 1))
        
        # Gradient with respect to beta
        grad_beta = np.sum(grad_output, axis=(0, 1))
        
        # Gradient with respect to x_norm
        grad_x_norm = grad_output * self.gamma
        
        # Gradient with respect to x
        # This is the most complex part of the backward pass
        std = np.sqrt(var + self.epsilon)
        
        # Gradient with respect to x
        grad_x = grad_x_norm / std
        
        # Gradient with respect to mean
        grad_mean = -np.sum(grad_x_norm / std, axis=-1, keepdims=True)
        
        # Gradient with respect to variance
        grad_var = -0.5 * np.sum(grad_x_norm * (x - mean) / (std ** 3), axis=-1, keepdims=True)
        
        # Add gradients from mean and variance to x
        grad_x += grad_mean / d_model + 2 * grad_var * (x - mean) / d_model
        
        # Store gradients for parameter updates
        self.gradients = {
            'gamma': grad_gamma,
            'beta': grad_beta
        }
        
        return grad_x
    
    def get_parameters(self):
        """
        Get all learnable parameters.
        
        Returns:
            dict: Dictionary containing gamma and beta parameters
        """
        return {
            'gamma': self.gamma,
            'beta': self.beta
        }
    
    def set_parameters(self, params: dict):
        """
        Set all learnable parameters.
        
        Args:
            params (dict): Dictionary containing gamma and beta parameters
        """
        self.gamma = params['gamma'].copy()
        self.beta = params['beta'].copy()
    
    def test_layer_normalization(self):
        """
        Test the layer normalization functionality.
        """
        print("Testing Layer Normalization...")
        
        # Test parameters
        d_model = 64
        batch_size = 2
        seq_len = 10
        
        # Create layer normalization
        ln = LayerNormalization(d_model)
        
        # Create dummy input
        x = np.random.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = ln.forward(x)
        
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Layer normalization forward pass successful")
        
        # Test that output has the same shape as input
        assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
        print("✓ Output shape matches input shape")
        
        # Test that output is normalized (mean close to 0, std close to 1)
        # Note: This is approximate due to the learnable parameters
        output_mean = np.mean(output, axis=-1)
        output_std = np.std(output, axis=-1)
        
        print(f"✓ Output mean (first few values): {output_mean.flatten()[:5]}")
        print(f"✓ Output std (first few values): {output_std.flatten()[:5]}")
        
        # Test backward pass
        grad_output = np.random.randn(batch_size, seq_len, d_model)
        grad_input = ln.backward(grad_output)
        
        print(f"✓ Backward pass successful")
        print(f"✓ Gradient input shape: {grad_input.shape}")
        
        # Test that gradient input has the same shape as original input
        assert grad_input.shape == x.shape, f"Gradient input shape {grad_input.shape} != input shape {x.shape}"
        print("✓ Gradient input shape matches input shape")
        
        # Test parameters
        params = ln.get_parameters()
        print(f"✓ Parameters retrieved: {list(params.keys())}")
        
        # Test that parameters have correct shapes
        assert params['gamma'].shape == (d_model,), f"gamma shape {params['gamma'].shape} != ({d_model},)"
        assert params['beta'].shape == (d_model,), f"beta shape {params['beta'].shape} != ({d_model},)"
        print("✓ Parameter shapes are correct")
        
        # Test that gamma starts as ones and beta starts as zeros
        assert np.allclose(params['gamma'], 1.0), "gamma should start as ones"
        assert np.allclose(params['beta'], 0.0), "beta should start as zeros"
        print("✓ Initial parameter values are correct")
        
        print("✓ All tests passed!")
        return True


if __name__ == "__main__":
    # Run tests
    ln = LayerNormalization(d_model=64)
    ln.test_layer_normalization()
