"""
Output Projection Layer for Transformer
======================================

This module implements the output projection layer that maps the final
decoder output to vocabulary logits for token prediction.
"""

import numpy as np
from typing import Optional


class OutputProjection:
    """
    Output projection layer that maps decoder output to vocabulary logits.
    
    This is essentially a linear transformation that projects the final
    decoder representations (d_model) to vocabulary size (vocab_size).
    """
    
    def __init__(self, d_model: int, vocab_size: int):
        """
        Initialize the output projection layer.
        
        Args:
            d_model: Dimension of the model
            vocab_size: Size of the vocabulary
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Weight matrix for projection
        # Initialize with Xavier/Glorot initialization
        self.weights = np.random.randn(d_model, vocab_size) * np.sqrt(2.0 / d_model)
        
        # Bias vector
        self.bias = np.zeros(vocab_size)
        
        # Gradient storage
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: project input to vocabulary logits.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        self.input = x
        batch_size, seq_len, d_model = x.shape
        
        # Reshape for matrix multiplication
        x_reshaped = x.reshape(-1, d_model)
        
        # Linear transformation: x @ weights + bias
        logits = x_reshaped @ self.weights + self.bias
        
        # Reshape back to (batch_size, seq_len, vocab_size)
        return logits.reshape(batch_size, seq_len, self.vocab_size)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass: compute gradients for weights, bias, and input.
        
        Args:
            grad_output: Gradient of loss with respect to output
                        Shape: (batch_size, seq_len, vocab_size)
            
        Returns:
            Gradient of loss with respect to input
        """
        batch_size, seq_len, vocab_size = grad_output.shape
        
        # Reshape for matrix operations
        grad_output_reshaped = grad_output.reshape(-1, vocab_size)
        input_reshaped = self.input.reshape(-1, self.d_model)
        
        # Gradient with respect to weights
        self.grad_weights += input_reshaped.T @ grad_output_reshaped
        
        # Gradient with respect to bias
        self.grad_bias += grad_output_reshaped.sum(axis=0)
        
        # Gradient with respect to input
        grad_input = grad_output_reshaped @ self.weights.T
        
        # Reshape back to (batch_size, seq_len, d_model)
        return grad_input.reshape(batch_size, seq_len, self.d_model)
    
    def update_parameters(self, learning_rate: float) -> None:
        """
        Update parameters using accumulated gradients.
        
        Args:
            learning_rate: Learning rate for parameter updates
        """
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias
        
        # Reset gradients
        self.grad_weights.fill(0)
        self.grad_bias.fill(0)
    
    def get_parameters(self) -> dict:
        """Get all parameters for saving/loading."""
        return {
            'weights': self.weights.copy(),
            'bias': self.bias.copy(),
            'd_model': self.d_model,
            'vocab_size': self.vocab_size
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set parameters from saved state."""
        self.weights = params['weights'].copy()
        self.bias = params['bias'].copy()
        self.d_model = params['d_model']
        self.vocab_size = params['vocab_size']


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax function for numerical stability.
    
    Args:
        x: Input array
        axis: Axis along which to compute softmax
        
    Returns:
        Softmax probabilities
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    return x_exp / np.sum(x_exp, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute log-softmax function for numerical stability.
    
    Args:
        x: Input array
        axis: Axis along which to compute log-softmax
        
    Returns:
        Log-softmax values
    """
    # Subtract max for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    x_exp = np.exp(x - x_max)
    log_sum_exp = np.log(np.sum(x_exp, axis=axis, keepdims=True))
    return x - x_max - log_sum_exp


def test_output_projection():
    """Test the output projection layer."""
    print("Testing OutputProjection...")
    
    # Test output projection
    d_model = 512
    vocab_size = 1000
    batch_size = 2
    seq_len = 10
    
    output_proj = OutputProjection(d_model, vocab_size)
    
    # Create input
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    logits = output_proj.forward(x)
    print(f"Output projection logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {vocab_size})")
    
    # Test softmax
    probs = softmax(logits)
    print(f"Softmax probabilities shape: {probs.shape}")
    print(f"Probabilities sum to 1: {np.allclose(probs.sum(axis=-1), 1.0)}")
    
    # Test log-softmax
    log_probs = log_softmax(logits)
    print(f"Log-softmax shape: {log_probs.shape}")
    print(f"Log-softmax range: [{log_probs.min():.4f}, {log_probs.max():.4f}]")
    
    # Test backward pass
    grad_output = np.random.randn(batch_size, seq_len, vocab_size)
    grad_input = output_proj.backward(grad_output)
    print(f"Gradient input shape: {grad_input.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {d_model})")
    
    # Test parameter update
    output_proj.update_parameters(0.01)
    
    print("✓ OutputProjection tests passed!")
    
    print("\nAll output projection tests passed! 🎉")


if __name__ == "__main__":
    test_output_projection()
