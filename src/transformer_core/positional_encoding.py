"""
Positional Encoding for Transformer Architecture

This module implements positional encoding to give the Transformer model
information about the position of tokens in a sequence. It uses sine and
cosine functions of different frequencies to create unique position encodings.
"""

import numpy as np
import matplotlib.pyplot as plt


class PositionalEncoding:
    """
    Positional Encoding for Transformer models.
    
    Adds positional information to token embeddings using sine and cosine
    functions of different frequencies. This allows the model to understand
    the relative positions of tokens in a sequence.
    
    Attributes:
        d_model (int): The dimension of the model embeddings
        max_len (int): Maximum sequence length for positional encoding
        pe (np.ndarray): Pre-computed positional encoding matrix
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model (int): The dimension of the model embeddings
            max_len (int): Maximum sequence length for positional encoding
        """
        self.d_model = d_model
        self.max_len = max_len
        
        # Create positional encoding matrix
        self.pe = np.zeros((max_len, d_model))
        
        # Calculate positional encodings
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # Apply sine to even indices
        self.pe[:, 0::2] = np.sin(position * div_term)
        # Apply cosine to odd indices
        self.pe[:, 1::2] = np.cos(position * div_term)
        
        # Add batch dimension for easier broadcasting
        self.pe = self.pe.reshape(1, max_len, d_model)
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x (np.ndarray): Input embeddings of shape (batch_size, seq_len, d_model)
            
        Returns:
            np.ndarray: Embeddings with positional encoding added
        """
        batch_size, seq_len, d_model = x.shape
        
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
        
        # Add positional encoding
        return x + self.pe[:, :seq_len, :]
    
    def get_positional_encoding(self, seq_len: int) -> np.ndarray:
        """
        Get positional encoding for a specific sequence length.
        
        Args:
            seq_len (int): Length of the sequence
            
        Returns:
            np.ndarray: Positional encoding of shape (1, seq_len, d_model)
        """
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum length {self.max_len}")
        
        return self.pe[:, :seq_len, :]
    
    def visualize_positions(self, seq_len: int = 50, d_model: int = 128):
        """
        Visualize positional encoding patterns.
        
        Args:
            seq_len (int): Number of positions to visualize
            d_model (int): Number of embedding dimensions to show
        """
        # Get positional encoding for visualization
        pe_vis = self.get_positional_encoding(seq_len)[0, :, :d_model]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        plt.imshow(pe_vis.T, aspect='auto', cmap='RdBu')
        plt.colorbar(label='Positional Encoding Value')
        plt.xlabel('Position in Sequence')
        plt.ylabel('Embedding Dimension')
        plt.title(f'Positional Encoding Visualization (seq_len={seq_len}, d_model={d_model})')
        plt.show()
        
        # Also show a few specific positions
        print(f"\nPositional Encoding for first 5 positions (showing first 10 dimensions):")
        for pos in range(min(5, seq_len)):
            print(f"Position {pos}: {pe_vis[pos, :10]}")
    
    def test_positional_encoding(self):
        """
        Test the positional encoding functionality.
        """
        print("Testing Positional Encoding...")
        
        # Test basic functionality
        d_model = 64
        seq_len = 20
        batch_size = 2
        
        pe = PositionalEncoding(d_model, max_len=100)
        
        # Create dummy embeddings
        x = np.random.randn(batch_size, seq_len, d_model)
        
        # Add positional encoding
        x_with_pe = pe(x)
        
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {x_with_pe.shape}")
        print(f"✓ Positional encoding added successfully")
        
        # Test that positional encoding is different for different positions
        pos_enc = pe.get_positional_encoding(seq_len)
        print(f"✓ Positional encoding shape: {pos_enc.shape}")
        
        # Check that different positions have different encodings
        pos_0 = pos_enc[0, 0, :]
        pos_1 = pos_enc[0, 1, :]
        pos_2 = pos_enc[0, 2, :]
        
        print(f"✓ Position 0 encoding (first 5 dims): {pos_0[:5]}")
        print(f"✓ Position 1 encoding (first 5 dims): {pos_1[:5]}")
        print(f"✓ Position 2 encoding (first 5 dims): {pos_2[:5]}")
        
        # Verify that positions are different
        assert not np.allclose(pos_0, pos_1), "Positions 0 and 1 should be different"
        assert not np.allclose(pos_1, pos_2), "Positions 1 and 2 should be different"
        print("✓ Position encodings are unique for different positions")
        
        print("✓ All tests passed!")
        return True


if __name__ == "__main__":
    # Run tests
    pe = PositionalEncoding(d_model=64, max_len=100)
    pe.test_positional_encoding()
    
    # Optional: Visualize (uncomment if you want to see the plot)
    # pe.visualize_positions(seq_len=30, d_model=32)
