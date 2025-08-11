"""
Tests for Transformer Core Components

This module tests all the core Transformer components:
- PositionalEncoding
- FeedForward
- LayerNormalization
- TransformerBlock
"""

import numpy as np
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformer_core.positional_encoding import PositionalEncoding
from transformer_core.feed_forward import FeedForward
from transformer_core.layer_norm import LayerNormalization
from transformer_core.transformer_block import TransformerBlock


def test_positional_encoding():
    """Test the PositionalEncoding class."""
    print("\n=== Testing PositionalEncoding ===")
    
    # Test basic functionality
    d_model = 64
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    
    # Test initialization
    assert pe.d_model == d_model
    assert pe.max_len == max_len
    assert pe.pe.shape == (1, max_len, d_model)
    print("✓ Initialization successful")
    
    # Test forward pass
    batch_size = 2
    seq_len = 20
    x = np.random.randn(batch_size, seq_len, d_model)
    output = pe(x)
    
    assert output.shape == x.shape
    assert not np.allclose(output, x)  # Should be different due to positional encoding
    print("✓ Forward pass successful")
    
    # Test that different positions have different encodings
    pos_enc = pe.get_positional_encoding(seq_len)
    pos_0 = pos_enc[0, 0, :]
    pos_1 = pos_enc[0, 1, :]
    assert not np.allclose(pos_0, pos_1)
    print("✓ Position encodings are unique")
    
    # Test error handling
    try:
        pe(np.random.randn(batch_size, max_len + 1, d_model))
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Error handling for sequence length works")
    
    print("✓ PositionalEncoding tests passed!")


def test_feed_forward():
    """Test the FeedForward class."""
    print("\n=== Testing FeedForward ===")
    
    # Test basic functionality
    d_model = 64
    d_ff = 128
    dropout_rate = 0.1
    ff = FeedForward(d_model, d_ff, dropout_rate)
    
    # Test initialization
    assert ff.d_model == d_model
    assert ff.d_ff == d_ff
    assert ff.dropout_rate == dropout_rate
    assert ff.W1.shape == (d_model, d_ff)
    assert ff.W2.shape == (d_ff, d_model)
    print("✓ Initialization successful")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model)
    output = ff.forward(x, training=True)
    
    assert output.shape == x.shape
    print("✓ Forward pass successful")
    
    # Test backward pass
    grad_output = np.random.randn(batch_size, seq_len, d_model)
    grad_input = ff.backward(grad_output)
    
    assert grad_input.shape == x.shape
    print("✓ Backward pass successful")
    
    # Test parameters
    params = ff.get_parameters()
    assert 'W1' in params and 'W2' in params
    assert 'b1' in params and 'b2' in params
    print("✓ Parameter management works")
    
    print("✓ FeedForward tests passed!")


def test_layer_normalization():
    """Test the LayerNormalization class."""
    print("\n=== Testing LayerNormalization ===")
    
    # Test basic functionality
    d_model = 64
    epsilon = 1e-6
    ln = LayerNormalization(d_model, epsilon)
    
    # Test initialization
    assert ln.d_model == d_model
    assert ln.epsilon == epsilon
    assert np.allclose(ln.gamma, 1.0)
    assert np.allclose(ln.beta, 0.0)
    print("✓ Initialization successful")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model)
    output = ln.forward(x, training=True)
    
    assert output.shape == x.shape
    print("✓ Forward pass successful")
    
    # Test backward pass
    grad_output = np.random.randn(batch_size, seq_len, d_model)
    grad_input = ln.backward(grad_output)
    
    assert grad_input.shape == x.shape
    print("✓ Backward pass successful")
    
    # Test parameters
    params = ln.get_parameters()
    assert 'gamma' in params and 'beta' in params
    print("✓ Parameter management works")
    
    print("✓ LayerNormalization tests passed!")


def test_transformer_block():
    """Test the TransformerBlock class."""
    print("\n=== Testing TransformerBlock ===")
    
    # Test basic functionality
    d_model = 64
    n_heads = 8
    d_ff = 128
    dropout_rate = 0.1
    block = TransformerBlock(d_model, n_heads, d_ff, dropout_rate)
    
    # Test initialization
    assert block.d_model == d_model
    assert block.n_heads == n_heads
    assert block.d_ff == d_ff
    assert block.dropout_rate == dropout_rate
    print("✓ Initialization successful")
    
    # Test forward pass
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model)
    output = block.forward(x, mask=None, training=True)
    
    assert output.shape == x.shape
    print("✓ Forward pass successful")
    
    # Test backward pass
    grad_output = np.random.randn(batch_size, seq_len, d_model)
    grad_input = block.backward(grad_output)
    
    assert grad_input.shape == x.shape
    print("✓ Backward pass successful")
    
    # Test parameters
    params = block.get_parameters()
    assert len(params) > 0
    print(f"✓ Parameter management works ({len(params)} parameters)")
    
    print("✓ TransformerBlock tests passed!")


def test_integration():
    """Test that all components work together."""
    print("\n=== Testing Integration ===")
    
    # Create a simple pipeline
    d_model = 64
    n_heads = 8
    d_ff = 128
    seq_len = 10
    batch_size = 2
    
    # Create components
    pe = PositionalEncoding(d_model)
    block = TransformerBlock(d_model, n_heads, d_ff)
    
    # Create input
    x = np.random.randn(batch_size, seq_len, d_model)
    
    # Add positional encoding
    x_with_pe = pe(x)
    
    # Pass through transformer block
    output = block.forward(x_with_pe, mask=None, training=True)
    
    # Check shapes
    assert x_with_pe.shape == (batch_size, seq_len, d_model)
    assert output.shape == (batch_size, seq_len, d_model)
    
    print("✓ Integration test successful")
    print("✓ All components work together!")


def run_all_tests():
    """Run all tests."""
    print("🚀 Starting Transformer Core Tests...")
    
    try:
        test_positional_encoding()
        test_feed_forward()
        test_layer_normalization()
        test_transformer_block()
        test_integration()
        
        print("\n🎉 All Transformer Core tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
