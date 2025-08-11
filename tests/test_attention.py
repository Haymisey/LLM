"""
Test Attention Mechanisms

This module tests all attention mechanisms to ensure they work correctly.
"""

import sys
import os
import numpy as np

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from attention.basic_attention import BasicAttention
from attention.self_attention import SelfAttention
from attention.multi_head_attention import MultiHeadAttention
from attention.sequence_attention import SequenceAttention

def test_basic_attention():
    """Test the basic attention mechanism."""
    print("Testing Basic Attention...")
    
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
    
    # Forward pass
    output = attention.forward(queries, keys, values)
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len_q, value_dim), f"Expected shape {(batch_size, seq_len_q, value_dim)}, got {output.shape}"
    
    # Verify attention weights shape
    attention_weights = attention.get_attention_weights()
    assert attention_weights.shape == (batch_size, seq_len_q, seq_len_k), f"Expected attention weights shape {(batch_size, seq_len_q, seq_len_k)}, got {attention_weights.shape}"
    
    # Verify attention weights sum to 1 along key dimension
    assert np.allclose(np.sum(attention_weights, axis=-1), 1.0), "Attention weights should sum to 1"
    
    print("✅ Basic attention tests passed!")

def test_self_attention():
    """Test the self-attention mechanism."""
    print("Testing Self Attention...")
    
    # Configuration
    batch_size = 2
    seq_len = 5
    embed_dim = 64
    num_heads = 8
    
    # Create self-attention mechanism
    self_attention = SelfAttention(embed_dim, num_heads)
    
    # Create dummy input data
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    output = self_attention.forward(x)
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, embed_dim), f"Expected shape {(batch_size, seq_len, embed_dim)}, got {output.shape}"
    
    # Verify attention weights shape
    attention_weights = self_attention.get_attention_weights()
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len), f"Expected attention weights shape {(batch_size, num_heads, seq_len, seq_len)}, got {attention_weights.shape}"
    
    # Verify attention patterns shape
    attention_patterns = self_attention.get_attention_patterns()
    assert attention_patterns.shape == (batch_size, seq_len, seq_len), f"Expected attention patterns shape {(batch_size, seq_len, seq_len)}, got {attention_patterns.shape}"
    
    print("✅ Self attention tests passed!")

def test_multi_head_attention():
    """Test the multi-head attention mechanism."""
    print("Testing Multi-Head Attention...")
    
    # Configuration
    batch_size = 2
    seq_len = 6
    embed_dim = 64
    num_heads = 8
    
    # Create multi-head attention mechanism
    multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
    
    # Create dummy input data
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    # Forward pass
    output = multi_head_attention.forward(x)
    
    # Verify output shape
    assert output.shape == (batch_size, seq_len, embed_dim), f"Expected shape {(batch_size, seq_len, embed_dim)}, got {output.shape}"
    
    # Verify attention weights shape
    attention_weights = multi_head_attention.get_attention_weights()
    assert attention_weights.shape == (batch_size, num_heads, seq_len, seq_len), f"Expected attention weights shape {(batch_size, num_heads, seq_len, seq_len)}, got {attention_weights.shape}"
    
    # Verify attention patterns shape
    attention_patterns = multi_head_attention.get_attention_patterns()
    assert attention_patterns.shape == (batch_size, seq_len, seq_len), f"Expected attention patterns shape {(batch_size, seq_len, seq_len)}, got {attention_patterns.shape}"
    
    # Verify individual head patterns shape
    head_patterns = multi_head_attention.get_head_attention_patterns()
    assert head_patterns.shape == (batch_size, num_heads, seq_len, seq_len), f"Expected head patterns shape {(batch_size, num_heads, seq_len, seq_len)}, got {head_patterns.shape}"
    
    print("✅ Multi-head attention tests passed!")

def test_sequence_attention():
    """Test the sequence-to-sequence attention mechanism."""
    print("Testing Sequence Attention...")
    
    # Configuration
    batch_size = 2
    seq_len = 4
    encoder_dim = 64
    decoder_dim = 64
    
    # Create sequence attention mechanism
    seq_attention = SequenceAttention(encoder_dim, decoder_dim)
    
    # Create dummy data
    encoder_states = np.random.randn(batch_size, seq_len, encoder_dim)
    decoder_state = np.random.randn(batch_size, decoder_dim)
    
    # Forward pass
    context_vector, attention_weights = seq_attention.forward(
        encoder_states, decoder_state
    )
    
    # Verify context vector shape
    assert context_vector.shape == (batch_size, encoder_dim), f"Expected context vector shape {(batch_size, encoder_dim)}, got {context_vector.shape}"
    
    # Verify attention weights shape
    assert attention_weights.shape == (batch_size, seq_len), f"Expected attention weights shape {(batch_size, seq_len)}, got {attention_weights.shape}"
    
    # Verify attention weights sum to 1
    assert np.allclose(np.sum(attention_weights, axis=-1), 1.0), "Attention weights should sum to 1"
    
    print("✅ Sequence attention tests passed!")

def test_attention_integration():
    """Test that all attention mechanisms work together."""
    print("Testing Attention Integration...")
    
    # Configuration
    batch_size = 2
    seq_len = 5
    embed_dim = 64
    num_heads = 8
    
    # Create input data
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    # Test pipeline: Basic -> Self -> Multi-Head
    print("  Testing attention pipeline...")
    
    # 1. Basic attention (simulate with self-attention)
    basic_attn = BasicAttention(embed_dim, embed_dim, embed_dim)
    basic_output = basic_attn.forward(x, x, x)
    
    # 2. Self attention
    self_attn = SelfAttention(embed_dim, num_heads)
    self_output = self_attn.forward(x)
    
    # 3. Multi-head attention
    multi_head_attn = MultiHeadAttention(embed_dim, num_heads)
    multi_head_output = multi_head_attn.forward(x)
    
    # Verify all outputs have correct shapes
    assert basic_output.shape == (batch_size, seq_len, embed_dim)
    assert self_output.shape == (batch_size, seq_len, embed_dim)
    assert multi_head_output.shape == (batch_size, seq_len, embed_dim)
    
    print("✅ Attention integration tests passed!")

def run_all_tests():
    """Run all attention mechanism tests."""
    print("🚀 RUNNING ATTENTION MECHANISM TESTS")
    print("=" * 60)
    
    try:
        test_basic_attention()
        test_self_attention()
        test_multi_head_attention()
        test_sequence_attention()
        test_attention_integration()
        
        print("\n" + "=" * 60)
        print("🎉 ALL ATTENTION TESTS PASSED!")
        print("=" * 60)
        
        print("\n📋 ATTENTION MECHANISMS VERIFIED:")
        print("✅ Basic Attention (Query-Key-Value)")
        print("✅ Self-Attention (Single sequence)")
        print("✅ Multi-Head Attention (Multiple heads)")
        print("✅ Sequence-to-Sequence Attention")
        print("✅ Attention Integration")
        
        print("\n🎯 Checkpoint 3 (Attention Mechanism) is COMPLETE!")
        print("Ready to move to Checkpoint 4: Transformer Core Components!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run all tests
    success = run_all_tests()
    
    if not success:
        sys.exit(1)
