"""
Attention Mechanism Examples

This module provides comprehensive examples demonstrating how to use
all the attention mechanisms implemented in this package.
"""

import numpy as np
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(project_root)

from src.attention import (
    BasicAttention, 
    SelfAttention, 
    MultiHeadAttention,
    AttentionVisualizer,
    SequenceAttention
)

def basic_attention_example():
    """Demonstrate basic attention mechanism."""
    print("=" * 60)
    print("BASIC ATTENTION EXAMPLE")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    seq_len_q = 3
    seq_len_k = 4
    query_dim = 8
    key_dim = 8
    value_dim = 6
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Query sequence length: {seq_len_q}")
    print(f"  Key/Value sequence length: {seq_len_k}")
    print(f"  Query dimension: {query_dim}")
    print(f"  Key dimension: {key_dim}")
    print(f"  Value dimension: {value_dim}")
    
    # Create attention mechanism
    attention = BasicAttention(query_dim, key_dim, value_dim)
    
    # Create dummy input data
    queries = np.random.randn(batch_size, seq_len_q, query_dim)
    keys = np.random.randn(batch_size, seq_len_k, key_dim)
    values = np.random.randn(batch_size, seq_len_k, value_dim)
    
    print(f"\nInput shapes:")
    print(f"  Queries: {queries.shape}")
    print(f"  Keys: {keys.shape}")
    print(f"  Values: {values.shape}")
    
    # Forward pass
    output = attention.forward(queries, keys, values)
    print(f"\nOutput shape: {output.shape}")
    
    # Get attention weights
    attention_weights = attention.get_attention_weights()
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Show attention weights for first batch
    print(f"\nAttention weights (first batch):")
    print(attention_weights[0])
    
    return attention_weights[0]

def self_attention_example():
    """Demonstrate self-attention mechanism."""
    print("\n" + "=" * 60)
    print("SELF-ATTENTION EXAMPLE")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    seq_len = 5
    embed_dim = 64
    num_heads = 8
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {embed_dim // num_heads}")
    
    # Create self-attention mechanism
    self_attention = SelfAttention(embed_dim, num_heads)
    
    # Create dummy input data
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = self_attention.forward(x)
    print(f"Output shape: {output.shape}")
    
    # Get attention weights
    attention_weights = self_attention.get_attention_weights()
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Get attention patterns (averaged across heads)
    attention_patterns = self_attention.get_attention_patterns()
    print(f"Attention patterns shape: {attention_patterns.shape}")
    
    # Show attention pattern for first batch
    print(f"\nAttention pattern (first batch, averaged across heads):")
    print(attention_patterns[0])
    
    return attention_patterns[0]

def multi_head_attention_example():
    """Demonstrate multi-head attention mechanism."""
    print("\n" + "=" * 60)
    print("MULTI-HEAD ATTENTION EXAMPLE")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    seq_len = 6
    embed_dim = 64
    num_heads = 8
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Number of heads: {num_heads}")
    print(f"  Head dimension: {embed_dim // num_heads}")
    
    # Create multi-head attention mechanism
    multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
    
    # Create dummy input data
    x = np.random.randn(batch_size, seq_len, embed_dim)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    output = multi_head_attention.forward(x)
    print(f"Output shape: {output.shape}")
    
    # Get attention weights
    attention_weights = multi_head_attention.get_attention_weights()
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Get attention patterns (averaged across heads)
    attention_patterns = multi_head_attention.get_attention_patterns()
    print(f"Attention patterns shape: {attention_patterns.shape}")
    
    # Get individual head attention patterns
    head_patterns = multi_head_attention.get_head_attention_patterns()
    print(f"Individual head patterns shape: {head_patterns.shape}")
    
    # Show attention pattern for first batch, first head
    print(f"\nAttention pattern (first batch, first head):")
    print(head_patterns[0, 0])
    
    # Show averaged attention pattern for first batch
    print(f"\nAveraged attention pattern (first batch, across all heads):")
    print(attention_patterns[0])
    
    return attention_patterns[0]

def sequence_attention_example():
    """Demonstrate sequence-to-sequence attention mechanism."""
    print("\n" + "=" * 60)
    print("SEQUENCE-TO-SEQUENCE ATTENTION EXAMPLE")
    print("=" * 60)
    
    # Configuration
    batch_size = 2
    seq_len = 4
    encoder_dim = 64
    decoder_dim = 64
    vocab_size = 1000
    
    print(f"Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Encoder dimension: {encoder_dim}")
    print(f"  Decoder dimension: {decoder_dim}")
    print(f"  Vocabulary size: {vocab_size}")
    
    # Test basic sequence attention
    print("\n1. Testing Basic Sequence Attention...")
    seq_attention = SequenceAttention(encoder_dim, decoder_dim)
    
    # Create dummy data
    encoder_states = np.random.randn(batch_size, seq_len, encoder_dim)
    decoder_state = np.random.randn(batch_size, decoder_dim)
    
    # Forward pass
    context_vector, attention_weights = seq_attention.forward(
        encoder_states, decoder_state
    )
    
    print(f"Encoder states shape: {encoder_states.shape}")
    print(f"Decoder state shape: {decoder_state.shape}")
    print(f"Context vector shape: {context_vector.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Test attention decoder
    print("\n2. Testing Attention Decoder...")
    decoder = SequenceAttention.AttentionDecoder(vocab_size, 128, decoder_dim, encoder_dim)
    
    # Create dummy input tokens
    input_tokens = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    
    # Forward pass
    output_logits, decoder_attention = decoder.forward(
        input_tokens, encoder_states
    )
    
    print(f"Input tokens shape: {input_tokens.shape}")
    print(f"Output logits shape: {output_logits.shape}")
    print(f"Decoder attention shape: {decoder_attention.shape}")
    
    return attention_weights[0]

def visualization_example():
    """Demonstrate attention visualization capabilities."""
    print("\n" + "=" * 60)
    print("ATTENTION VISUALIZATION EXAMPLE")
    print("=" * 60)
    
    # Create visualizer
    visualizer = AttentionVisualizer()
    
    # Create sample attention weights
    seq_len = 5
    attention_weights = np.array([
        [0.8, 0.1, 0.05, 0.03, 0.02],
        [0.1, 0.7, 0.15, 0.03, 0.02],
        [0.05, 0.1, 0.6, 0.2, 0.05],
        [0.02, 0.05, 0.15, 0.65, 0.13],
        [0.01, 0.02, 0.05, 0.12, 0.8]
    ])
    
    # Create sample labels
    labels = ['Word1', 'Word2', 'Word3', 'Word4', 'Word5']
    
    print("Creating sample attention visualizations...")
    
    # Plot single attention weights
    print("1. Plotting single attention weights...")
    visualizer.plot_attention_weights(attention_weights, 
                                    "Sample Attention Weights",
                                    labels)
    
    # Create sample multi-head attention
    batch_size, num_heads = 1, 4
    multi_head_weights = np.random.rand(batch_size, num_heads, seq_len, seq_len)
    # Normalize to make it look like attention weights
    for b in range(batch_size):
        for h in range(num_heads):
            multi_head_weights[b, h] = visualizer._softmax(multi_head_weights[b, h])
    
    # Plot multi-head attention
    print("2. Plotting multi-head attention...")
    visualizer.plot_multi_head_attention(multi_head_weights,
                                       "Sample Multi-Head Attention",
                                       labels)
    
    print("✅ Sample visualizations created!")

def run_all_examples():
    """Run all attention mechanism examples."""
    print("🚀 RUNNING ALL ATTENTION MECHANISM EXAMPLES")
    print("=" * 80)
    
    try:
        # Run all examples
        basic_attn_weights = basic_attention_example()
        self_attn_weights = self_attention_example()
        multi_head_weights = multi_head_attention_example()
        seq_attn_weights = sequence_attention_example()
        
        # Run visualization example (this will show plots)
        print("\n" + "=" * 80)
        print("🎨 ATTENTION VISUALIZATION")
        print("=" * 80)
        visualization_example()
        
        print("\n" + "=" * 80)
        print("🎉 ALL ATTENTION EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        # Summary of what we've built
        print("\n📋 SUMMARY OF ATTENTION MECHANISMS:")
        print("✅ Basic Attention (Query-Key-Value)")
        print("✅ Self-Attention (Single sequence)")
        print("✅ Multi-Head Attention (Multiple attention heads)")
        print("✅ Sequence-to-Sequence Attention (Encoder-Decoder)")
        print("✅ Attention Visualization Tools")
        
        print("\n🔧 NEXT STEPS:")
        print("1. Test individual components")
        print("2. Integrate with Transformer architecture")
        print("3. Build training pipeline")
        print("4. Implement Bible translation model")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run all examples
    success = run_all_examples()
    
    if success:
        print("\n🎯 Checkpoint 3 (Attention Mechanism) is ready!")
        print("Ready to move to Checkpoint 4: Transformer Core Components!")
    else:
        print("\n⚠️  Some examples failed. Please check the error messages above.")
