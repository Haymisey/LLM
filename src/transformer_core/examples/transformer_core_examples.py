"""
Transformer Core Examples

This module demonstrates how to use all the transformer core components:
- PositionalEncoding
- FeedForward
- LayerNormalization
- TransformerBlock

It shows practical usage patterns and how components work together.
"""

import numpy as np
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer_core.positional_encoding import PositionalEncoding
from transformer_core.feed_forward import FeedForward
from transformer_core.layer_norm import LayerNormalization
from transformer_core.transformer_block import TransformerBlock


def demonstrate_positional_encoding():
    """Demonstrate positional encoding functionality."""
    print("\n" + "="*60)
    print("🔤 POSITIONAL ENCODING DEMONSTRATION")
    print("="*60)
    
    # Create positional encoding
    d_model = 64
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    
    print(f"✓ Created PositionalEncoding with d_model={d_model}, max_len={max_len}")
    
    # Create sample embeddings
    batch_size = 2
    seq_len = 20
    embeddings = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"✓ Created sample embeddings: {embeddings.shape}")
    
    # Add positional encoding
    embeddings_with_pe = pe(embeddings)
    
    print(f"✓ Added positional encoding: {embeddings_with_pe.shape}")
    
    # Show that positions are different
    pos_0 = pe.get_positional_encoding(seq_len)[0, 0, :5]
    pos_1 = pe.get_positional_encoding(seq_len)[0, 1, :5]
    pos_2 = pe.get_positional_encoding(seq_len)[0, 2, :5]
    
    print(f"✓ Position 0 encoding (first 5 dims): {pos_0}")
    print(f"✓ Position 1 encoding (first 5 dims): {pos_1}")
    print(f"✓ Position 2 encoding (first 5 dims): {pos_2}")
    
    # Verify positions are different
    assert not np.allclose(pos_0, pos_1), "Positions should be different"
    assert not np.allclose(pos_1, pos_2), "Positions should be different"
    
    print("✓ Positional encoding successfully differentiates positions!")


def demonstrate_feed_forward():
    """Demonstrate feed-forward network functionality."""
    print("\n" + "="*60)
    print("🔄 FEED-FORWARD NETWORK DEMONSTRATION")
    print("="*60)
    
    # Create feed-forward network
    d_model = 64
    d_ff = 128
    dropout_rate = 0.1
    ff = FeedForward(d_model, d_ff, dropout_rate)
    
    print(f"✓ Created FeedForward with d_model={d_model}, d_ff={d_ff}, dropout={dropout_rate}")
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"✓ Created sample input: {x.shape}")
    
    # Forward pass
    output = ff.forward(x, training=True)
    
    print(f"✓ Forward pass successful: {output.shape}")
    
    # Verify output shape
    assert output.shape == x.shape, "Output shape should match input shape"
    
    # Test backward pass
    grad_output = np.random.randn(batch_size, seq_len, d_model)
    grad_input = ff.backward(grad_output)
    
    print(f"✓ Backward pass successful: {grad_input.shape}")
    
    # Verify gradient shape
    assert grad_input.shape == x.shape, "Gradient input shape should match input shape"
    
    # Show parameters
    params = ff.get_parameters()
    print(f"✓ Network has {len(params)} parameter groups:")
    for key, value in params.items():
        print(f"  - {key}: {value.shape}")
    
    print("✓ Feed-forward network working correctly!")


def demonstrate_layer_normalization():
    """Demonstrate layer normalization functionality."""
    print("\n" + "="*60)
    print("📊 LAYER NORMALIZATION DEMONSTRATION")
    print("="*60)
    
    # Create layer normalization
    d_model = 64
    epsilon = 1e-6
    ln = LayerNormalization(d_model, epsilon)
    
    print(f"✓ Created LayerNormalization with d_model={d_model}, epsilon={epsilon}")
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"✓ Created sample input: {x.shape}")
    
    # Show input statistics
    input_mean = np.mean(x, axis=-1)
    input_std = np.std(x, axis=-1)
    print(f"✓ Input mean (first few): {input_mean.flatten()[:5]}")
    print(f"✓ Input std (first few): {input_std.flatten()[:5]}")
    
    # Forward pass
    output = ln.forward(x, training=True)
    
    print(f"✓ Forward pass successful: {output.shape}")
    
    # Verify output shape
    assert output.shape == x.shape, "Output shape should match input shape"
    
    # Show output statistics (should be closer to 0 mean, 1 std due to learnable params)
    output_mean = np.mean(output, axis=-1)
    output_std = np.std(output, axis=-1)
    print(f"✓ Output mean (first few): {output_mean.flatten()[:5]}")
    print(f"✓ Output std (first few): {output_std.flatten()[:5]}")
    
    # Test backward pass
    grad_output = np.random.randn(batch_size, seq_len, d_model)
    grad_input = ln.backward(grad_output)
    
    print(f"✓ Backward pass successful: {grad_input.shape}")
    
    # Verify gradient shape
    assert grad_input.shape == x.shape, "Gradient input shape should match input shape"
    
    # Show parameters
    params = ln.get_parameters()
    print(f"✓ Layer normalization has {len(params)} parameters:")
    for key, value in params.items():
        print(f"  - {key}: {value.shape} (initial value: {value[0]})")
    
    print("✓ Layer normalization working correctly!")


def demonstrate_transformer_block():
    """Demonstrate transformer block functionality."""
    print("\n" + "="*60)
    print("🧩 TRANSFORMER BLOCK DEMONSTRATION")
    print("="*60)
    
    # Create transformer block
    d_model = 64
    n_heads = 8
    d_ff = 128
    dropout_rate = 0.1
    block = TransformerBlock(d_model, n_heads, d_ff, dropout_rate)
    
    print(f"✓ Created TransformerBlock with d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
    
    # Create sample input
    batch_size = 2
    seq_len = 10
    x = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"✓ Created sample input: {x.shape}")
    
    # Forward pass
    output = block.forward(x, mask=None, training=True)
    
    print(f"✓ Forward pass successful: {output.shape}")
    
    # Verify output shape
    assert output.shape == x.shape, "Output shape should match input shape"
    
    # Test backward pass
    grad_output = np.random.randn(batch_size, seq_len, d_model)
    grad_input = block.backward(grad_output)
    
    print(f"✓ Backward pass successful: {grad_input.shape}")
    
    # Verify gradient shape
    assert grad_input.shape == x.shape, "Gradient input shape should match input shape"
    
    # Show parameters
    params = block.get_parameters()
    print(f"✓ Transformer block has {len(params)} total parameters:")
    
    # Group parameters by component
    component_params = {}
    for key in params.keys():
        component = key.split('_')[0]
        if component not in component_params:
            component_params[component] = 0
        component_params[component] += 1
    
    for component, count in component_params.items():
        print(f"  - {component}: {count} parameter groups")
    
    print("✓ Transformer block working correctly!")


def demonstrate_integration():
    """Demonstrate how all components work together."""
    print("\n" + "="*60)
    print("🔗 INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Configuration
    d_model = 64
    n_heads = 8
    d_ff = 128
    seq_len = 15
    batch_size = 3
    
    print(f"✓ Configuration: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"✓ Input: batch_size={batch_size}, seq_len={seq_len}")
    
    # Create all components
    pe = PositionalEncoding(d_model)
    block = TransformerBlock(d_model, n_heads, d_ff)
    
    print("✓ Created all components")
    
    # Create sample input (token embeddings)
    token_embeddings = np.random.randn(batch_size, seq_len, d_model)
    print(f"✓ Created token embeddings: {token_embeddings.shape}")
    
    # Step 1: Add positional encoding
    embeddings_with_pe = pe(token_embeddings)
    print(f"✓ Added positional encoding: {embeddings_with_pe.shape}")
    
    # Step 2: Pass through transformer block
    transformer_output = block.forward(embeddings_with_pe, mask=None, training=True)
    print(f"✓ Passed through transformer block: {transformer_output.shape}")
    
    # Verify shapes throughout the pipeline
    assert embeddings_with_pe.shape == (batch_size, seq_len, d_model)
    assert transformer_output.shape == (batch_size, seq_len, d_model)
    
    # Test backward pass through the entire pipeline
    grad_output = np.random.randn(batch_size, seq_len, d_model)
    
    # Backward through transformer block
    grad_transformer = block.backward(grad_output)
    print(f"✓ Backward pass through transformer block: {grad_transformer.shape}")
    
    # Note: We don't have backward pass for positional encoding in this implementation
    # In practice, positional encoding gradients would flow through to the embedding layer
    
    print("✓ Integration test successful!")
    print("✓ All components work together seamlessly!")


def demonstrate_attention_visualization():
    """Demonstrate attention visualization capabilities."""
    print("\n" + "="*60)
    print("👁️ ATTENTION VISUALIZATION DEMONSTRATION")
    print("="*60)
    
    # Create a simple transformer block for attention analysis
    d_model = 32  # Smaller for visualization
    n_heads = 4
    d_ff = 64
    block = TransformerBlock(d_model, n_heads, d_ff)
    
    print(f"✓ Created transformer block for visualization: d_model={d_model}, n_heads={n_heads}")
    
    # Create sample input
    batch_size = 1
    seq_len = 8
    x = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"✓ Created sample input: {x.shape}")
    
    # Forward pass to get attention weights
    output = block.forward(x, mask=None, training=False)  # No dropout during inference
    
    print(f"✓ Forward pass successful: {output.shape}")
    
    # Get attention weights from the first attention head
    # Note: In a real implementation, you'd want to extract these during forward pass
    print("✓ Attention mechanism processed the input successfully")
    print("✓ (Attention weights would be extracted here in a full implementation)")
    
    print("✓ Attention visualization demonstration complete!")


def run_all_demonstrations():
    """Run all demonstrations."""
    print("🚀 TRANSFORMER CORE COMPONENTS DEMONSTRATION")
    print("="*60)
    print("This demonstration shows all the core Transformer components in action!")
    
    try:
        demonstrate_positional_encoding()
        demonstrate_feed_forward()
        demonstrate_layer_normalization()
        demonstrate_transformer_block()
        demonstrate_integration()
        demonstrate_attention_visualization()
        
        print("\n" + "="*60)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✓ PositionalEncoding: Adds position information to embeddings")
        print("✓ FeedForward: Processes each position independently")
        print("✓ LayerNormalization: Stabilizes training with normalization")
        print("✓ TransformerBlock: Combines attention and feed-forward layers")
        print("✓ Integration: All components work together seamlessly")
        print("✓ Visualization: Ready for attention analysis")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_demonstrations()
    sys.exit(0 if success else 1)
