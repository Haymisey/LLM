"""
Complete Transformer Examples
============================

This module provides comprehensive examples demonstrating the usage of
all transformer components and the complete model.
"""

import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from transformer.embeddings import TokenEmbedding, PositionalEmbedding
from transformer.encoder import TransformerEncoder, TransformerEncoderLayer, create_padding_mask
from transformer.decoder import TransformerDecoder, TransformerDecoderLayer, create_causal_mask, create_look_ahead_mask
from transformer.output_projection import OutputProjection, softmax, log_softmax
from transformer.transformer import Transformer


def example_embeddings():
    """Demonstrate embedding layers functionality."""
    print("🔤 Embedding Layers Examples")
    print("=" * 50)
    
    # Token Embedding Example
    print("\n1. Token Embedding:")
    vocab_size = 1000
    d_model = 512
    batch_size = 2
    seq_len = 10
    
    token_embedding = TokenEmbedding(vocab_size, d_model)
    
    # Create sample token IDs
    token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input token IDs shape: {token_ids.shape}")
    print(f"Sample tokens: {token_ids[0, :5]}")  # First 5 tokens of first sequence
    
    # Forward pass
    embedded = token_embedding.forward(token_ids)
    print(f"Embedded output shape: {embedded.shape}")
    print(f"Embedding dimension: {embedded.shape[-1]}")
    
    # Positional Embedding Example
    print("\n2. Positional Embedding:")
    pos_embedding = PositionalEmbedding(d_model, max_seq_len=100)
    
    # Get positional encodings for sequence length
    pos_encodings = pos_embedding.forward(seq_len)
    print(f"Positional encodings shape: {pos_encodings.shape}")
    
    # Check that different positions have different encodings
    pos_diff = np.abs(pos_encodings[0] - pos_encodings[1]).sum()
    print(f"Difference between positions 0 and 1: {pos_diff:.6f}")
    
    # Combine token and positional embeddings
    combined = embedded + pos_encodings.reshape(1, seq_len, d_model)
    print(f"Combined embeddings shape: {combined.shape}")
    
    print("✓ Embedding examples completed!\n")


def example_encoder():
    """Demonstrate encoder functionality."""
    print("🔒 Encoder Examples")
    print("=" * 50)
    
    # Encoder Layer Example
    print("\n1. Single Encoder Layer:")
    d_model = 512
    n_heads = 8
    d_ff = 2048
    batch_size = 2
    seq_len = 10
    
    encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff)
    
    # Create input
    x = np.random.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output, attention_weights = encoder_layer.forward(x)
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attention_weights.shape}")
    print(f"Number of attention heads: {attention_weights.shape[1]}")
    
    # Encoder Stack Example
    print("\n2. Encoder Stack:")
    n_layers = 3
    encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers)
    
    # Forward pass through stack
    output, all_attention_weights = encoder.forward(x)
    print(f"Encoder stack output shape: {output.shape}")
    print(f"Number of layers: {len(all_attention_weights)}")
    print(f"Attention weights per layer: {all_attention_weights[0].shape}")
    
    # Padding Mask Example
    print("\n3. Padding Mask:")
    seq = np.array([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
    mask = create_padding_mask(seq, pad_token_id=0)
    print(f"Sequence: {seq}")
    print(f"Padding mask shape: {mask.shape}")
    print(f"Mask for first sequence: {mask[0, 0, 0]}")
    print(f"Mask for second sequence: {mask[1, 0, 0]}")
    
    print("✓ Encoder examples completed!\n")


def example_decoder():
    """Demonstrate decoder functionality."""
    print("🔓 Decoder Examples")
    print("=" * 50)
    
    # Decoder Layer Example
    print("\n1. Single Decoder Layer:")
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
    print(f"Decoder input shape: {x.shape}")
    print(f"Encoder output shape: {encoder_output.shape}")
    
    # Forward pass
    output, attention_weights = decoder_layer.forward(x, encoder_output)
    print(f"Decoder output shape: {output.shape}")
    print(f"Number of attention weight sets: {len(attention_weights)}")
    print(f"Self-attention weights shape: {attention_weights[0].shape}")
    print(f"Cross-attention weights shape: {attention_weights[1].shape}")
    
    # Decoder Stack Example
    print("\n2. Decoder Stack:")
    n_layers = 3
    decoder = TransformerDecoder(d_model, n_heads, d_ff, n_layers)
    
    # Forward pass through stack
    output, all_attention_weights = decoder.forward(x, encoder_output)
    print(f"Decoder stack output shape: {output.shape}")
    print(f"Number of layers: {len(all_attention_weights)}")
    print(f"Attention weights per layer: {len(all_attention_weights[0])} sets")
    
    # Mask Examples
    print("\n3. Attention Masks:")
    seq_len = 5
    
    # Causal mask
    causal_mask = create_causal_mask(seq_len)
    print(f"Causal mask shape: {causal_mask.shape}")
    print(f"Causal mask:\n{causal_mask[0, 0]}")
    
    # Look-ahead mask
    look_ahead_mask = create_look_ahead_mask(seq_len)
    print(f"Look-ahead mask identical to causal: {np.array_equal(causal_mask, look_ahead_mask)}")
    
    print("✓ Decoder examples completed!\n")


def example_output_projection():
    """Demonstrate output projection functionality."""
    print("📤 Output Projection Examples")
    print("=" * 50)
    
    # Output Projection Example
    print("\n1. Output Projection Layer:")
    d_model = 512
    vocab_size = 1000
    batch_size = 2
    seq_len = 10
    
    output_proj = OutputProjection(d_model, vocab_size)
    
    # Create input
    x = np.random.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    logits = output_proj.forward(x)
    print(f"Logits shape: {logits.shape}")
    print(f"Vocabulary size: {logits.shape[-1]}")
    
    # Softmax and Log-softmax
    probs = softmax(logits)
    log_probs = log_softmax(logits)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Log-probabilities shape: {log_probs.shape}")
    
    # Check probability normalization
    prob_sums = probs.sum(axis=-1)
    print(f"Probability sums (should be 1.0): {prob_sums[0, :5]}")
    
    # Backward pass
    grad_output = np.random.randn(batch_size, seq_len, vocab_size)
    grad_input = output_proj.backward(grad_output)
    print(f"Gradient input shape: {grad_input.shape}")
    
    print("✓ Output projection examples completed!\n")


def example_complete_transformer():
    """Demonstrate complete transformer functionality."""
    print("🚀 Complete Transformer Examples")
    print("=" * 50)
    
    # Create Transformer Model
    print("\n1. Model Creation:")
    src_vocab_size = 1000
    tgt_vocab_size = 800
    d_model = 512
    n_heads = 8
    d_ff = 2048
    n_encoder_layers = 3
    n_decoder_layers = 3
    
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers
    )
    
    print(f"Transformer created with:")
    print(f"  Source vocabulary: {transformer.src_vocab_size}")
    print(f"  Target vocabulary: {transformer.tgt_vocab_size}")
    print(f"  Model dimension: {transformer.d_model}")
    print(f"  Attention heads: {transformer.n_heads}")
    print(f"  Feed-forward dim: {transformer.d_ff}")
    print(f"  Encoder layers: {transformer.n_encoder_layers}")
    print(f"  Decoder layers: {transformer.n_decoder_layers}")
    
    # Forward Pass Example
    print("\n2. Forward Pass:")
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    # Create sample data
    src_tokens = np.random.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt_tokens = np.random.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    print(f"Source tokens shape: {src_tokens.shape}")
    print(f"Target tokens shape: {tgt_tokens.shape}")
    print(f"Sample source tokens: {src_tokens[0, :5]}")
    print(f"Sample target tokens: {tgt_tokens[0, :5]}")
    
    # Forward pass
    logits, attention_weights = transformer.forward(src_tokens, tgt_tokens)
    print(f"Output logits shape: {logits.shape}")
    print(f"Encoder attention weights: {len(attention_weights['encoder'])} layers")
    print(f"Decoder attention weights: {len(attention_weights['decoder'])} layers")
    
    # Backward Pass Example
    print("\n3. Backward Pass:")
    grad_output = np.random.randn(batch_size, tgt_seq_len, tgt_vocab_size)
    grad_src, grad_tgt = transformer.backward(grad_output)
    print(f"Gradient source shape: {grad_src.shape}")
    print(f"Gradient target shape: {grad_tgt.shape}")
    
    # Text Generation Example
    print("\n4. Text Generation:")
    generated = transformer.generate(
        src_tokens, 
        max_length=6, 
        start_token=1, 
        end_token=2
    )
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated tokens:\n{generated}")
    
    # Parameter Management Example
    print("\n5. Parameter Management:")
    params = transformer.get_parameters()
    print(f"Number of parameter groups: {len(params)}")
    print(f"Parameter keys: {list(params.keys())}")
    
    # Create new model and load parameters
    new_transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers
    )
    new_transformer.set_parameters(params)
    print("✓ Parameters loaded into new model")
    
    print("✓ Complete transformer examples completed!\n")


def example_translation_scenario():
    """Demonstrate a realistic translation scenario."""
    print("🌍 Translation Scenario Example")
    print("=" * 50)
    
    # Create a smaller model for demonstration
    print("\n1. Setting up Translation Model:")
    src_vocab_size = 5000  # Amharic vocabulary
    tgt_vocab_size = 4000  # Oromiffa vocabulary
    d_model = 256
    n_heads = 4
    d_ff = 1024
    n_encoder_layers = 2
    n_decoder_layers = 2
    
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers
    )
    
    print(f"Translation model created:")
    print(f"  Source (Amharic) vocabulary: {src_vocab_size}")
    print(f"  Target (Oromiffa) vocabulary: {tgt_vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention heads: {n_heads}")
    print(f"  Encoder/Decoder layers: {n_encoder_layers}")
    
    # Simulate translation process
    print("\n2. Translation Process:")
    batch_size = 1
    src_seq_len = 8
    
    # Simulate Amharic input (random token IDs)
    amharic_tokens = np.random.randint(0, src_vocab_size, (batch_size, src_seq_len))
    print(f"Amharic input tokens: {amharic_tokens[0]}")
    
    # Generate Oromiffa translation
    oromiffa_tokens = transformer.generate(
        amharic_tokens,
        max_length=10,
        start_token=1,  # Start of sequence
        end_token=2     # End of sequence
    )
    print(f"Generated Oromiffa tokens: {oromiffa_tokens[0]}")
    
    # Show attention analysis
    print("\n3. Attention Analysis:")
    # Get attention weights for analysis
    _, attention_weights = transformer.forward(amharic_tokens, oromiffa_tokens)
    
    print(f"Encoder self-attention layers: {len(attention_weights['encoder'])}")
    print(f"Decoder self-attention layers: {len(attention_weights['decoder'])}")
    print(f"Decoder cross-attention layers: {len(attention_weights['decoder'])}")
    
    # Analyze attention patterns
    for i, layer_weights in enumerate(attention_weights['encoder']):
        print(f"  Encoder layer {i+1}: {layer_weights.shape}")
    
    for i, layer_weights in enumerate(attention_weights['decoder']):
        print(f"  Decoder layer {i+1}: {len(layer_weights)} attention types")
        print(f"    Self-attention: {layer_weights[0].shape}")
        print(f"    Cross-attention: {layer_weights[1].shape}")
    
    print("\n4. Model Performance:")
    print(f"Input sequence length: {src_seq_len}")
    print(f"Output sequence length: {oromiffa_tokens.shape[1]}")
    print(f"Model parameters: {sum(len(str(v)) for v in transformer.get_parameters().values())} groups")
    
    print("✓ Translation scenario example completed!\n")


def run_all_examples():
    """Run all transformer examples."""
    print("🎯 Complete Transformer Implementation Examples")
    print("=" * 60)
    print("This demonstrates the full transformer architecture from scratch!\n")
    
    # Run all examples
    example_embeddings()
    example_encoder()
    example_decoder()
    example_output_projection()
    example_complete_transformer()
    example_translation_scenario()
    
    print("🎉 All transformer examples completed successfully!")
    print("\n🏆 Checkpoint 5 Complete: Complete Transformer!")

if __name__ == "__main__":
    run_all_examples()
