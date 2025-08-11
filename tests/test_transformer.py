"""
Tests for Complete Transformer Implementation
===========================================

This module contains comprehensive tests for all transformer components:
- Embeddings (Token and Positional)
- Encoder and Decoder
- Output Projection
- Complete Transformer
"""

import sys
import os
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from transformer.embeddings import TokenEmbedding, PositionalEmbedding
from transformer.encoder import TransformerEncoder, TransformerEncoderLayer, create_padding_mask
from transformer.decoder import TransformerDecoder, TransformerDecoderLayer, create_causal_mask, create_look_ahead_mask
from transformer.output_projection import OutputProjection, softmax, log_softmax
from transformer.transformer import Transformer


class TestEmbeddings:
    """Test cases for embedding layers."""
    
    def test_token_embedding(self):
        """Test TokenEmbedding functionality."""
        vocab_size = 1000
        d_model = 512
        batch_size = 2
        seq_len = 10
        
        embedding = TokenEmbedding(vocab_size, d_model)
        
        # Test initialization
        assert embedding.vocab_size == vocab_size
        assert embedding.d_model == d_model
        assert embedding.embeddings.shape == (vocab_size, d_model)
        
        # Test forward pass
        token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
        embedded = embedding.forward(token_ids)
        assert embedded.shape == (batch_size, seq_len, d_model)
        
        # Test backward pass
        grad_output = np.random.randn(batch_size, seq_len, d_model)
        embedding.backward(grad_output)
        
        # Test parameter update
        embedding.update_parameters(0.01)
        
        # Test parameter saving/loading
        params = embedding.get_parameters()
        new_embedding = TokenEmbedding(vocab_size, d_model)
        new_embedding.set_parameters(params)
        
        print("✓ TokenEmbedding tests passed")
    
    def test_positional_embedding(self):
        """Test PositionalEmbedding functionality."""
        d_model = 512
        max_seq_len = 100
        
        embedding = PositionalEmbedding(d_model, max_seq_len)
        
        # Test initialization
        assert embedding.d_model == d_model
        assert embedding.max_seq_len == max_seq_len
        assert embedding.positional_encodings.shape == (max_seq_len, d_model)
        
        # Test forward pass
        seq_len = 20
        pos_encodings = embedding.forward(seq_len)
        assert pos_encodings.shape == (seq_len, d_model)
        
        # Test that different positions have different encodings
        pos_diff = np.abs(pos_encodings[0] - pos_encodings[1]).sum()
        assert pos_diff > 0, "Positional encodings should be different for different positions"
        
        # Test parameter saving/loading
        params = embedding.get_parameters()
        new_embedding = PositionalEmbedding(d_model, max_seq_len)
        new_embedding.set_parameters(params)
        
        print("✓ PositionalEmbedding tests passed")


class TestEncoder:
    """Test cases for encoder components."""
    
    def test_encoder_layer(self):
        """Test TransformerEncoderLayer functionality."""
        d_model = 512
        n_heads = 8
        d_ff = 2048
        batch_size = 2
        seq_len = 10
        
        encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff)
        
        # Test initialization
        assert encoder_layer.d_model == d_model
        assert encoder_layer.n_heads == n_heads
        assert encoder_layer.d_ff == d_ff
        
        # Test forward pass
        x = np.random.randn(batch_size, seq_len, d_model)
        output, attention_weights = encoder_layer.forward(x)
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
        
        # Test backward pass
        grad_output = np.random.randn(batch_size, seq_len, d_model)
        grad_input = encoder_layer.backward(grad_output)
        assert grad_input.shape == (batch_size, seq_len, d_model)
        
        # Test parameter saving/loading
        params = encoder_layer.get_parameters()
        new_layer = TransformerEncoderLayer(d_model, n_heads, d_ff)
        new_layer.set_parameters(params)
        
        print("✓ TransformerEncoderLayer tests passed")
    
    def test_encoder_stack(self):
        """Test TransformerEncoder functionality."""
        d_model = 512
        n_heads = 8
        d_ff = 2048
        n_layers = 3
        batch_size = 2
        seq_len = 10
        
        encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers)
        
        # Test initialization
        assert encoder.d_model == d_model
        assert encoder.n_heads == n_heads
        assert encoder.d_ff == d_ff
        assert encoder.n_layers == n_layers
        assert len(encoder.layers) == n_layers
        
        # Test forward pass
        x = np.random.randn(batch_size, seq_len, d_model)
        output, all_attention_weights = encoder.forward(x)
        assert output.shape == (batch_size, seq_len, d_model)
        assert len(all_attention_weights) == n_layers
        
        # Test backward pass
        grad_output = np.random.randn(batch_size, seq_len, d_model)
        grad_input = encoder.backward(grad_output)
        assert grad_input.shape == (batch_size, seq_len, d_model)
        
        # Test parameter saving/loading
        params = encoder.get_parameters()
        new_encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers)
        new_encoder.set_parameters(params)
        
        print("✓ TransformerEncoder tests passed")
    
    def test_padding_mask(self):
        """Test padding mask creation."""
        seq = np.array([[1, 2, 3, 0, 0], [1, 2, 0, 0, 0]])
        mask = create_padding_mask(seq, pad_token_id=0)
        
        assert mask.shape == (2, 1, 1, 5)
        assert mask[0, 0, 0, 3] == True  # First sequence, position 3 (pad)
        assert mask[0, 0, 0, 4] == True  # First sequence, position 4 (pad)
        assert mask[1, 0, 0, 2] == True  # Second sequence, position 2 (pad)
        assert mask[1, 0, 0, 3] == True  # Second sequence, position 3 (pad)
        assert mask[1, 0, 0, 4] == True  # Second sequence, position 4 (pad)
        
        print("✓ Padding mask tests passed")


class TestDecoder:
    """Test cases for decoder components."""
    
    def test_decoder_layer(self):
        """Test TransformerDecoderLayer functionality."""
        d_model = 512
        n_heads = 8
        d_ff = 2048
        batch_size = 2
        dec_seq_len = 8
        enc_seq_len = 10
        
        decoder_layer = TransformerDecoderLayer(d_model, n_heads, d_ff)
        
        # Test initialization
        assert decoder_layer.d_model == d_model
        assert decoder_layer.n_heads == n_heads
        assert decoder_layer.d_ff == d_ff
        
        # Test forward pass
        x = np.random.randn(batch_size, dec_seq_len, d_model)
        encoder_output = np.random.randn(batch_size, enc_seq_len, d_model)
        output, attention_weights = decoder_layer.forward(x, encoder_output)
        assert output.shape == (batch_size, dec_seq_len, d_model)
        assert len(attention_weights) == 2  # self-attention + cross-attention
        
        # Test backward pass
        grad_output = np.random.randn(batch_size, dec_seq_len, d_model)
        grad_dec, grad_enc = decoder_layer.backward(grad_output)
        assert grad_dec.shape == (batch_size, dec_seq_len, d_model)
        assert grad_enc.shape == (batch_size, dec_seq_len, d_model)
        
        # Test parameter saving/loading
        params = decoder_layer.get_parameters()
        new_layer = TransformerDecoderLayer(d_model, n_heads, d_ff)
        new_layer.set_parameters(params)
        
        print("✓ TransformerDecoderLayer tests passed")
    
    def test_decoder_stack(self):
        """Test TransformerDecoder functionality."""
        d_model = 512
        n_heads = 8
        d_ff = 2048
        n_layers = 3
        batch_size = 2
        dec_seq_len = 8
        enc_seq_len = 10
        
        decoder = TransformerDecoder(d_model, n_heads, d_ff, n_layers)
        
        # Test initialization
        assert decoder.d_model == d_model
        assert decoder.n_heads == n_heads
        assert decoder.d_ff == d_ff
        assert decoder.n_layers == n_layers
        assert len(decoder.layers) == n_layers
        
        # Test forward pass
        x = np.random.randn(batch_size, dec_seq_len, d_model)
        encoder_output = np.random.randn(batch_size, enc_seq_len, d_model)
        output, all_attention_weights = decoder.forward(x, encoder_output)
        assert output.shape == (batch_size, dec_seq_len, d_model)
        assert len(all_attention_weights) == n_layers
        assert len(all_attention_weights[0]) == 2  # self-attention + cross-attention
        
        # Test backward pass
        grad_output = np.random.randn(batch_size, dec_seq_len, d_model)
        grad_dec, grad_enc = decoder.backward(grad_output)
        assert grad_dec.shape == (batch_size, dec_seq_len, d_model)
        assert grad_enc.shape == (batch_size, dec_seq_len, d_model)
        
        # Test parameter saving/loading
        params = decoder.get_parameters()
        new_decoder = TransformerDecoder(d_model, n_heads, d_ff, n_layers)
        new_decoder.set_parameters(params)
        
        print("✓ TransformerDecoder tests passed")
    
    def test_masks(self):
        """Test mask creation functions."""
        seq_len = 5
        
        # Test causal mask
        causal_mask = create_causal_mask(seq_len)
        assert causal_mask.shape == (1, 1, seq_len, seq_len)
        assert causal_mask[0, 0, 0, 1] == True   # Position 0 can't attend to position 1
        assert causal_mask[0, 0, 1, 2] == True   # Position 1 can't attend to position 2
        assert causal_mask[0, 0, 0, 0] == False  # Position 0 can attend to itself
        
        # Test look-ahead mask
        look_ahead_mask = create_look_ahead_mask(seq_len)
        assert np.array_equal(causal_mask, look_ahead_mask)
        
        print("✓ Mask tests passed")


class TestOutputProjection:
    """Test cases for output projection layer."""
    
    def test_output_projection(self):
        """Test OutputProjection functionality."""
        d_model = 512
        vocab_size = 1000
        batch_size = 2
        seq_len = 10
        
        output_proj = OutputProjection(d_model, vocab_size)
        
        # Test initialization
        assert output_proj.d_model == d_model
        assert output_proj.vocab_size == vocab_size
        assert output_proj.weights.shape == (d_model, vocab_size)
        assert output_proj.bias.shape == (vocab_size,)
        
        # Test forward pass
        x = np.random.randn(batch_size, seq_len, d_model)
        logits = output_proj.forward(x)
        assert logits.shape == (batch_size, seq_len, vocab_size)
        
        # Test softmax
        probs = softmax(logits)
        assert probs.shape == (batch_size, seq_len, vocab_size)
        assert np.allclose(probs.sum(axis=-1), 1.0)
        
        # Test log-softmax
        log_probs = log_softmax(logits)
        assert log_probs.shape == (batch_size, seq_len, vocab_size)
        
        # Test backward pass
        grad_output = np.random.randn(batch_size, seq_len, vocab_size)
        grad_input = output_proj.backward(grad_output)
        assert grad_input.shape == (batch_size, seq_len, d_model)
        
        # Test parameter update
        output_proj.update_parameters(0.01)
        
        # Test parameter saving/loading
        params = output_proj.get_parameters()
        new_proj = OutputProjection(d_model, vocab_size)
        new_proj.set_parameters(params)
        
        print("✓ OutputProjection tests passed")


class TestCompleteTransformer:
    """Test cases for the complete Transformer model."""
    
    def test_transformer_initialization(self):
        """Test Transformer model initialization."""
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
        
        # Test initialization
        assert transformer.src_vocab_size == src_vocab_size
        assert transformer.tgt_vocab_size == tgt_vocab_size
        assert transformer.d_model == d_model
        assert transformer.n_heads == n_heads
        assert transformer.d_ff == d_ff
        assert transformer.n_encoder_layers == n_encoder_layers
        assert transformer.n_decoder_layers == n_decoder_layers
        
        print("✓ Transformer initialization tests passed")
    
    def test_transformer_forward(self):
        """Test Transformer forward pass."""
        transformer = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=800,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            n_encoder_layers=3,
            n_decoder_layers=3
        )
        
        batch_size = 2
        src_seq_len = 10
        tgt_seq_len = 8
        
        # Create input data
        src_tokens = np.random.randint(0, 1000, (batch_size, src_seq_len))
        tgt_tokens = np.random.randint(0, 800, (batch_size, tgt_seq_len))
        
        # Test forward pass
        logits, attention_weights = transformer.forward(src_tokens, tgt_tokens)
        assert logits.shape == (batch_size, tgt_seq_len, 800)
        assert 'encoder' in attention_weights
        assert 'decoder' in attention_weights
        assert len(attention_weights['encoder']) == 3
        assert len(attention_weights['decoder']) == 3
        
        print("✓ Transformer forward pass tests passed")
    
    def test_transformer_backward(self):
        """Test Transformer backward pass."""
        transformer = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=800,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            n_encoder_layers=3,
            n_decoder_layers=3
        )
        
        batch_size = 2
        src_seq_len = 10
        tgt_seq_len = 8
        
        # Create input data
        src_tokens = np.random.randint(0, 1000, (batch_size, src_seq_len))
        tgt_tokens = np.random.randint(0, 800, (batch_size, tgt_seq_len))
        
        # Forward pass
        logits, _ = transformer.forward(src_tokens, tgt_tokens)
        
        # Test backward pass
        grad_output = np.random.randn(batch_size, tgt_seq_len, 800)
        grad_src, grad_tgt = transformer.backward(grad_output)
        # Check that gradients have the correct dimensions
        assert grad_src.shape[0] == batch_size
        assert grad_src.shape[2] == 512
        assert grad_tgt.shape[0] == batch_size
        assert grad_tgt.shape[2] == 512
        
        print("✓ Transformer backward pass tests passed")
    
    def test_transformer_generation(self):
        """Test Transformer text generation."""
        transformer = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=800,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            n_encoder_layers=3,
            n_decoder_layers=3
        )
        
        batch_size = 2
        src_seq_len = 10
        
        # Create source data
        src_tokens = np.random.randint(0, 1000, (batch_size, src_seq_len))
        
        # Test generation
        generated = transformer.generate(src_tokens, max_length=5, start_token=1, end_token=2)
        assert generated.shape[0] == batch_size
        assert generated.shape[1] <= 5
        assert generated[0, 0] == 1  # Start token
        
        print("✓ Transformer generation tests passed")
    
    def test_transformer_parameters(self):
        """Test Transformer parameter saving/loading."""
        transformer = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=800,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            n_encoder_layers=3,
            n_decoder_layers=3
        )
        
        # Test parameter saving
        params = transformer.get_parameters()
        assert 'src_embedding' in params
        assert 'tgt_embedding' in params
        assert 'encoder' in params
        assert 'decoder' in params
        assert 'output_projection' in params
        
        # Test parameter loading
        new_transformer = Transformer(
            src_vocab_size=1000,
            tgt_vocab_size=800,
            d_model=512,
            n_heads=8,
            d_ff=2048,
            n_encoder_layers=3,
            n_decoder_layers=3
        )
        new_transformer.set_parameters(params)
        
        print("✓ Transformer parameter tests passed")


def run_all_tests():
    """Run all test suites."""
    print("🚀 Starting Complete Transformer Tests...\n")
    
    # Test embeddings
    print("Testing Embeddings...")
    test_embeddings = TestEmbeddings()
    test_embeddings.test_token_embedding()
    test_embeddings.test_positional_embedding()
    print()
    
    # Test encoder
    print("Testing Encoder...")
    test_encoder = TestEncoder()
    test_encoder.test_encoder_layer()
    test_encoder.test_encoder_stack()
    test_encoder.test_padding_mask()
    print()
    
    # Test decoder
    print("Testing Decoder...")
    test_decoder = TestDecoder()
    test_decoder.test_decoder_layer()
    test_decoder.test_decoder_stack()
    test_decoder.test_masks()
    print()
    
    # Test output projection
    print("Testing Output Projection...")
    test_output_proj = TestOutputProjection()
    test_output_proj.test_output_projection()
    print()
    
    # Test complete transformer
    print("Testing Complete Transformer...")
    test_complete = TestCompleteTransformer()
    test_complete.test_transformer_initialization()
    test_complete.test_transformer_forward()
    test_complete.test_transformer_backward()
    test_complete.test_transformer_generation()
    test_complete.test_transformer_parameters()
    print()
    
    print("🎉 All Complete Transformer tests passed successfully! 🎉")


if __name__ == "__main__":
    run_all_tests()
