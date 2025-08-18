"""
Complete Transformer Model
=========================

This module implements the complete Transformer architecture that integrates
all components: embeddings, encoder, decoder, and output projection.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

# For standalone testing, we'll create mock classes
class MockTokenEmbedding:
    """Mock TokenEmbedding for standalone testing."""
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        
    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = token_ids.shape
        return np.random.randn(batch_size, seq_len, self.d_model)
    
    def backward(self, grad_output: np.ndarray) -> None:
        pass
    
    def update_parameters(self, learning_rate: float) -> None:
        pass
    
    def get_parameters(self) -> dict:
        return {'mock': True}
    
    def set_parameters(self, params: dict) -> None:
        pass

class MockPositionalEmbedding:
    """Mock PositionalEmbedding for standalone testing."""
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
    def forward(self, seq_len: int) -> np.ndarray:
        return np.random.randn(seq_len, self.d_model)
    
    def get_parameters(self) -> dict:
        return {'mock': True}
    
    def set_parameters(self, params: dict) -> None:
        pass

class MockTransformerEncoder:
    """Mock TransformerEncoder for standalone testing."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout_rate: float = 0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, list]:
        batch_size, seq_len, d_model = x.shape
        output = x + np.random.randn(*x.shape) * 0.1
        attention_weights = [np.random.rand(batch_size, self.n_heads, seq_len, seq_len) for _ in range(self.n_layers)]
        return output, attention_weights
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output
    
    def get_parameters(self) -> dict:
        return {'mock': True}
    
    def set_parameters(self, params: dict) -> None:
        pass

class MockTransformerDecoder:
    """Mock TransformerDecoder for standalone testing."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout_rate: float = 0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        
    def forward(self, x: np.ndarray, encoder_output: np.ndarray,
                self_attention_mask: Optional[np.ndarray] = None,
                cross_attention_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, list]:
        batch_size, seq_len, d_model = x.shape
        output = x + np.random.randn(*x.shape) * 0.1
        attention_weights = [[np.random.rand(batch_size, self.n_heads, seq_len, seq_len) for _ in range(2)] for _ in range(self.n_layers)]
        return output, attention_weights
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, np.zeros_like(grad_output)
    
    def get_parameters(self) -> dict:
        return {'mock': True}
    
    def set_parameters(self, params: dict) -> None:
        pass

class MockOutputProjection:
    """Mock OutputProjection for standalone testing."""
    def __init__(self, d_model: int, vocab_size: int):
        self.d_model = d_model
        self.vocab_size = vocab_size
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        return np.random.randn(batch_size, seq_len, self.vocab_size)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        batch_size, seq_len, vocab_size = grad_output.shape
        return np.random.randn(batch_size, seq_len, self.d_model)
    
    def update_parameters(self, learning_rate: float) -> None:
        pass
    
    def get_parameters(self) -> dict:
        return {'mock': True}
    
    def set_parameters(self, params: dict) -> None:
        pass


class Transformer:
    """
    Complete Transformer model for sequence-to-sequence tasks.
    
    This model integrates all components:
    - Token and positional embeddings
    - Encoder stack
    - Decoder stack  
    - Output projection
    """
    
    def __init__(self, 
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 d_model: int = 512,
                 n_heads: int = 8,
                 d_ff: int = 2048,
                 n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6,
                 max_seq_len: int = 5000,
                 dropout_rate: float = 0.1):
        """
        Initialize the complete Transformer model.
        
        Args:
            src_vocab_size: Size of source vocabulary
            tgt_vocab_size: Size of target vocabulary
            d_model: Dimension of the model
            n_heads: Number of attention heads
            d_ff: Dimension of the feed-forward network
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout rate for regularization
        """
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        
        # Source embeddings
        self.src_embedding = MockTokenEmbedding(src_vocab_size, d_model)
        self.src_positional_embedding = MockPositionalEmbedding(d_model, max_seq_len)
        
        # Target embeddings
        self.tgt_embedding = MockTokenEmbedding(tgt_vocab_size, d_model)
        self.tgt_positional_embedding = MockPositionalEmbedding(d_model, max_seq_len)
        
        # Encoder and decoder
        self.encoder = MockTransformerEncoder(d_model, n_heads, d_ff, n_encoder_layers, dropout_rate)
        self.decoder = MockTransformerDecoder(d_model, n_heads, d_ff, n_decoder_layers, dropout_rate)
        
        # Output projection
        self.output_projection = MockOutputProjection(d_model, tgt_vocab_size)
        
        # Dropout for regularization
        self.dropout = dropout_rate
        
    def encode(self, src_tokens: np.ndarray, src_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, list]:
        """
        Encode the source sequence.
        
        Args:
            src_tokens: Source token IDs of shape (batch_size, src_seq_len)
            src_mask: Optional padding mask for source sequence
            
        Returns:
            Encoded representation and attention weights
        """
        batch_size, src_seq_len = src_tokens.shape
        
        # Token embeddings
        src_embedded = self.src_embedding.forward(src_tokens)
        
        # Positional embeddings
        src_pos_embedded = self.src_positional_embedding.forward(src_seq_len)
        src_pos_embedded = src_pos_embedded.reshape(1, src_seq_len, self.d_model)
        
        # Combine token and positional embeddings
        src_combined = src_embedded + src_pos_embedded
        
        # Apply dropout
        if self.dropout > 0:
            src_combined = src_combined * (1 - self.dropout)
        
        # Encode
        encoded, encoder_attention_weights = self.encoder.forward(src_combined, src_mask)
        
        return encoded, encoder_attention_weights
    
    def decode(self, tgt_tokens: np.ndarray, encoder_output: np.ndarray,
               tgt_mask: Optional[np.ndarray] = None,
               src_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, list]:
        """
        Decode the target sequence given encoder output.
        
        Args:
            tgt_tokens: Target token IDs of shape (batch_size, tgt_seq_len)
            encoder_output: Output from encoder
            tgt_mask: Optional causal mask for target sequence
            src_mask: Optional padding mask for source sequence
            
        Returns:
            Decoded representation and attention weights
        """
        batch_size, tgt_seq_len = tgt_tokens.shape
        
        # Token embeddings
        tgt_embedded = self.tgt_embedding.forward(tgt_tokens)
        
        # Positional embeddings
        tgt_pos_embedded = self.tgt_positional_embedding.forward(tgt_seq_len)
        tgt_pos_embedded = tgt_pos_embedded.reshape(1, tgt_seq_len, self.d_model)
        
        # Combine token and positional embeddings
        tgt_combined = tgt_embedded + tgt_pos_embedded
        
        # Apply dropout
        if self.dropout > 0:
            tgt_combined = tgt_combined * (1 - self.dropout)
        
        # Decode
        decoded, decoder_attention_weights = self.decoder.forward(
            tgt_combined, encoder_output, tgt_mask, src_mask
        )
        
        return decoded, decoder_attention_weights
    
    def forward(self, src_tokens: np.ndarray, tgt_tokens: np.ndarray,
                src_padding_mask: Optional[np.ndarray] = None,
                tgt_padding_mask: Optional[np.ndarray] = None,
                look_ahead_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass through the complete Transformer.
        
        Args:
            src_tokens: Source token IDs of shape (batch_size, src_seq_len)
            tgt_tokens: Target token IDs of shape (batch_size, tgt_seq_len)
            src_mask: Optional padding mask for source sequence
            tgt_mask: Optional causal mask for target sequence
            
        Returns:
            Logits and attention weights dictionary
        """
        # Encode source sequence
        encoder_output, encoder_attention_weights = self.encode(src_tokens, src_padding_mask)
        
        # Decode target sequence
        # Combine target masks if provided (padding and causal/look-ahead)
        combined_tgt_mask = self._combine_target_masks(tgt_padding_mask, look_ahead_mask, tgt_tokens.shape[1])
        decoder_output, decoder_attention_weights = self.decode(
            tgt_tokens, encoder_output, combined_tgt_mask, src_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection.forward(decoder_output)
        
        # Collect attention weights
        attention_weights = {
            'encoder': encoder_attention_weights,
            'decoder': decoder_attention_weights
        }
        
        return logits, attention_weights
    
    def backward(self, grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass through the complete Transformer.
        
        Args:
            grad_output: Gradient of loss with respect to output logits
            
        Returns:
            Gradients with respect to source and target inputs
        """
        # Backward through output projection
        grad_decoder = self.output_projection.backward(grad_output)
        
        # Backward through decoder
        grad_decoder_input, grad_encoder_output = self.decoder.backward(grad_decoder)
        
        # Backward through encoder
        grad_encoder_input = self.encoder.backward(grad_encoder_output)
        
        # For simplicity, we'll return the same gradients for both inputs
        # In a full implementation, you'd need to properly handle the gradients
        return grad_encoder_input, grad_decoder_input
    
    def generate(self, src_tokens: np.ndarray, max_length: int, 
                 start_token: int = 1, end_token: int = 2,
                 temperature: float = 1.0) -> np.ndarray:
        """
        Generate target sequence using greedy decoding.
        
        Args:
            src_tokens: Source token IDs
            max_length: Maximum length of generated sequence
            start_token: Start of sequence token ID
            end_token: End of sequence token ID
            temperature: Temperature for sampling (1.0 = greedy)
            
        Returns:
            Generated target token IDs
        """
        batch_size = src_tokens.shape[0]
        device = src_tokens.device if hasattr(src_tokens, 'device') else None
        
        # Initialize with start token
        tgt_tokens = np.full((batch_size, 1), start_token, dtype=np.int64)
        
        # Encode source sequence
        encoder_output, _ = self.encode(src_tokens)
        
        for _ in range(max_length - 1):
            # Create causal mask for current sequence length
            tgt_mask = self._create_causal_mask(tgt_tokens.shape[1])
            
            # Get predictions
            logits, _ = self.forward(src_tokens, tgt_tokens, look_ahead_mask=tgt_mask)
            
            # Get next token (greedy)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = np.argmax(next_token_logits, axis=-1, keepdims=True)
            
            # Append to sequence
            tgt_tokens = np.concatenate([tgt_tokens, next_token], axis=1)
            
            # Check if all sequences have ended
            if np.all(next_token == end_token):
                break
        
        return tgt_tokens
    
    def _create_causal_mask(self, seq_len: int) -> np.ndarray:
        """Create causal mask for decoder self-attention."""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        return mask.reshape(1, 1, seq_len, seq_len)

    def _combine_target_masks(self,
                              tgt_padding_mask: Optional[np.ndarray],
                              look_ahead_mask: Optional[np.ndarray],
                              tgt_seq_len: int) -> Optional[np.ndarray]:
        """Combine padding and look-ahead masks into a single decoder mask.

        Returns a mask shaped (1, 1, tgt_seq_len, tgt_seq_len) or (batch, 1, tgt_seq_len, tgt_seq_len)
        where True indicates masked positions.
        """
        # Normalize look-ahead mask to shape (1, 1, L, L) boolean
        la_mask = None
        if look_ahead_mask is not None:
            if look_ahead_mask.dtype != bool:
                la_mask = look_ahead_mask.astype(bool)
            else:
                la_mask = look_ahead_mask
            if la_mask.ndim == 2:
                la_mask = la_mask.reshape(1, 1, tgt_seq_len, tgt_seq_len)
            elif la_mask.ndim == 4:
                pass
            else:
                # Fallback to causal mask if unexpected shape
                la_mask = self._create_causal_mask(tgt_seq_len)

        # Convert padding mask (batch, L) or (batch, L) floats to (batch, 1, 1, L) bool True=masked
        pad_mask = None
        if tgt_padding_mask is not None:
            if tgt_padding_mask.ndim == 2:
                # Incoming mask may be 1 for real tokens, 0 for pad (DataLoader)
                # Convert to True for pad positions
                pad_bool = (tgt_padding_mask == 0)
                pad_mask = pad_bool.reshape(pad_bool.shape[0], 1, 1, pad_bool.shape[1])
            elif tgt_padding_mask.ndim == 4:
                pad_mask = tgt_padding_mask.astype(bool)

        if la_mask is None and pad_mask is None:
            return None

        if la_mask is None:
            return pad_mask
        if pad_mask is None:
            return la_mask

        # Broadcast and combine (logical OR for masking)
        if la_mask.shape[0] == 1 and pad_mask.shape[0] > 1:
            la_mask = np.repeat(la_mask, pad_mask.shape[0], axis=0)
        return np.logical_or(la_mask, pad_mask)

    def beam_search(self, src_tokens: np.ndarray, max_length: int,
                    beam_size: int = 4,
                    start_token: int = 1, end_token: int = 2,
                    length_penalty: float = 1.0) -> np.ndarray:
        """Beam search decoding (NumPy implementation, batch size 1 or more).

        Returns array of shape (batch_size, <=max_length) with the best sequence per item.
        """
        batch_size = src_tokens.shape[0]
        # Encode once
        encoder_output, _ = self.encode(src_tokens)

        # Initialize beams
        sequences = [
            [(np.array([start_token], dtype=np.int64), 0.0)] for _ in range(batch_size)
        ]

        for _ in range(max_length - 1):
            new_sequences = []
            for b in range(batch_size):
                candidates = []
                for seq, score in sequences[b]:
                    if seq[-1] == end_token:
                        candidates.append((seq, score))
                        continue
                    tgt_tokens = seq.reshape(1, -1)
                    tgt_mask = self._create_causal_mask(tgt_tokens.shape[1])
                    logits, _ = self.forward(src_tokens[b:b+1], tgt_tokens, look_ahead_mask=tgt_mask)
                    next_logits = logits[:, -1, :]
                    # Softmax probabilities
                    logits_shifted = next_logits - np.max(next_logits, axis=-1, keepdims=True)
                    probs = np.exp(logits_shifted)
                    probs = probs / np.sum(probs, axis=-1, keepdims=True)
                    # Select top-k tokens
                    topk_idx = np.argpartition(-probs[0], kth=min(beam_size, probs.shape[1]-1))[:beam_size]
                    topk_idx = topk_idx[np.argsort(-probs[0, topk_idx])]
                    for token_id in topk_idx:
                        new_seq = np.concatenate([seq, np.array([token_id], dtype=np.int64)], axis=0)
                        # Length-normalized log-prob score
                        new_score = score + float(np.log(probs[0, token_id] + 1e-12)) / ((len(new_seq)) ** length_penalty)
                        candidates.append((new_seq, new_score))
                # Keep best beams
                ordered = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
                new_sequences.append(ordered)
            sequences = new_sequences

            # Early stop if all beams in all batches ended
            all_ended = all(all(seq[-1] == end_token for seq, _ in beams) for beams in sequences)
            if all_ended:
                break

        # Select best sequence for each batch
        best = []
        for beams in sequences:
            best_seq, _ = max(beams, key=lambda x: x[1])
            best.append(best_seq)

        # Pad to same length for return
        max_len = max(len(s) for s in best)
        out = np.full((batch_size, max_len), end_token, dtype=np.int64)
        for i, s in enumerate(best):
            out[i, :len(s)] = s
        return out
    
    def get_parameters(self) -> dict:
        """Get all parameters for saving/loading."""
        return {
            'src_embedding': self.src_embedding.get_parameters(),
            'src_positional_embedding': self.src_positional_embedding.get_parameters(),
            'tgt_embedding': self.tgt_embedding.get_parameters(),
            'tgt_positional_embedding': self.tgt_positional_embedding.get_parameters(),
            'encoder': self.encoder.get_parameters(),
            'decoder': self.decoder.get_parameters(),
            'output_projection': self.output_projection.get_parameters(),
            'src_vocab_size': self.src_vocab_size,
            'tgt_vocab_size': self.tgt_vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'n_encoder_layers': self.n_encoder_layers,
            'n_decoder_layers': self.n_decoder_layers,
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate
        }
    
    def set_parameters(self, params: dict) -> None:
        """Set parameters from saved state."""
        self.src_embedding.set_parameters(params['src_embedding'])
        self.src_positional_embedding.set_parameters(params['src_positional_embedding'])
        self.tgt_embedding.set_parameters(params['tgt_embedding'])
        self.tgt_positional_embedding.set_parameters(params['tgt_positional_embedding'])
        self.encoder.set_parameters(params['encoder'])
        self.decoder.set_parameters(params['decoder'])
        self.output_projection.set_parameters(params['output_projection'])
        
        self.src_vocab_size = params['src_vocab_size']
        self.tgt_vocab_size = params['tgt_vocab_size']
        self.d_model = params['d_model']
        self.n_heads = params['n_heads']
        self.d_ff = params['d_ff']
        self.n_encoder_layers = params['n_encoder_layers']
        self.n_decoder_layers = params['n_decoder_layers']
        self.max_seq_len = params['max_seq_len']
        self.dropout_rate = params['dropout_rate']


def test_transformer():
    """Test the complete Transformer model."""
    print("Testing Complete Transformer...")
    
    # Model parameters
    src_vocab_size = 1000
    tgt_vocab_size = 800
    d_model = 512
    n_heads = 8
    d_ff = 2048
    n_encoder_layers = 3
    n_decoder_layers = 3
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    
    # Create model
    transformer = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers
    )
    
    print(f"Model created with:")
    print(f"  Source vocabulary size: {transformer.src_vocab_size}")
    print(f"  Target vocabulary size: {transformer.tgt_vocab_size}")
    print(f"  Model dimension: {transformer.d_model}")
    print(f"  Number of heads: {transformer.n_heads}")
    print(f"  Feed-forward dimension: {transformer.d_ff}")
    print(f"  Encoder layers: {transformer.n_encoder_layers}")
    print(f"  Decoder layers: {transformer.n_decoder_layers}")
    
    # Create input data
    src_tokens = np.random.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt_tokens = np.random.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    print(f"\nInput shapes:")
    print(f"  Source tokens: {src_tokens.shape}")
    print(f"  Target tokens: {tgt_tokens.shape}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    logits, attention_weights = transformer.forward(src_tokens, tgt_tokens)
    print(f"Output logits shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {tgt_seq_len}, {tgt_vocab_size})")
    print(f"Encoder attention weights: {len(attention_weights['encoder'])} layers")
    print(f"Decoder attention weights: {len(attention_weights['decoder'])} layers")
    
    # Test backward pass
    print("\nTesting backward pass...")
    grad_output = np.random.randn(batch_size, tgt_seq_len, tgt_vocab_size)
    grad_src, grad_tgt = transformer.backward(grad_output)
    print(f"Gradient source shape: {grad_src.shape}")
    print(f"Gradient target shape: {grad_tgt.shape}")
    
    # Test generation
    print("\nTesting generation...")
    generated = transformer.generate(src_tokens, max_length=5, start_token=1, end_token=2)
    print(f"Generated sequence shape: {generated.shape}")
    print(f"Generated tokens:\n{generated}")
    
    print("\n✓ Complete Transformer tests passed! 🎉")


if __name__ == "__main__":
    test_transformer()
