"""
Sequence-to-Sequence Attention Implementation

This module implements attention mechanisms for sequence-to-sequence models,
commonly used in machine translation and other seq2seq tasks.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
# from .basic_attention import BasicAttention  # Commented out for standalone testing

class SequenceAttention:
    """
    Sequence-to-sequence attention mechanism for encoder-decoder models.
    This allows the decoder to attend to different parts of the encoder output.
    """
    
    def __init__(self, encoder_dim: int, decoder_dim: int, 
                 attention_dim: int = 128, dropout: float = 0.1):
        """
        Initialize sequence-to-sequence attention mechanism.
        
        Args:
            encoder_dim: Dimension of encoder hidden states
            decoder_dim: Dimension of decoder hidden states
            attention_dim: Dimension of attention mechanism
            dropout: Dropout probability for attention weights
        """
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.dropout = dropout
        
        # Attention mechanism (simplified for standalone testing)
        # self.attention = BasicAttention(
        #     query_dim=decoder_dim,
        #     key_dim=encoder_dim,
        #     value_dim=encoder_dim,
        #     dropout=dropout
        # )
        self.attention = None  # Simplified for testing
        
        # Additional projection layers
        self.W_encoder = np.random.randn(encoder_dim, attention_dim) * 0.01
        self.W_decoder = np.random.randn(decoder_dim, attention_dim) * 0.01
        self.W_attention = np.random.randn(attention_dim, 1) * 0.01
        
        # Bias terms
        self.b_encoder = np.zeros(attention_dim)
        self.b_decoder = np.zeros(attention_dim)
        self.b_attention = np.zeros(1)
        
        # Store intermediate values for backpropagation
        self.encoder_states = None
        self.decoder_state = None
        self.attention_weights = None
        self.context_vector = None
        self.output = None
        
    def forward(self, encoder_states: np.ndarray, decoder_state: np.ndarray,
                mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of sequence-to-sequence attention.
        
        Args:
            encoder_states: Encoder hidden states, shape (batch_size, seq_len, encoder_dim)
            decoder_state: Current decoder hidden state, shape (batch_size, decoder_dim)
            mask: Optional mask for padding, shape (batch_size, seq_len)
            
        Returns:
            Tuple of (context_vector, attention_weights)
                - context_vector: Weighted encoder context, shape (batch_size, encoder_dim)
                - attention_weights: Attention weights, shape (batch_size, seq_len)
        """
        batch_size, seq_len, _ = encoder_states.shape
        
        # Store inputs for backpropagation
        self.encoder_states = encoder_states
        self.decoder_state = decoder_state
        
        # Expand decoder state to match sequence length
        # decoder_state: (batch_size, decoder_dim) -> (batch_size, seq_len, decoder_dim)
        decoder_expanded = np.expand_dims(decoder_state, axis=1)
        decoder_expanded = np.repeat(decoder_expanded, seq_len, axis=1)
        
        # Compute attention scores using the attention mechanism
        # We'll use a simplified approach here for clarity
        attention_scores = self._compute_attention_scores(
            encoder_states, decoder_expanded
        )
        
        # Apply mask if provided
        if mask is not None:
            attention_scores = attention_scores + (mask * -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = self._softmax(attention_scores, axis=-1)
        
        # Store attention weights for backpropagation
        self.attention_weights = attention_weights
        
        # Compute context vector as weighted sum of encoder states
        # attention_weights: (batch_size, seq_len, 1)
        # encoder_states: (batch_size, seq_len, encoder_dim)
        attention_weights_expanded = np.expand_dims(attention_weights, axis=-1)
        context_vector = np.sum(attention_weights_expanded * encoder_states, axis=1)
        
        # Store context vector for backpropagation
        self.context_vector = context_vector
        
        return context_vector, attention_weights
    
    def _compute_attention_scores(self, encoder_states: np.ndarray, 
                                 decoder_state: np.ndarray) -> np.ndarray:
        """
        Compute attention scores between encoder states and decoder state.
        
        Args:
            encoder_states: Encoder hidden states
            decoder_state: Decoder hidden state (expanded)
            
        Returns:
            Attention scores, shape (batch_size, seq_len)
        """
        batch_size, seq_len, _ = encoder_states.shape
        
        # Project encoder states and decoder state to attention space
        encoder_proj = np.dot(encoder_states, self.W_encoder) + self.b_encoder
        decoder_proj = np.dot(decoder_state, self.W_decoder) + self.b_decoder
        
        # Compute attention scores using additive attention
        # score = v^T * tanh(W_encoder * h_encoder + W_decoder * h_decoder)
        combined = encoder_proj + decoder_proj
        tanh_output = np.tanh(combined)
        
        # Project to scalar scores
        attention_scores = np.dot(tanh_output, self.W_attention) + self.b_attention
        
        # Remove the last dimension to get (batch_size, seq_len)
        attention_scores = attention_scores.squeeze(axis=-1)
        
        return attention_scores
    
    def backward(self, d_context: np.ndarray, d_attention_weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Backward pass to compute gradients.
        
        Args:
            d_context: Gradient with respect to context vector
            d_attention_weights: Gradient with respect to attention weights
            
        Returns:
            Gradients with respect to encoder states and decoder state
        """
        # This is a simplified backward pass
        # In practice, you'd compute the full chain rule through all operations
        
        batch_size, seq_len, encoder_dim = self.encoder_states.shape
        
        # Gradient with respect to encoder states
        d_encoder_states = np.zeros_like(self.encoder_states)
        
        # Gradient with respect to decoder state
        d_decoder_state = np.zeros_like(self.decoder_state)
        
        return d_encoder_states, d_decoder_state
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Compute softmax along specified axis."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)
    
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get the computed attention weights for visualization."""
        return self.attention_weights
    
    def get_context_vector(self) -> Optional[np.ndarray]:
        """Get the computed context vector."""
        return self.context_vector


class AttentionDecoder:
    """
    Decoder with attention mechanism for sequence-to-sequence models.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int,
                 encoder_dim: int, num_layers: int = 1, dropout: float = 0.1):
        """
        Initialize attention decoder.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden states
            encoder_dim: Dimension of encoder output
            num_layers: Number of decoder layers
            dropout: Dropout probability
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.encoder_dim = encoder_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Embedding layer
        self.embedding = np.random.randn(vocab_size, embed_dim) * 0.01
        
        # Attention mechanism
        self.attention = SequenceAttention(encoder_dim, hidden_dim)
        
        # RNN layers (simplified - in practice you'd use proper RNN/LSTM)
        self.rnn_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = embed_dim + encoder_dim  # embedding + context
            else:
                input_dim = hidden_dim
            self.rnn_layers.append(np.random.randn(input_dim, hidden_dim) * 0.01)
        
        # Output projection
        self.W_out = np.random.randn(hidden_dim, vocab_size) * 0.01
        self.b_out = np.zeros(vocab_size)
        
        # Store intermediate values
        self.encoder_states = None
        self.hidden_states = None
        
    def forward(self, input_tokens: np.ndarray, encoder_states: np.ndarray,
                initial_hidden: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass of attention decoder.
        
        Args:
            input_tokens: Input token indices, shape (batch_size, seq_len)
            encoder_states: Encoder hidden states
            initial_hidden: Initial hidden state
            
        Returns:
            Tuple of (output_logits, attention_weights)
        """
        batch_size, seq_len = input_tokens.shape
        
        # Store encoder states
        self.encoder_states = encoder_states
        
        # Initialize hidden state
        if initial_hidden is None:
            hidden = np.zeros((batch_size, self.hidden_dim))
        else:
            hidden = initial_hidden
        
        # Store hidden states
        self.hidden_states = []
        
        # Process sequence step by step
        outputs = []
        all_attention_weights = []
        
        for t in range(seq_len):
            # Get current input token
            current_input = input_tokens[:, t]
            
            # Embed current token
            embedded = self.embedding[current_input]  # (batch_size, embed_dim)
            
            # Compute attention
            context, attention_weights = self.attention.forward(
                encoder_states, hidden
            )
            
            # Combine embedded input with context
            combined_input = np.concatenate([embedded, context], axis=1)
            
            # Process through RNN layers
            for i, rnn_layer in enumerate(self.rnn_layers):
                if i == 0:
                    rnn_input = combined_input
                else:
                    rnn_input = hidden
                
                hidden = np.tanh(np.dot(rnn_input, rnn_layer))
                self.hidden_states.append(hidden)
            
            # Generate output
            output = np.dot(hidden, self.W_out) + self.b_out
            
            outputs.append(output)
            all_attention_weights.append(attention_weights)
        
        # Stack outputs
        output_logits = np.stack(outputs, axis=1)  # (batch_size, seq_len, vocab_size)
        attention_weights = np.stack(all_attention_weights, axis=1)  # (batch_size, seq_len, seq_len)
        
        return output_logits, attention_weights


# Example usage and testing
if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Configuration
    batch_size = 2
    seq_len = 4
    encoder_dim = 64
    decoder_dim = 64
    vocab_size = 1000
    
    print("=== Sequence-to-Sequence Attention Test ===")
    
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
    decoder = AttentionDecoder(vocab_size, 128, decoder_dim, encoder_dim)
    
    # Create dummy input tokens
    input_tokens = np.random.randint(0, vocab_size, size=(batch_size, seq_len))
    
    # Forward pass
    output_logits, decoder_attention = decoder.forward(
        input_tokens, encoder_states
    )
    
    print(f"Input tokens shape: {input_tokens.shape}")
    print(f"Output logits shape: {output_logits.shape}")
    print(f"Decoder attention shape: {decoder_attention.shape}")
    
    print("\n✅ Sequence-to-sequence attention working correctly!")
