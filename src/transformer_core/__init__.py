"""
Transformer Core Package

This package contains the core components of the Transformer architecture:
- PositionalEncoding: Adds positional information to embeddings
- FeedForward: Two-layer feed-forward network with ReLU activation
- LayerNormalization: Normalizes inputs across feature dimensions
- TransformerBlock: Complete Transformer block with attention and feed-forward

These components form the foundation for building complete Transformer models.
"""

from .positional_encoding import PositionalEncoding
from .feed_forward import FeedForward
from .layer_norm import LayerNormalization
from .transformer_block import TransformerBlock

__version__ = "1.0.0"
__author__ = "Transformer Workshop Team"

__all__ = [
    'PositionalEncoding',
    'FeedForward', 
    'LayerNormalization',
    'TransformerBlock'
]
