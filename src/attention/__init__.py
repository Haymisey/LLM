"""
Attention Mechanism Package

This package implements attention mechanisms from scratch for educational purposes.
It includes all the core components needed to build attention-based models.

Components:
- BasicAttention: Simple query-key-value attention
- SelfAttention: Self-attention for sequences
- MultiHeadAttention: Multi-head attention mechanism
- AttentionVisualizer: Visualization tools for attention weights
- SequenceAttention: Sequence-to-sequence attention

Example:
    from attention import BasicAttention, SelfAttention, MultiHeadAttention
    
    # Create attention mechanisms
    basic_attn = BasicAttention(query_dim=64, key_dim=64, value_dim=64)
    self_attn = SelfAttention(embed_dim=64, num_heads=8)
    multi_head = MultiHeadAttention(embed_dim=64, num_heads=8)
"""

from .basic_attention import BasicAttention
from .self_attention import SelfAttention
from .multi_head_attention import MultiHeadAttention
from .attention_visualizer import AttentionVisualizer
from .sequence_attention import SequenceAttention

__version__ = "1.0.0"
__author__ = "Transformer Workshop Team"

__all__ = [
    'BasicAttention',
    'SelfAttention', 
    'MultiHeadAttention',
    'AttentionVisualizer',
    'SequenceAttention'
]
