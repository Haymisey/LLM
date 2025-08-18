"""
Data Preparation Module for Bible Translation
============================================

This module provides all the necessary components for preparing and processing
parallel text data for training the Transformer model.

Components:
- TextPreprocessor: Text cleaning and normalization
- Tokenizer: Text tokenization (character, word, subword)
- Vocabulary: Dual-language vocabulary management
- DataLoader: Batch generation and sequence padding
- DataAugmentation: Data augmentation techniques
- AmharicOromiffaDataset: Real dataset handler for parallel text
"""

from .text_preprocessor import TextPreprocessor
from .tokenizer import Tokenizer
from .vocabulary import Vocabulary
from .data_loader import DataLoader
from .augmentation import DataAugmentation
from .amharic_oromiffa_dataset import AmharicOromiffaDataset, create_progressive_dataset

__all__ = [
    'TextPreprocessor',
    'Tokenizer', 
    'Vocabulary',
    'DataLoader',
    'DataAugmentation',
    'AmharicOromiffaDataset',
    'create_progressive_dataset'
]
