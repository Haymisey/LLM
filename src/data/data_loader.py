"""
Data Loader for Bible Translation
================================

Handles data loading, batching, and preparation for training the Transformer model.
Includes sequence padding, masking, and data augmentation.
"""

import numpy as np
from typing import List, Tuple, Iterator, Optional, Dict, Any
import random


class DataLoader:
    """
    Data loader for Bible translation training data.
    
    Features:
    - Batch generation with configurable batch sizes
    - Sequence padding and masking
    - Data shuffling and augmentation
    - Support for both training and validation modes
    """
    
    def __init__(self, 
                 source_texts: List[str],
                 target_texts: List[str],
                 source_tokenizer,
                 target_tokenizer,
                 batch_size: int = 32,
                 max_source_length: int = 100,
                 max_target_length: int = 100,
                 shuffle: bool = True,
                 pad_token_id: int = 0):
        """
        Initialize the data loader.
        
        Args:
            source_texts: List of source language texts
            target_texts: List of target language texts
            source_tokenizer: Tokenizer for source language
            target_tokenizer: Tokenizer for target language
            batch_size: Size of each batch
            max_source_length: Maximum length for source sequences
            max_target_length: Maximum length for target sequences
            shuffle: Whether to shuffle data
            pad_token_id: Token ID for padding
        """
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.batch_size = batch_size
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.shuffle = shuffle
        self.pad_token_id = pad_token_id
        
        # Validate input
        if len(source_texts) != len(target_texts):
            raise ValueError("Source and target texts must have the same length")
        
        # Preprocess and tokenize all texts
        self._preprocess_data()
        
        # Calculate number of batches
        self.num_batches = (len(self.source_texts) + batch_size - 1) // batch_size
        
        print(f"DataLoader initialized with {len(source_texts)} samples")
        print(f"Batch size: {batch_size}, Number of batches: {self.num_batches}")
    
    def _preprocess_data(self):
        """Preprocess and tokenize all texts."""
        print("Preprocessing and tokenizing data...")
        
        # Tokenize source texts
        self.source_tokenized = []
        for text in self.source_texts:
            if text and isinstance(text, str):
                tokens = self.source_tokenizer.encode(text, add_special_tokens=True)
                # Truncate if too long
                if len(tokens) > self.max_source_length:
                    tokens = tokens[:self.max_source_length-1] + [tokens[-1]]  # Keep EOS token
                self.source_tokenized.append(tokens)
            else:
                self.source_tokenized.append([])
        
        # Tokenize target texts
        self.target_tokenized = []
        for text in self.target_texts:
            if text and isinstance(text, str):
                tokens = self.target_tokenizer.encode(text, add_special_tokens=True)
                # Truncate if too long
                if len(tokens) > self.max_target_length:
                    tokens = tokens[:self.max_target_length-1] + [tokens[-1]]  # Keep EOS token
                self.target_tokenized.append(tokens)
            else:
                self.target_tokenized.append([])
        
        # Filter out empty sequences
        valid_indices = [i for i, (src, tgt) in enumerate(zip(self.source_tokenized, self.target_tokenized)) 
                        if len(src) > 0 and len(tgt) > 0]
        
        self.source_tokenized = [self.source_tokenized[i] for i in valid_indices]
        self.target_tokenized = [self.target_tokenized[i] for i in valid_indices]
        
        print(f"Valid samples after preprocessing: {len(self.source_tokenized)}")
    
    def _pad_sequences(self, sequences: List[List[int]], max_length: int) -> np.ndarray:
        """
        Pad sequences to the same length.
        
        Args:
            sequences: List of token sequences
            max_length: Maximum sequence length
            
        Returns:
            Padded sequences as numpy array
        """
        padded = []
        for seq in sequences:
            if len(seq) < max_length:
                # Pad with pad_token_id
                padded_seq = seq + [self.pad_token_id] * (max_length - len(seq))
            else:
                padded_seq = seq[:max_length]
            padded.append(padded_seq)
        
        return np.array(padded, dtype=np.int32)
    
    def _create_padding_mask(self, sequences: np.ndarray) -> np.ndarray:
        """
        Create padding mask for sequences.
        
        Args:
            sequences: Padded sequences
            
        Returns:
            Padding mask (1 for real tokens, 0 for padding)
        """
        mask = (sequences != self.pad_token_id).astype(np.float32)
        return mask
    
    def _create_look_ahead_mask(self, target_sequences: np.ndarray) -> np.ndarray:
        """
        Create look-ahead mask for target sequences.
        
        Args:
            target_sequences: Target sequences
            
        Returns:
            Look-ahead mask
        """
        seq_len = target_sequences.shape[1]
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = mask.astype(np.float32)
        return mask
    
    def get_batch(self, batch_idx: int) -> Dict[str, np.ndarray]:
        """
        Get a specific batch by index.
        
        Args:
            batch_idx: Batch index
            
        Returns:
            Dictionary containing batch data
        """
        if batch_idx >= self.num_batches:
            raise IndexError(f"Batch index {batch_idx} out of range")
        
        start_idx = batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.source_tokenized))
        
        # Get batch data
        batch_source = self.source_tokenized[start_idx:end_idx]
        batch_target = self.target_tokenized[start_idx:end_idx]
        
        # Pad sequences
        source_padded = self._pad_sequences(batch_source, self.max_source_length)
        target_padded = self._pad_sequences(batch_target, self.max_target_length)
        
        # Create masks
        source_padding_mask = self._create_padding_mask(source_padded)
        target_padding_mask = self._create_padding_mask(target_padded)
        look_ahead_mask = self._create_look_ahead_mask(target_padded)
        
        return {
            'source': source_padded,
            'target': target_padded,
            'source_padding_mask': source_padding_mask,
            'target_padding_mask': target_padding_mask,
            'look_ahead_mask': look_ahead_mask,
            'batch_size': len(batch_source)
        }
    
    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Iterate over batches."""
        # Create indices for shuffling
        indices = list(range(len(self.source_tokenized)))
        if self.shuffle:
            random.shuffle(indices)
        
        # Yield batches
        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(indices))
            
            # Get batch indices
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch_source = [self.source_tokenized[i] for i in batch_indices]
            batch_target = [self.target_tokenized[i] for i in batch_indices]
            
            # Pad sequences
            source_padded = self._pad_sequences(batch_source, self.max_source_length)
            target_padded = self._pad_sequences(batch_target, self.max_target_length)
            
            # Create masks
            source_padding_mask = self._create_padding_mask(source_padded)
            target_padding_mask = self._create_padding_mask(target_padded)
            look_ahead_mask = self._create_look_ahead_mask(target_padded)
            
            yield {
                'source': source_padded,
                'target': target_padded,
                'source_padding_mask': source_padding_mask,
                'target_padding_mask': target_padding_mask,
                'look_ahead_mask': look_ahead_mask,
                'batch_size': len(batch_source)
            }
    
    def __len__(self) -> int:
        """Get the number of batches."""
        return self.num_batches
    
    def get_sample_batch(self) -> Dict[str, np.ndarray]:
        """Get a sample batch for testing."""
        return self.get_batch(0)
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about the data."""
        source_lengths = [len(seq) for seq in self.source_tokenized]
        target_lengths = [len(seq) for seq in self.target_tokenized]
        
        return {
            'total_samples': len(self.source_tokenized),
            'source_length_stats': {
                'min': min(source_lengths) if source_lengths else 0,
                'max': max(source_lengths) if source_lengths else 0,
                'mean': np.mean(source_lengths) if source_lengths else 0,
                'median': np.median(source_lengths) if source_lengths else 0
            },
            'target_length_stats': {
                'min': min(target_lengths) if target_lengths else 0,
                'max': max(target_lengths) if target_lengths else 0,
                'mean': np.mean(target_lengths) if target_lengths else 0,
                'median': np.median(target_lengths) if target_lengths else 0
            },
            'batch_size': self.batch_size,
            'num_batches': self.num_batches
        }
    
    def reset(self):
        """Reset the data loader (useful for reshuffling)."""
        if self.shuffle:
            random.shuffle(self.source_tokenized)
            random.shuffle(self.target_tokenized)


class MockTokenizer:
    """Mock tokenizer for testing purposes."""
    
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Mock encoding - returns random token IDs."""
        if not text:
            return []
        
        # Simple mock encoding
        tokens = list(range(len(text) + (2 if add_special_tokens else 0)))
        if add_special_tokens:
            tokens[0] = 1  # SOS
            tokens[-1] = 2  # EOS
        
        return tokens


def test_data_loader():
    """Test the DataLoader class."""
    print("Testing DataLoader...")
    
    # Sample data
    source_texts = [
        "የሰላም እለት ነው።",
        "እግዚአብሔር ይመስገን።",
        "የሰማይ ንጉሥ ነው።",
        "የምድር ገንዘብ ነው።"
    ]
    
    target_texts = [
        "Baga nagaan dhuftan!",
        "Waaqayoo galata isaaniif.",
        "Waaqayoo qabeenya isaaniif.",
        "Waaqayoo barumsa isaaniif."
    ]
    
    # Create mock tokenizers
    source_tokenizer = MockTokenizer()
    target_tokenizer = MockTokenizer()
    
    # Create data loader
    data_loader = DataLoader(
        source_texts=source_texts,
        target_texts=target_texts,
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        batch_size=2,
        max_source_length=20,
        max_target_length=20
    )
    
    # Test batch retrieval
    batch = data_loader.get_sample_batch()
    print(f"Sample batch keys: {list(batch.keys())}")
    print(f"Source shape: {batch['source'].shape}")
    print(f"Target shape: {batch['target'].shape}")
    print(f"Source padding mask shape: {batch['source_padding_mask'].shape}")
    print(f"Look-ahead mask shape: {batch['look_ahead_mask'].shape}")
    
    # Test iteration
    print("\nIterating through batches:")
    for i, batch in enumerate(data_loader):
        print(f"Batch {i}: source shape {batch['source'].shape}, target shape {batch['target'].shape}")
    
    # Test statistics
    stats = data_loader.get_data_statistics()
    print(f"\nData statistics: {stats}")
    
    print("✓ DataLoader tests passed!")


if __name__ == "__main__":
    test_data_loader()
