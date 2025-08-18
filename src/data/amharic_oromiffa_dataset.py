"""
Amharic-Oromiffa Parallel Dataset Handler
=========================================

Handles the real dataset of 208,906 parallel sentences for training the Bible translation model.
Supports progressive training (starting with subset, gradually increasing data).
"""

import os
import json
import numpy as np
from typing import List, Tuple, Iterator, Optional, Dict, Any
from .text_preprocessor import TextPreprocessor
from .tokenizer import Tokenizer
from .vocabulary import Vocabulary


class AmharicOromiffaDataset:
    """
    Dataset handler for Amharic-Oromiffa parallel text.
    
    Features:
    - Loads tab-separated parallel text file
    - Progressive training support (start with subset)
    - Real text preprocessing and tokenization
    - Vocabulary building from actual data
    - Train/validation split
    """
    
    def __init__(self, 
                 data_file_path: str,
                 max_samples: Optional[int] = None,
                 train_split: float = 0.9,
                 max_source_length: int = 100,
                 max_target_length: int = 100,
                 min_freq: int = 2,
                 initial_train_size: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            data_file_path: Path to tab-separated parallel text file
            max_samples: Maximum number of samples to load (for progressive training)
            train_split: Fraction of data to use for training
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
            min_freq: Minimum token frequency for vocabulary
        """
        self.data_file_path = data_file_path
        # Note: max_samples is not used to truncate the loaded dataset anymore.
        # We always load the full dataset to build stable vocabularies and maintain
        # a fixed validation set. Progressive training is handled via initial_train_size
        # and increase_training_data().
        self.max_samples = max_samples
        self.train_split = train_split
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.min_freq = min_freq
        self.initial_train_size = initial_train_size
        
        # Initialize components
        self.text_preprocessor = TextPreprocessor()
        self.src_tokenizer = Tokenizer(tokenization_type='word', vocab_size=10000, min_freq=min_freq)
        self.tgt_tokenizer = Tokenizer(tokenization_type='word', vocab_size=10000, min_freq=min_freq)
        
        # Data storage
        self.raw_data = []
        self.processed_data = []
        self.train_data = []
        self.val_data = []
        self._remaining_train_pool = []  # pool of samples available to grow training set progressively
        
        # Load and process data
        self._load_data()
        self._preprocess_data()
        self._build_vocabularies()
        self._split_data()
        
        print(f"Dataset loaded: {len(self.processed_data)} total samples")
        print(f"Training samples: {len(self.train_data)}")
        print(f"Validation samples: {len(self.val_data)}")
        print(f"Source vocabulary size: {len(self.src_tokenizer.vocab)}")
        print(f"Target vocabulary size: {len(self.tgt_tokenizer.vocab)}")
    
    def _load_data(self):
        """Load ALL parallel text data from file (no truncation)."""
        print(f"Loading data from {self.data_file_path}...")
        
        if not os.path.exists(self.data_file_path):
            raise FileNotFoundError(f"Data file not found: {self.data_file_path}")
        
        with open(self.data_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse tab-separated lines
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Split by tab character
            parts = line.split('\t')
            if len(parts) != 2:
                print(f"Warning: Line {line_num + 1} has {len(parts)} parts, skipping: {line[:50]}...")
                continue
            
            amharic_text, oromiffa_text = parts[0].strip(), parts[1].strip()
            
            # Basic validation
            if len(amharic_text) < 2 or len(oromiffa_text) < 2:
                continue  # Skip very short texts
            
            self.raw_data.append((amharic_text, oromiffa_text))
        
        print(f"Loaded {len(self.raw_data)} parallel sentences")
    
    def _preprocess_data(self):
        """Preprocess all texts using TextPreprocessor."""
        print("Preprocessing texts...")
        
        for amharic, oromiffa in self.raw_data:
            # Preprocess both languages
            processed_amharic = self.text_preprocessor.preprocess(amharic, 'amharic')
            processed_oromiffa = self.text_preprocessor.preprocess(oromiffa, 'oromiffa')
            
            # Skip if preprocessing failed
            if not processed_amharic or not processed_oromiffa:
                continue
            
            self.processed_data.append((processed_amharic, processed_oromiffa))
        
        print(f"Preprocessed {len(self.processed_data)} sentences")
    
    def _build_vocabularies(self):
        """Build vocabularies from processed data."""
        print("Building vocabularies...")
        
        # Collect all source and target texts
        src_texts = [pair[0] for pair in self.processed_data]
        tgt_texts = [pair[1] for pair in self.processed_data]
        
        # Build vocabularies
        self.src_tokenizer.fit(src_texts, 'amharic')
        self.tgt_tokenizer.fit(tgt_texts, 'oromiffa')
        
        print(f"Source vocabulary: {len(self.src_tokenizer.vocab)} tokens")
        print(f"Target vocabulary: {len(self.tgt_tokenizer.vocab)} tokens")
    
    def _split_data(self):
        """Split data into training and validation sets with a fixed validation holdout."""
        total_samples = len(self.processed_data)
        if total_samples == 0:
            self.train_data = []
            self.val_data = []
            self._remaining_train_pool = []
            return

        # Determine validation size: at least 1000, or 10% of total, whichever is larger
        min_val_size = max(1000, int(total_samples * 0.1))
        min_val_size = min(min_val_size, total_samples)  # cannot exceed total

        # Shuffle indices
        random_indices = np.random.permutation(total_samples)

        # Reserve validation set
        val_indices = random_indices[:min_val_size]
        train_candidates_indices = random_indices[min_val_size:]

        all_train_candidates = [self.processed_data[i] for i in train_candidates_indices]
        self.val_data = [self.processed_data[i] for i in val_indices]

        # Determine initial training size
        if self.initial_train_size is not None:
            init_size = min(self.initial_train_size, len(all_train_candidates))
            self.train_data = all_train_candidates[:init_size]
            self._remaining_train_pool = all_train_candidates[init_size:]
        else:
            # Fallback to ratio-based split if no explicit size is provided
            split_idx = int(len(all_train_candidates) * self.train_split)
            self.train_data = all_train_candidates[:split_idx]
            self._remaining_train_pool = all_train_candidates[split_idx:]
    
    def get_training_data(self) -> Tuple[List[str], List[str]]:
        """Get training data as separate source and target lists."""
        src_texts = [pair[0] for pair in self.train_data]
        tgt_texts = [pair[1] for pair in self.train_data]
        return src_texts, tgt_texts
    
    def get_validation_data(self) -> Tuple[List[str], List[str]]:
        """Get validation data as separate source and target lists."""
        src_texts = [pair[0] for pair in self.val_data]
        tgt_texts = [pair[1] for pair in self.val_data]
        return src_texts, tgt_texts
    
    def get_tokenizers(self) -> Tuple[Tokenizer, Tokenizer]:
        """Get source and target tokenizers."""
        return self.src_tokenizer, self.tgt_tokenizer
    
    def get_vocab_sizes(self) -> Tuple[int, int]:
        """Get source and target vocabulary sizes."""
        return len(self.src_tokenizer.vocab), len(self.tgt_tokenizer.vocab)
    
    def increase_training_data(self, additional_samples: int):
        """
        Increase training data size for progressive training by moving samples
        from the remaining training pool into the active training set, while
        keeping the validation set fixed.
        
        Args:
            additional_samples: Number of additional samples to add
        """
        if additional_samples <= 0:
            print("Warning: additional_samples must be positive")
            return 0

        if not self._remaining_train_pool:
            print("Warning: No more samples available to add to training")
            return 0

        samples_to_move = min(additional_samples, len(self._remaining_train_pool))
        samples_to_move_list = self._remaining_train_pool[:samples_to_move]
        self._remaining_train_pool = self._remaining_train_pool[samples_to_move:]
        self.train_data.extend(samples_to_move_list)

        print(f"Moved {samples_to_move} samples to training")
        print(f"New training size: {len(self.train_data)}")
        print(f"Validation size remains fixed: {len(self.val_data)}")
        print(f"Remaining pool size: {len(self._remaining_train_pool)}")

        return samples_to_move
    
    def save_vocabularies(self, save_dir: str):
        """Save vocabularies to disk."""
        os.makedirs(save_dir, exist_ok=True)
        
        src_vocab_path = os.path.join(save_dir, "src_vocabulary.json")
        tgt_vocab_path = os.path.join(save_dir, "tgt_vocabulary.json")
        
        self.src_tokenizer.save_vocab(src_vocab_path)
        self.tgt_tokenizer.save_vocab(tgt_vocab_path)
        
        print(f"Vocabularies saved to {save_dir}")
    
    def load_vocabularies(self, save_dir: str):
        """Load vocabularies from disk."""
        src_vocab_path = os.path.join(save_dir, "src_vocabulary.json")
        tgt_vocab_path = os.path.join(save_dir, "tgt_vocabulary.json")
        
        if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
            self.src_tokenizer.load_vocab(src_vocab_path)
            self.tgt_tokenizer.load_vocab(tgt_vocab_path)
            print(f"Vocabularies loaded from {save_dir}")
        else:
            print(f"Vocabulary files not found in {save_dir}")


def create_progressive_dataset(data_file_path: str, 
                             initial_samples: int = 10000,
                             max_samples: int = 208906) -> AmharicOromiffaDataset:
    """
    Create a dataset for progressive training.
    
    Args:
        data_file_path: Path to parallel text file
        initial_samples: Number of samples to start with
        max_samples: Maximum samples to use eventually
    
    Returns:
        Dataset configured for progressive training
    """
    # Load full dataset for stable vocab and fixed validation; start training
    # with 'initial_samples' and grow progressively.
    return AmharicOromiffaDataset(
        data_file_path=data_file_path,
        max_samples=None,
        train_split=0.9,
        max_source_length=100,
        max_target_length=100,
        min_freq=2,
        initial_train_size=initial_samples
    )
