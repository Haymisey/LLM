"""
Vocabulary Management for Bible Translation
=========================================

Manages vocabulary for both Amharic (source) and Oromiffa (target) languages.
Handles vocabulary building, statistics, and management.
"""

import json
import pickle
from typing import Dict, List, Set, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np


class Vocabulary:
    """
    Vocabulary management for Amharic and Oromiffa Bible translation.
    
    Features:
    - Dual language vocabulary management
    - Frequency-based vocabulary building
    - Vocabulary statistics and analysis
    - Export/import functionality
    """
    
    def __init__(self, 
                 source_lang: str = 'amharic',
                 target_lang: str = 'oromiffa',
                 max_vocab_size: int = 10000,
                 min_freq: int = 2,
                 special_tokens: Optional[List[str]] = None):
        """
        Initialize vocabulary manager.
        
        Args:
            source_lang: Source language identifier
            target_lang: Target language identifier
            max_vocab_size: Maximum vocabulary size for each language
            min_freq: Minimum frequency for tokens to be included
            special_tokens: List of special tokens
        """
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        
        # Default special tokens
        self.special_tokens = special_tokens or [
            '<PAD>', '<UNK>', '<SOS>', '<EOS>', '<MASK>'
        ]
        
        # Initialize vocabularies
        self.source_vocab = {}
        self.target_vocab = {}
        self.source_reverse_vocab = {}
        self.target_reverse_vocab = {}
        
        # Token frequencies
        self.source_token_freqs = Counter()
        self.target_token_freqs = Counter()
        
        # Vocabulary statistics
        self.source_stats = {}
        self.target_stats = {}
        
        # Add special tokens
        self._add_special_tokens()
    
    def _add_special_tokens(self):
        """Add special tokens to both vocabularies."""
        for i, token in enumerate(self.special_tokens):
            # Source language
            self.source_vocab[token] = i
            self.source_reverse_vocab[i] = token
            
            # Target language
            self.target_vocab[token] = i
            self.target_reverse_vocab[i] = token
    
    def build_vocabulary(self, 
                        source_texts: List[str], 
                        target_texts: List[str],
                        tokenization_type: str = 'word'):
        """
        Build vocabulary from training texts.
        
        Args:
            source_texts: List of source language texts
            target_texts: List of target language texts
            tokenization_type: Type of tokenization ('word', 'char', 'subword')
        """
        print(f"Building vocabulary for {self.source_lang} -> {self.target_lang}")
        
        # Build source vocabulary
        self._build_language_vocabulary(
            source_texts, 
            self.source_vocab, 
            self.source_reverse_vocab, 
            self.source_token_freqs,
            'source'
        )
        
        # Build target vocabulary
        self._build_language_vocabulary(
            target_texts, 
            self.target_vocab, 
            self.target_reverse_vocab, 
            self.target_token_freqs,
            'target'
        )
        
        # Calculate statistics
        self._calculate_statistics()
        
        print(f"Source vocabulary size: {len(self.source_vocab)}")
        print(f"Target vocabulary size: {len(self.target_vocab)}")
    
    def _build_language_vocabulary(self, 
                                  texts: List[str], 
                                  vocab: Dict[str, int], 
                                  reverse_vocab: Dict[int, str],
                                  token_freqs: Counter,
                                  lang_type: str):
        """Build vocabulary for a specific language."""
        print(f"Building {lang_type} vocabulary...")
        
        # Collect all tokens
        all_tokens = []
        for text in texts:
            if not text or not isinstance(text, str):
                continue
            
            # Simple tokenization based on type
            if lang_type == 'source' and self.source_lang == 'amharic':
                # Amharic-specific tokenization
                tokens = self._tokenize_amharic(text)
            elif lang_type == 'target' and self.target_lang == 'oromiffa':
                # Oromiffa-specific tokenization
                tokens = self._tokenize_oromiffa(text)
            else:
                # General tokenization
                tokens = self._tokenize_general(text)
            
            all_tokens.extend(tokens)
        
        # Count frequencies
        token_freqs.update(all_tokens)
        
        # Filter by minimum frequency
        filtered_tokens = [(token, freq) for token, freq in token_freqs.items() 
                          if freq >= self.min_freq and token not in self.special_tokens]
        
        # Sort by frequency (descending)
        filtered_tokens.sort(key=lambda x: x[1], reverse=True)
        
        # Add to vocabulary (respecting max_vocab_size limit)
        vocab_start = len(self.special_tokens)
        for i, (token, freq) in enumerate(filtered_tokens):
            if vocab_start + i >= self.max_vocab_size:
                break
            token_id = vocab_start + i
            vocab[token] = token_id
            reverse_vocab[token_id] = token
        
        print(f"{lang_type.capitalize()} vocabulary built with {len(vocab)} tokens")
    
    def _tokenize_amharic(self, text: str) -> List[str]:
        """Tokenize Amharic text."""
        # Simple word splitting for Amharic
        import re
        words = re.findall(r'[\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F]+', text)
        return [word for word in words if word.strip()]
    
    def _tokenize_oromiffa(self, text: str) -> List[str]:
        """Tokenize Oromiffa text."""
        # Simple word splitting for Oromiffa (Latin script)
        import re
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        return [word for word in words if word.strip()]
    
    def _tokenize_general(self, text: str) -> List[str]:
        """General tokenization."""
        import re
        words = re.findall(r'\b\w+\b', text)
        return [word for word in words if word.strip()]
    
    def _calculate_statistics(self):
        """Calculate vocabulary statistics."""
        # Source language statistics
        self.source_stats = {
            'total_tokens': len(self.source_vocab),
            'special_tokens': len(self.special_tokens),
            'regular_tokens': len(self.source_vocab) - len(self.special_tokens),
            'most_common': self.source_token_freqs.most_common(10),
            'avg_freq': np.mean(list(self.source_token_freqs.values())) if self.source_token_freqs else 0,
            'median_freq': np.median(list(self.source_token_freqs.values())) if self.source_token_freqs else 0
        }
        
        # Target language statistics
        self.target_stats = {
            'total_tokens': len(self.target_vocab),
            'special_tokens': len(self.special_tokens),
            'regular_tokens': len(self.target_vocab) - len(self.special_tokens),
            'most_common': self.target_token_freqs.most_common(10),
            'avg_freq': np.mean(list(self.target_token_freqs.values())) if self.target_token_freqs else 0,
            'median_freq': np.median(list(self.target_token_freqs.values())) if self.target_token_freqs else 0
        }
    
    def get_vocab_size(self, language: str = 'source') -> int:
        """Get vocabulary size for specified language."""
        if language == 'source':
            return len(self.source_vocab)
        elif language == 'target':
            return len(self.target_vocab)
        else:
            raise ValueError(f"Unknown language: {language}")
    
    def get_token_id(self, token: str, language: str = 'source') -> int:
        """Get token ID for a given token and language."""
        if language == 'source':
            return self.source_vocab.get(token, self.source_vocab['<UNK>'])
        elif language == 'target':
            return self.target_vocab.get(token, self.target_vocab['<UNK>'])
        else:
            raise ValueError(f"Unknown language: {language}")
    
    def get_token(self, token_id: int, language: str = 'source') -> str:
        """Get token for a given token ID and language."""
        if language == 'source':
            return self.source_reverse_vocab.get(token_id, '<UNK>')
        elif language == 'target':
            return self.target_reverse_vocab.get(token_id, '<UNK>')
        else:
            raise ValueError(f"Unknown language: {language}")
    
    def get_token_frequency(self, token: str, language: str = 'source') -> int:
        """Get frequency of a token in the specified language."""
        if language == 'source':
            return self.source_token_freqs.get(token, 0)
        elif language == 'target':
            return self.target_token_freqs.get(token, 0)
        else:
            raise ValueError(f"Unknown language: {language}")
    
    def get_statistics(self, language: str = 'source') -> dict:
        """Get vocabulary statistics for the specified language."""
        if language == 'source':
            return self.source_stats.copy()
        elif language == 'target':
            return self.target_stats.copy()
        else:
            raise ValueError(f"Unknown language: {language}")
    
    def get_coverage_analysis(self, texts: List[str], language: str = 'source') -> dict:
        """
        Analyze vocabulary coverage for given texts.
        
        Args:
            texts: List of texts to analyze
            language: Language to analyze
            
        Returns:
            Dictionary with coverage statistics
        """
        if language == 'source':
            vocab = self.source_vocab
            token_freqs = self.source_token_freqs
        elif language == 'target':
            vocab = self.target_vocab
            token_freqs = self.target_token_freqs
        else:
            raise ValueError(f"Unknown language: {language}")
        
        total_tokens = 0
        covered_tokens = 0
        unknown_tokens = set()
        
        for text in texts:
            if not text or not isinstance(text, str):
                continue
            
            # Tokenize text
            if language == 'source' and self.source_lang == 'amharic':
                tokens = self._tokenize_amharic(text)
            elif language == 'target' and self.target_lang == 'oromiffa':
                tokens = self._tokenize_oromiffa(text)
            else:
                tokens = self._tokenize_general(text)
            
            for token in tokens:
                total_tokens += 1
                if token in vocab:
                    covered_tokens += 1
                else:
                    unknown_tokens.add(token)
        
        coverage_rate = covered_tokens / total_tokens if total_tokens > 0 else 0
        
        return {
            'total_tokens': total_tokens,
            'covered_tokens': covered_tokens,
            'unknown_tokens': len(unknown_tokens),
            'coverage_rate': coverage_rate,
            'unknown_token_list': list(unknown_tokens)[:20]  # First 20 unknown tokens
        }
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to a file."""
        vocab_data = {
            'source_lang': self.source_lang,
            'target_lang': self.target_lang,
            'max_vocab_size': self.max_vocab_size,
            'min_freq': self.min_freq,
            'special_tokens': self.special_tokens,
            'source_vocab': self.source_vocab,
            'target_vocab': self.target_vocab,
            'source_reverse_vocab': self.source_reverse_vocab,
            'target_reverse_vocab': self.target_reverse_vocab,
            'source_token_freqs': dict(self.source_token_freqs),
            'target_token_freqs': dict(self.target_token_freqs),
            'source_stats': self.source_stats,
            'target_stats': self.target_stats
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str):
        """Load vocabulary from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Restore all attributes
        self.source_lang = vocab_data['source_lang']
        self.target_lang = vocab_data['target_lang']
        self.max_vocab_size = vocab_data['max_vocab_size']
        self.min_freq = vocab_data['min_freq']
        self.special_tokens = vocab_data['special_tokens']
        self.source_vocab = vocab_data['source_vocab']
        self.target_vocab = vocab_data['target_vocab']
        self.source_reverse_vocab = vocab_data['source_reverse_vocab']
        self.target_reverse_vocab = vocab_data['target_reverse_vocab']
        self.source_token_freqs = Counter(vocab_data['source_token_freqs'])
        self.target_token_freqs = Counter(vocab_data['target_token_freqs'])
        self.source_stats = vocab_data['source_stats']
        self.target_stats = vocab_data['target_stats']
        
        print(f"Vocabulary loaded from {filepath}")


def test_vocabulary():
    """Test the Vocabulary class."""
    print("Testing Vocabulary...")
    
    # Sample texts
    amharic_texts = [
        "የሰላም እለት ነው።",
        "እግዚአብሔር ይመስገን።",
        "የሰማይ ንጉሥ ነው።",
        "የምድር ገንዘብ ነው።"
    ]
    
    oromiffa_texts = [
        "Baga nagaan dhuftan!",
        "Waaqayoo galata isaaniif.",
        "Waaqayoo qabeenya isaaniif.",
        "Waaqayoo barumsa isaaniif."
    ]
    
    # Create vocabulary
    vocab = Vocabulary(
        source_lang='amharic',
        target_lang='oromiffa',
        max_vocab_size=100,
        min_freq=1
    )
    
    # Build vocabulary
    vocab.build_vocabulary(amharic_texts, oromiffa_texts)
    
    # Test vocabulary sizes
    print(f"Source vocab size: {vocab.get_vocab_size('source')}")
    print(f"Target vocab size: {vocab.get_vocab_size('target')}")
    
    # Test token ID retrieval
    test_token = "የሰላም"
    token_id = vocab.get_token_id(test_token, 'source')
    retrieved_token = vocab.get_token(token_id, 'source')
    print(f"Token: {test_token} -> ID: {token_id} -> Retrieved: {retrieved_token}")
    
    # Test statistics
    source_stats = vocab.get_statistics('source')
    target_stats = vocab.get_statistics('target')
    print(f"Source stats: {source_stats}")
    print(f"Target stats: {target_stats}")
    
    # Test coverage analysis
    coverage = vocab.get_coverage_analysis(amharic_texts, 'source')
    print(f"Coverage analysis: {coverage}")
    
    print("✓ Vocabulary tests passed!")


if __name__ == "__main__":
    test_vocabulary()
