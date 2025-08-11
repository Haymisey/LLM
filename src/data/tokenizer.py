"""
Tokenizer for Bible Translation
==============================

Handles tokenization of Amharic and Oromiffa text for the Transformer model.
Supports both character-level and word-level tokenization.
"""

import re
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter


class Tokenizer:
    """
    Tokenizer for Amharic and Oromiffa Bible text.
    
    Supports:
    - Character-level tokenization
    - Word-level tokenization
    - Subword tokenization (basic implementation)
    - Special token handling
    """
    
    def __init__(self, 
                 tokenization_type: str = 'word',
                 vocab_size: int = 10000,
                 min_freq: int = 2,
                 special_tokens: Optional[List[str]] = None):
        """
        Initialize the tokenizer.
        
        Args:
            tokenization_type: 'char', 'word', or 'subword'
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for a token to be included
            special_tokens: List of special tokens to add
        """
        self.tokenization_type = tokenization_type
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.special_tokens = special_tokens or ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        
        # Initialize vocabulary
        self.vocab = {}
        self.reverse_vocab = {}
        self.token_freqs = Counter()
        
        # Add special tokens
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
            self.reverse_vocab[i] = token
        
        # Special token indices
        self.pad_token_id = self.vocab['<PAD>']
        self.unk_token_id = self.vocab['<UNK>']
        self.sos_token_id = self.vocab['<SOS>']
        self.eos_token_id = self.vocab['<EOS>']
        
        # Tokenization patterns
        self.word_pattern = re.compile(r'\b\w+\b')
        self.char_pattern = re.compile(r'.')
        
    def fit(self, texts: List[str], language: str = 'amharic'):
        """
        Build vocabulary from training texts.
        
        Args:
            texts: List of training texts
            language: Language identifier
        """
        print(f"Building vocabulary for {language} texts...")
        
        # Collect all tokens
        all_tokens = []
        for text in texts:
            if self.tokenization_type == 'char':
                tokens = self._tokenize_char(text)
            elif self.tokenization_type == 'word':
                tokens = self._tokenize_word(text)
            elif self.tokenization_type == 'subword':
                tokens = self._tokenize_subword(text)
            else:
                raise ValueError(f"Unknown tokenization type: {self.tokenization_type}")
            
            all_tokens.extend(tokens)
        
        # Count frequencies
        self.token_freqs = Counter(all_tokens)
        
        # Filter by minimum frequency
        filtered_tokens = [(token, freq) for token, freq in self.token_freqs.items() 
                          if freq >= self.min_freq]
        
        # Sort by frequency (descending)
        filtered_tokens.sort(key=lambda x: x[1], reverse=True)
        
        # Add to vocabulary (respecting vocab_size limit)
        vocab_start = len(self.special_tokens)
        for i, (token, freq) in enumerate(filtered_tokens):
            if vocab_start + i >= self.vocab_size:
                break
            token_id = vocab_start + i
            self.vocab[token] = token_id
            self.reverse_vocab[token_id] = token
        
        print(f"Vocabulary built with {len(self.vocab)} tokens")
        print(f"Most common tokens: {filtered_tokens[:10]}")
    
    def _tokenize_char(self, text: str) -> List[str]:
        """Character-level tokenization."""
        return list(text)
    
    def _tokenize_word(self, text: str) -> List[str]:
        """Word-level tokenization."""
        # Simple word splitting - can be enhanced for Amharic/Oromiffa
        words = re.findall(r'\b\w+\b', text)
        return [word for word in words if word.strip()]
    
    def _tokenize_subword(self, text: str) -> List[str]:
        """Basic subword tokenization."""
        # Simple implementation - can be enhanced with BPE or SentencePiece
        words = self._tokenize_word(text)
        subwords = []
        
        for word in words:
            if len(word) <= 3:
                subwords.append(word)
            else:
                # Split long words into subwords
                for i in range(0, len(word), 3):
                    subwords.append(word[i:i+3])
        
        return subwords
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Convert text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add SOS/EOS tokens
            
        Returns:
            List of token IDs
        """
        # Tokenize the text
        if self.tokenization_type == 'char':
            tokens = self._tokenize_char(text)
        elif self.tokenization_type == 'word':
            tokens = self._tokenize_word(text)
        elif self.tokenization_type == 'subword':
            tokens = self._tokenize_subword(text)
        else:
            raise ValueError(f"Unknown tokenization type: {self.tokenization_type}")
        
        # Convert tokens to IDs
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.sos_token_id)
        
        for token in tokens:
            token_id = self.vocab.get(token, self.unk_token_id)
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        return token_ids
    
    def decode(self, token_ids: List[int], remove_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            remove_special_tokens: Whether to remove special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            token = self.reverse_vocab.get(token_id, '<UNK>')
            if remove_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        
        if self.tokenization_type == 'char':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def batch_decode(self, token_id_lists: List[List[int]], remove_special_tokens: bool = True) -> List[str]:
        """Decode a batch of token ID lists."""
        return [self.decode(token_ids, remove_special_tokens) for token_ids in token_id_lists]
    
    def get_vocab_size(self) -> int:
        """Get the current vocabulary size."""
        return len(self.vocab)
    
    def get_token_frequency(self, token: str) -> int:
        """Get the frequency of a specific token."""
        return self.token_freqs.get(token, 0)
    
    def save_vocab(self, filepath: str):
        """Save vocabulary to a file."""
        import json
        vocab_data = {
            'vocab': self.vocab,
            'reverse_vocab': self.reverse_vocab,
            'token_freqs': dict(self.token_freqs),
            'tokenization_type': self.tokenization_type,
            'special_tokens': self.special_tokens
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocab(self, filepath: str):
        """Load vocabulary from a file."""
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.vocab = vocab_data['vocab']
        self.reverse_vocab = vocab_data['reverse_vocab']
        self.token_freqs = Counter(vocab_data['token_freqs'])
        self.tokenization_type = vocab_data['tokenization_type']
        self.special_tokens = vocab_data['special_tokens']
        
        # Update special token indices
        self.pad_token_id = self.vocab['<PAD>']
        self.unk_token_id = self.vocab['<UNK>']
        self.sos_token_id = self.vocab['<SOS>']
        self.eos_token_id = self.vocab['<EOS>']
        
        print(f"Vocabulary loaded from {filepath}")


def test_tokenizer():
    """Test the Tokenizer class."""
    print("Testing Tokenizer...")
    
    # Sample texts
    amharic_texts = [
        "የሰላም እለት ነው።",
        "እግዚአብሔር ይመስገን።",
        "የሰማይ ንጉሥ ነው።"
    ]
    
    oromiffa_texts = [
        "Baga nagaan dhuftan!",
        "Waaqayoo galata isaaniif.",
        "Waaqayoo qabeenya isaaniif."
    ]
    
    # Test word-level tokenization
    print("\n--- Testing Word-Level Tokenization ---")
    word_tokenizer = Tokenizer(tokenization_type='word', vocab_size=100)
    word_tokenizer.fit(amharic_texts + oromiffa_texts)
    
    # Test encoding/decoding
    test_text = "የሰላም እለት ነው።"
    encoded = word_tokenizer.encode(test_text)
    decoded = word_tokenizer.decode(encoded)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Test character-level tokenization
    print("\n--- Testing Character-Level Tokenization ---")
    char_tokenizer = Tokenizer(tokenization_type='char', vocab_size=200)
    char_tokenizer.fit(amharic_texts + oromiffa_texts)
    
    encoded_char = char_tokenizer.encode(test_text)
    decoded_char = char_tokenizer.decode(encoded_char)
    
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded_char}")
    print(f"Decoded: {decoded_char}")
    
    # Test batch operations
    print("\n--- Testing Batch Operations ---")
    batch_encoded = word_tokenizer.batch_encode(amharic_texts[:2])
    batch_decoded = word_tokenizer.batch_decode(batch_encoded)
    
    print("Batch encoded:", batch_encoded)
    print("Batch decoded:", batch_decoded)
    
    print("✓ Tokenizer tests passed!")


if __name__ == "__main__":
    test_tokenizer()
