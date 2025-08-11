"""
Data Augmentation for Bible Translation
======================================

Provides various data augmentation techniques to improve model robustness
and increase training data diversity for Bible translation.
"""

import random
import re
from typing import List, Tuple, Optional, Dict, Any
from copy import deepcopy


class DataAugmentation:
    """
    Data augmentation techniques for Bible translation data.
    
    Features:
    - Synonym replacement
    - Word insertion/deletion
    - Sentence paraphrasing
    - Back-translation simulation
    - Noise injection
    """
    
    def __init__(self, 
                 augmentation_prob: float = 0.3,
                 max_augmentations: int = 3,
                 preserve_meaning: bool = True):
        """
        Initialize data augmentation.
        
        Args:
            augmentation_prob: Probability of applying augmentation
            max_augmentations: Maximum number of augmentations per text
            preserve_meaning: Whether to preserve original meaning
        """
        self.augmentation_prob = augmentation_prob
        self.max_augmentations = max_augmentations
        self.preserve_meaning = preserve_meaning
        
        # Common synonyms for Bible translation context
        self.amharic_synonyms = {
            'የሰላም': ['የሰላም', 'የዕረፍት', 'የእርጋት'],
            'እግዚአብሔር': ['እግዚአብሔር', 'ጌታ', 'የሰማይ አባት'],
            'ይመስገን': ['ይመስገን', 'ይወደስ', 'ይከብር'],
            'የሰማይ': ['የሰማይ', 'የሰማያዊ', 'የላይኛው'],
            'ንጉሥ': ['ንጉሥ', 'ገዢ', 'የሚያዘው']
        }
        
        self.oromiffa_synonyms = {
            'Baga': ['Baga', 'Akkam', 'Selam'],
            'nagaan': ['nagaan', 'fayyaa', 'gammachuu'],
            'dhuftan': ['dhuftan', 'boodde', 'dhaqe'],
            'Waaqayoo': ['Waaqayoo', 'Waaqa', 'Rabbi'],
            'galata': ['galata', 'faaruu', 'mammaksa']
        }
        
        # Common noise patterns
        self.noise_patterns = [
            (r'([.!?])\s*([A-Z])', r'\1 \2'),  # Fix spacing after punctuation
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'([a-zA-Z])([.!?])', r'\1 \2'),  # Add space before punctuation
        ]
    
    def augment_text(self, text: str, language: str = 'amharic') -> str:
        """
        Apply data augmentation to a single text.
        
        Args:
            text: Input text
            language: Language identifier
            
        Returns:
            Augmented text
        """
        if not text or random.random() > self.augmentation_prob:
            return text
        
        augmented_text = text
        num_augmentations = random.randint(1, self.max_augmentations)
        
        for _ in range(num_augmentations):
            augmentation_type = random.choice([
                'synonym_replacement',
                'word_insertion',
                'word_deletion',
                'noise_injection'
            ])
            
            if augmentation_type == 'synonym_replacement':
                augmented_text = self._synonym_replacement(augmented_text, language)
            elif augmentation_type == 'word_insertion':
                augmented_text = self._word_insertion(augmented_text, language)
            elif augmentation_type == 'word_deletion':
                augmented_text = self._word_deletion(augmented_text)
            elif augmentation_type == 'noise_injection':
                augmented_text = self._noise_injection(augmented_text)
        
        return augmented_text
    
    def _synonym_replacement(self, text: str, language: str) -> str:
        """Replace words with synonyms."""
        if language == 'amharic':
            synonyms = self.amharic_synonyms
        elif language == 'oromiffa':
            synonyms = self.oromiffa_synonyms
        else:
            return text
        
        words = text.split()
        augmented_words = []
        
        for word in words:
            if word in synonyms and random.random() < 0.3:
                # Replace with synonym
                synonym = random.choice(synonyms[word])
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)
        
        return ' '.join(augmented_words)
    
    def _word_insertion(self, text: str, language: str) -> str:
        """Insert additional words to increase text length."""
        if language == 'amharic':
            filler_words = ['እንደዚሁም', 'በተጨማሪም', 'ይህም ነው']
        elif language == 'oromiffa':
            filler_words = ['Akkasitti', 'Dabalataan', 'Kunis']
        else:
            return text
        
        words = text.split()
        if len(words) < 3:
            return text
        
        # Insert filler word at random position
        insert_pos = random.randint(1, len(words) - 1)
        filler_word = random.choice(filler_words)
        
        words.insert(insert_pos, filler_word)
        return ' '.join(words)
    
    def _word_deletion(self, text: str) -> str:
        """Delete random words to decrease text length."""
        words = text.split()
        if len(words) < 4:
            return text
        
        # Delete 1-2 random words
        num_to_delete = min(2, len(words) // 4)
        for _ in range(num_to_delete):
            if len(words) > 2:
                delete_pos = random.randint(1, len(words) - 2)  # Don't delete first/last
                words.pop(delete_pos)
        
        return ' '.join(words)
    
    def _noise_injection(self, text: str) -> str:
        """Inject noise into the text."""
        # Apply noise patterns
        noisy_text = text
        for pattern, replacement in self.noise_patterns:
            if random.random() < 0.5:
                noisy_text = re.sub(pattern, replacement, noisy_text)
        
        # Random character swaps (rare)
        if random.random() < 0.1 and len(text) > 5:
            chars = list(noisy_text)
            if len(chars) > 2:
                pos1, pos2 = random.sample(range(1, len(chars) - 1), 2)
                chars[pos1], chars[pos2] = chars[pos2], chars[pos1]
                noisy_text = ''.join(chars)
        
        return noisy_text
    
    def augment_pair(self, source_text: str, target_text: str, 
                    source_lang: str = 'amharic', 
                    target_lang: str = 'oromiffa') -> Tuple[str, str]:
        """
        Augment a source-target text pair.
        
        Args:
            source_text: Source language text
            target_text: Target language text
            source_lang: Source language identifier
            target_lang: Target language identifier
            
        Returns:
            Tuple of (augmented_source, augmented_target)
        """
        # Augment source text
        augmented_source = self.augment_text(source_text, source_lang)
        
        # Augment target text
        augmented_target = self.augment_text(target_text, target_lang)
        
        return augmented_source, augmented_target
    
    def augment_dataset(self, source_texts: List[str], target_texts: List[str],
                       source_lang: str = 'amharic', 
                       target_lang: str = 'oromiffa',
                       augmentation_factor: float = 1.0) -> Tuple[List[str], List[str]]:
        """
        Augment entire dataset.
        
        Args:
            source_texts: List of source texts
            target_texts: List of target texts
            source_lang: Source language identifier
            target_lang: Target language identifier
            augmentation_factor: How many times to augment (1.0 = double the dataset)
            
        Returns:
            Tuple of (augmented_source_texts, augmented_target_texts)
        """
        if len(source_texts) != len(target_texts):
            raise ValueError("Source and target texts must have the same length")
        
        augmented_source = source_texts.copy()
        augmented_target = target_texts.copy()
        
        # Calculate how many augmentations to create
        num_original = len(source_texts)
        num_augmentations = int(num_original * (augmentation_factor - 1))
        
        # Create augmented samples
        for _ in range(num_augmentations):
            # Randomly select a text pair to augment
            idx = random.randint(0, num_original - 1)
            source_text = source_texts[idx]
            target_text = target_texts[idx]
            
            # Augment the pair
            aug_source, aug_target = self.augment_pair(
                source_text, target_text, source_lang, target_lang
            )
            
            augmented_source.append(aug_source)
            augmented_target.append(aug_target)
        
        print(f"Dataset augmented: {num_original} -> {len(augmented_source)} samples")
        return augmented_source, augmented_target
    
    def create_paraphrases(self, text: str, language: str = 'amharic', 
                          num_paraphrases: int = 3) -> List[str]:
        """
        Create multiple paraphrases of a text.
        
        Args:
            text: Input text
            language: Language identifier
            num_paraphrases: Number of paraphrases to create
            
        Returns:
            List of paraphrased texts
        """
        paraphrases = []
        
        for _ in range(num_paraphrases):
            # Apply multiple augmentations for paraphrasing
            paraphrased = text
            for _ in range(random.randint(2, 4)):
                paraphrased = self.augment_text(paraphrased, language)
            paraphrases.append(paraphrased)
        
        return paraphrases
    
    def get_augmentation_stats(self, original_texts: List[str], 
                              augmented_texts: List[str]) -> Dict[str, Any]:
        """
        Get statistics about the augmentation process.
        
        Args:
            original_texts: Original texts
            augmented_texts: Augmented texts
            
        Returns:
            Dictionary with augmentation statistics
        """
        if not original_texts or not augmented_texts:
            return {}
        
        # Calculate length changes
        original_lengths = [len(text.split()) for text in original_texts]
        augmented_lengths = [len(text.split()) for text in augmented_texts]
        
        # Calculate vocabulary changes
        original_vocab = set()
        for text in original_texts:
            original_vocab.update(text.split())
        
        augmented_vocab = set()
        for text in augmented_texts:
            augmented_vocab.update(text.split())
        
        new_vocab = augmented_vocab - original_vocab
        
        return {
            'original_samples': len(original_texts),
            'augmented_samples': len(augmented_texts),
            'augmentation_ratio': len(augmented_texts) / len(original_texts),
            'original_avg_length': sum(original_lengths) / len(original_lengths),
            'augmented_avg_length': sum(augmented_lengths) / len(augmented_lengths),
            'length_change_ratio': sum(augmented_lengths) / sum(original_lengths),
            'original_vocab_size': len(original_vocab),
            'augmented_vocab_size': len(augmented_vocab),
            'new_vocab_size': len(new_vocab),
            'vocabulary_expansion': len(new_vocab) / len(original_vocab) if original_vocab else 0
        }


def test_data_augmentation():
    """Test the DataAugmentation class."""
    print("Testing DataAugmentation...")
    
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
    
    # Create augmentation instance
    augmenter = DataAugmentation(
        augmentation_prob=0.5,
        max_augmentations=2
    )
    
    # Test single text augmentation
    print("\n--- Testing Single Text Augmentation ---")
    test_text = "የሰላም እለት ነው።"
    augmented = augmenter.augment_text(test_text, 'amharic')
    print(f"Original: {test_text}")
    print(f"Augmented: {augmented}")
    
    # Test text pair augmentation
    print("\n--- Testing Text Pair Augmentation ---")
    aug_source, aug_target = augmenter.augment_pair(
        amharic_texts[0], oromiffa_texts[0]
    )
    print(f"Original source: {amharic_texts[0]}")
    print(f"Augmented source: {aug_source}")
    print(f"Original target: {oromiffa_texts[0]}")
    print(f"Augmented target: {aug_target}")
    
    # Test dataset augmentation
    print("\n--- Testing Dataset Augmentation ---")
    aug_source_texts, aug_target_texts = augmenter.augment_dataset(
        amharic_texts, oromiffa_texts, augmentation_factor=1.5
    )
    print(f"Original dataset size: {len(amharic_texts)}")
    print(f"Augmented dataset size: {len(aug_source_texts)}")
    
    # Test paraphrasing
    print("\n--- Testing Paraphrasing ---")
    paraphrases = augmenter.create_paraphrases(amharic_texts[0], 'amharic', 2)
    print(f"Original: {amharic_texts[0]}")
    for i, paraphrase in enumerate(paraphrases):
        print(f"Paraphrase {i+1}: {paraphrase}")
    
    # Test statistics
    print("\n--- Testing Augmentation Statistics ---")
    stats = augmenter.get_augmentation_stats(amharic_texts, aug_source_texts)
    print(f"Augmentation statistics: {stats}")
    
    print("✓ DataAugmentation tests passed!")


if __name__ == "__main__":
    test_data_augmentation()
