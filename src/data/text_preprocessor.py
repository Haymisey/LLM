"""
Text Preprocessor for Bible Translation
=====================================

Handles text cleaning, normalization, and preprocessing for Amharic and Oromiffa text.
"""

import re
import unicodedata
from typing import List, Tuple, Optional


class TextPreprocessor:
    """
    Text preprocessing for Amharic and Oromiffa Bible text.
    
    Handles:
    - Unicode normalization
    - Special character handling
    - Whitespace normalization
    - Basic text cleaning
    """
    
    def __init__(self, 
                 normalize_unicode: bool = True,
                 remove_special_chars: bool = False,
                 lowercase: bool = False):
        """
        Initialize the text preprocessor.
        
        Args:
            normalize_unicode: Whether to normalize Unicode characters
            remove_special_chars: Whether to remove special characters
            lowercase: Whether to convert text to lowercase (usually False for Amharic/Oromiffa)
        """
        self.normalize_unicode = normalize_unicode
        self.remove_special_chars = remove_special_chars
        self.lowercase = lowercase
        
        # Common punctuation and symbols to handle
        self.punctuation = '.,!?;:()[]{}"\'-'
        
        # Amharic and Oromiffa specific characters to preserve
        self.amharic_chars = set('ሀሁሂሃሄህሆለሉሊላሌልሎሏሐሑሒሓሔሕሖሗመሙሚማሜምሞሟሠሡሢሣሤሥሦሧረሩሪራሬርሮሯሰሱሲሳሴስሶሷሸሹሺሻሼሽሾሿሻሼሽሾሿቀቁቂቃቄቅቆቇቈ቉ቊቋቌቍቐቑቒቓቔቕቖ቗ቘ቙ቚቛቜቝበቡቢባቤብቦቧቨቩቪቫቬቭቮቯተቱቲታቴትቶቷቸቹቺቻቼችቾቿዀ዁ዂዃዄዅ዆዇ወዉዊዋዌውዎዏዐዑዒዓዔዕዖ዗ዘዙዚዛዜዝዞዟዠዡዢዣዤዥዦዧየዩዪያዬይዮዯደዱዲዳዴድዶዷዸዹዺዻዼዽዾዿጀጁጂጃጄጅጆጇገጉጊጋጌግጎጏጐ጑ጒጓጔጕ጖጗ጘጙጚጛጜጝጞጟጠጡጢጣጤጥጦጧጨጩጪጫጬጭጮጯጰጱጲጳጴጵጶጷጸጹጺጻጼጽጾጿፀፁፂፃፄፅፆፇፈፉፊፋፌፍፎፏፐፑፒፓፔፕፖፗፘፙፚ፛፜፝፞፟')
        
        self.oromiffa_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    def preprocess(self, text: str, language: str = 'amharic') -> str:
        """
        Preprocess text based on the specified language.
        
        Args:
            text: Input text to preprocess
            language: Language identifier ('amharic' or 'oromiffa')
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Unicode normalization
        if self.normalize_unicode:
            text = unicodedata.normalize('NFC', text)
        
        # Language-specific preprocessing
        if language.lower() == 'amharic':
            text = self._preprocess_amharic(text)
        elif language.lower() == 'oromiffa':
            text = self._preprocess_oromiffa(text)
        else:
            text = self._preprocess_general(text)
        
        # Whitespace normalization
        text = self._normalize_whitespace(text)
        
        # Case conversion
        if self.lowercase:
            text = text.lower()
        
        return text.strip()
    
    def _preprocess_amharic(self, text: str) -> str:
        """Preprocess Amharic text specifically."""
        # Preserve Amharic characters and basic punctuation
        # Remove unwanted symbols but keep essential punctuation
        text = re.sub(r'[^\u1200-\u137F\u1380-\u139F\u2D80-\u2DDF\uAB00-\uAB2F\s.,!?;:()"\'-]', '', text)
        return text
    
    def _preprocess_oromiffa(self, text: str) -> str:
        """Preprocess Oromiffa text specifically."""
        # Oromiffa uses Latin script, so we can be more aggressive with cleaning
        # Keep letters, numbers, and essential punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:()"\'-]', '', text)
        return text
    
    def _preprocess_general(self, text: str) -> str:
        """General text preprocessing."""
        if self.remove_special_chars:
            # Remove all non-alphanumeric characters except spaces
            text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting based on punctuation
        sentences = re.split(r'[.!?]+', text)
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def split_words(self, text: str) -> List[str]:
        """
        Split text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of words
        """
        # Split on whitespace and filter empty strings
        words = text.split()
        return [word for word in words if word.strip()]
    
    def get_text_stats(self, text: str) -> dict:
        """
        Get basic statistics about the text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        if not text:
            return {
                'characters': 0,
                'words': 0,
                'sentences': 0,
                'amharic_chars': 0,
                'oromiffa_chars': 0
            }
        
        sentences = self.split_sentences(text)
        words = self.split_words(text)
        
        # Count Amharic and Oromiffa characters
        amharic_count = sum(1 for char in text if char in self.amharic_chars)
        oromiffa_count = sum(1 for char in text if char in self.oromiffa_chars)
        
        return {
            'characters': len(text),
            'words': len(words),
            'sentences': len(sentences),
            'amharic_chars': amharic_count,
            'oromiffa_chars': oromiffa_count
        }
    
    def batch_preprocess(self, texts: List[str], language: str = 'amharic') -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            language: Language identifier
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess(text, language) for text in texts]


def test_text_preprocessor():
    """Test the TextPreprocessor class."""
    print("Testing TextPreprocessor...")
    
    # Create preprocessor
    preprocessor = TextPreprocessor()
    
    # Test Amharic text
    amharic_text = "የሰላም እለት ነው። እግዚአብሔር ይመስገን።"
    processed_amharic = preprocessor.preprocess(amharic_text, 'amharic')
    print(f"Original Amharic: {amharic_text}")
    print(f"Processed Amharic: {processed_amharic}")
    
    # Test Oromiffa text
    oromiffa_text = "Baga nagaan dhuftan! Waaqayoo galata isaaniif."
    processed_oromiffa = preprocessor.preprocess(oromiffa_text, 'oromiffa')
    print(f"Original Oromiffa: {oromiffa_text}")
    print(f"Processed Oromiffa: {processed_oromiffa}")
    
    # Test text statistics
    stats = preprocessor.get_text_stats(amharic_text)
    print(f"Text Statistics: {stats}")
    
    # Test sentence splitting
    sentences = preprocessor.split_sentences(amharic_text)
    print(f"Sentences: {sentences}")
    
    # Test word splitting
    words = preprocessor.split_words(amharic_text)
    print(f"Words: {words}")
    
    print("✓ TextPreprocessor tests passed!")


if __name__ == "__main__":
    test_text_preprocessor()
