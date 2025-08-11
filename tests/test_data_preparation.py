"""
Tests for Data Preparation Components
===================================

Comprehensive tests for all data preparation components:
- TextPreprocessor
- Tokenizer  
- Vocabulary
- DataLoader
- DataAugmentation
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import unittest
import numpy as np
from data.text_preprocessor import TextPreprocessor
from data.tokenizer import Tokenizer
from data.vocabulary import Vocabulary
from data.data_loader import DataLoader, MockTokenizer
from data.augmentation import DataAugmentation


class TestTextPreprocessor(unittest.TestCase):
    """Test TextPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
        
        # Test texts
        self.amharic_text = "የሰላም እለት ነው። እግዚአብሔር ይመስገን።"
        self.oromiffa_text = "Baga nagaan dhuftan! Waaqayoo galata isaaniif."
    
    def test_initialization(self):
        """Test TextPreprocessor initialization."""
        self.assertIsInstance(self.preprocessor, TextPreprocessor)
        self.assertTrue(self.preprocessor.normalize_unicode)
        self.assertFalse(self.preprocessor.remove_special_chars)
        self.assertFalse(self.preprocessor.lowercase)
    
    def test_amharic_preprocessing(self):
        """Test Amharic text preprocessing."""
        processed = self.preprocessor.preprocess(self.amharic_text, 'amharic')
        self.assertIsInstance(processed, str)
        self.assertIn('የሰላም', processed)
        self.assertIn('እለት', processed)
    
    def test_oromiffa_preprocessing(self):
        """Test Oromiffa text preprocessing."""
        processed = self.preprocessor.preprocess(self.oromiffa_text, 'oromiffa')
        self.assertIsInstance(processed, str)
        self.assertIn('Baga', processed)
        self.assertIn('nagaan', processed)
    
    def test_sentence_splitting(self):
        """Test sentence splitting."""
        sentences = self.preprocessor.split_sentences(self.amharic_text)
        self.assertIsInstance(sentences, list)
        self.assertGreater(len(sentences), 0)
    
    def test_word_splitting(self):
        """Test word splitting."""
        words = self.preprocessor.split_words(self.amharic_text)
        self.assertIsInstance(words, list)
        self.assertGreater(len(words), 0)
    
    def test_text_statistics(self):
        """Test text statistics calculation."""
        stats = self.preprocessor.get_text_stats(self.amharic_text)
        self.assertIsInstance(stats, dict)
        self.assertIn('characters', stats)
        self.assertIn('words', stats)
        self.assertIn('sentences', stats)
    
    def test_batch_preprocessing(self):
        """Test batch preprocessing."""
        texts = [self.amharic_text, self.oromiffa_text]
        processed = self.preprocessor.batch_preprocess(texts, 'amharic')
        self.assertEqual(len(processed), len(texts))


class TestTokenizer(unittest.TestCase):
    """Test Tokenizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer = Tokenizer(tokenization_type='word', vocab_size=100)
        
        # Test texts
        self.amharic_texts = [
            "የሰላም እለት ነው።",
            "እግዚአብሔር ይመስገን።"
        ]
        
        self.oromiffa_texts = [
            "Baga nagaan dhuftan!",
            "Waaqayoo galata isaaniif."
        ]
    
    def test_initialization(self):
        """Test Tokenizer initialization."""
        self.assertIsInstance(self.tokenizer, Tokenizer)
        self.assertEqual(self.tokenizer.tokenization_type, 'word')
        self.assertEqual(self.tokenizer.vocab_size, 100)
        self.assertIn('<PAD>', self.tokenizer.vocab)
        self.assertIn('<UNK>', self.tokenizer.vocab)
    
    def test_vocabulary_building(self):
        """Test vocabulary building."""
        self.tokenizer.fit(self.amharic_texts + self.oromiffa_texts)
        self.assertGreaterEqual(len(self.tokenizer.vocab), len(self.tokenizer.special_tokens))
    
    def test_encoding_decoding(self):
        """Test text encoding and decoding."""
        self.tokenizer.fit(self.amharic_texts + self.oromiffa_texts)
        
        test_text = "የሰላም እለት ነው።"
        encoded = self.tokenizer.encode(test_text)
        decoded = self.tokenizer.decode(encoded)
        
        self.assertIsInstance(encoded, list)
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(encoded), 0)
    
    def test_batch_operations(self):
        """Test batch encoding and decoding."""
        self.tokenizer.fit(self.amharic_texts + self.oromiffa_texts)
        
        batch_encoded = self.tokenizer.batch_encode(self.amharic_texts[:2])
        batch_decoded = self.tokenizer.batch_decode(batch_encoded)
        
        self.assertEqual(len(batch_encoded), 2)
        self.assertEqual(len(batch_decoded), 2)
    
    def test_special_tokens(self):
        """Test special token handling."""
        self.assertEqual(self.tokenizer.pad_token_id, 0)
        self.assertEqual(self.tokenizer.unk_token_id, 1)
        self.assertEqual(self.tokenizer.sos_token_id, 2)
        self.assertEqual(self.tokenizer.eos_token_id, 3)


class TestVocabulary(unittest.TestCase):
    """Test Vocabulary class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab = Vocabulary(
            source_lang='amharic',
            target_lang='oromiffa',
            max_vocab_size=100,
            min_freq=1
        )
        
        # Test texts
        self.amharic_texts = [
            "የሰላም እለት ነው።",
            "እግዚአብሔር ይመስገን።"
        ]
        
        self.oromiffa_texts = [
            "Baga nagaan dhuftan!",
            "Waaqayoo galata isaaniif."
        ]
    
    def test_initialization(self):
        """Test Vocabulary initialization."""
        self.assertIsInstance(self.vocab, Vocabulary)
        self.assertEqual(self.vocab.source_lang, 'amharic')
        self.assertEqual(self.vocab.target_lang, 'oromiffa')
        self.assertEqual(self.vocab.max_vocab_size, 100)
    
    def test_vocabulary_building(self):
        """Test vocabulary building."""
        self.vocab.build_vocabulary(self.amharic_texts, self.oromiffa_texts)
        
        self.assertGreater(len(self.vocab.source_vocab), 0)
        self.assertGreater(len(self.vocab.target_vocab), 0)
    
    def test_token_id_retrieval(self):
        """Test token ID retrieval."""
        self.vocab.build_vocabulary(self.amharic_texts, self.oromiffa_texts)
        
        # Test source language
        source_size = self.vocab.get_vocab_size('source')
        self.assertGreater(source_size, 0)
        
        # Test target language
        target_size = self.vocab.get_vocab_size('target')
        self.assertGreater(target_size, 0)
    
    def test_statistics(self):
        """Test vocabulary statistics."""
        self.vocab.build_vocabulary(self.amharic_texts, self.oromiffa_texts)
        
        source_stats = self.vocab.get_statistics('source')
        target_stats = self.vocab.get_statistics('target')
        
        self.assertIsInstance(source_stats, dict)
        self.assertIsInstance(target_stats, dict)
        self.assertIn('total_tokens', source_stats)
        self.assertIn('total_tokens', target_stats)
    
    def test_coverage_analysis(self):
        """Test coverage analysis."""
        self.vocab.build_vocabulary(self.amharic_texts, self.oromiffa_texts)
        
        coverage = self.vocab.get_coverage_analysis(self.amharic_texts, 'source')
        self.assertIsInstance(coverage, dict)
        self.assertIn('coverage_rate', coverage)


class TestDataLoader(unittest.TestCase):
    """Test DataLoader class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample data
        self.source_texts = [
            "የሰላም እለት ነው።",
            "እግዚአብሔር ይመስገን።",
            "የሰማይ ንጉሥ ነው።",
            "የምድር ገንዘብ ነው።"
        ]
        
        self.target_texts = [
            "Baga nagaan dhuftan!",
            "Waaqayoo galata isaaniif.",
            "Waaqayoo qabeenya isaaniif.",
            "Waaqayoo barumsa isaaniif."
        ]
        
        # Create mock tokenizers
        self.source_tokenizer = MockTokenizer()
        self.target_tokenizer = MockTokenizer()
        
        # Create data loader
        self.data_loader = DataLoader(
            source_texts=self.source_texts,
            target_texts=self.target_texts,
            source_tokenizer=self.source_tokenizer,
            target_tokenizer=self.target_tokenizer,
            batch_size=2,
            max_source_length=20,
            max_target_length=20
        )
    
    def test_initialization(self):
        """Test DataLoader initialization."""
        self.assertIsInstance(self.data_loader, DataLoader)
        self.assertEqual(len(self.data_loader.source_texts), len(self.data_loader.target_texts))
        self.assertEqual(self.data_loader.batch_size, 2)
    
    def test_batch_retrieval(self):
        """Test batch retrieval."""
        batch = self.data_loader.get_sample_batch()
        
        self.assertIsInstance(batch, dict)
        self.assertIn('source', batch)
        self.assertIn('target', batch)
        self.assertIn('source_padding_mask', batch)
        self.assertIn('look_ahead_mask', batch)
    
    def test_batch_shapes(self):
        """Test batch shapes."""
        batch = self.data_loader.get_sample_batch()
        
        # Check shapes
        self.assertEqual(batch['source'].shape[0], batch['batch_size'])
        self.assertEqual(batch['target'].shape[0], batch['batch_size'])
        self.assertEqual(batch['source'].shape[1], self.data_loader.max_source_length)
        self.assertEqual(batch['target'].shape[1], self.data_loader.max_target_length)
    
    def test_iteration(self):
        """Test data loader iteration."""
        batches = list(self.data_loader)
        self.assertEqual(len(batches), self.data_loader.num_batches)
        
        for batch in batches:
            self.assertIsInstance(batch, dict)
            self.assertIn('source', batch)
            self.assertIn('target', batch)
    
    def test_statistics(self):
        """Test data statistics."""
        stats = self.data_loader.get_data_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('total_samples', stats)
        self.assertIn('source_length_stats', stats)
        self.assertIn('target_length_stats', stats)
    
    def test_padding_masks(self):
        """Test padding mask creation."""
        batch = self.data_loader.get_sample_batch()
        
        source_mask = batch['source_padding_mask']
        target_mask = batch['target_padding_mask']
        
        # Check mask shapes
        self.assertEqual(source_mask.shape, batch['source'].shape)
        self.assertEqual(target_mask.shape, batch['target'].shape)
        
        # Check mask values (0 for padding, 1 for real tokens)
        self.assertTrue(np.all((source_mask == 0) | (source_mask == 1)))


class TestDataAugmentation(unittest.TestCase):
    """Test DataAugmentation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.augmenter = DataAugmentation(
            augmentation_prob=0.5,
            max_augmentations=2
        )
        
        # Test texts
        self.amharic_texts = [
            "የሰላም እለት ነው።",
            "እግዚአብሔር ይመስገን።"
        ]
        
        self.oromiffa_texts = [
            "Baga nagaan dhuftan!",
            "Waaqayoo galata isaaniif."
        ]
    
    def test_initialization(self):
        """Test DataAugmentation initialization."""
        self.assertIsInstance(self.augmenter, DataAugmentation)
        self.assertEqual(self.augmenter.augmentation_prob, 0.5)
        self.assertEqual(self.augmenter.max_augmentations, 2)
    
    def test_single_text_augmentation(self):
        """Test single text augmentation."""
        test_text = "የሰላም እለት ነው።"
        augmented = self.augmenter.augment_text(test_text, 'amharic')
        
        self.assertIsInstance(augmented, str)
        self.assertGreater(len(augmented), 0)
    
    def test_text_pair_augmentation(self):
        """Test text pair augmentation."""
        aug_source, aug_target = self.augmenter.augment_pair(
            self.amharic_texts[0], self.oromiffa_texts[0]
        )
        
        self.assertIsInstance(aug_source, str)
        self.assertIsInstance(aug_target, str)
    
    def test_dataset_augmentation(self):
        """Test dataset augmentation."""
        aug_source, aug_target = self.augmenter.augment_dataset(
            self.amharic_texts, self.oromiffa_texts, augmentation_factor=1.5
        )
        
        self.assertIsInstance(aug_source, list)
        self.assertIsInstance(aug_target, list)
        self.assertGreater(len(aug_source), len(self.amharic_texts))
    
    def test_paraphrasing(self):
        """Test text paraphrasing."""
        paraphrases = self.augmenter.create_paraphrases(
            self.amharic_texts[0], 'amharic', 2
        )
        
        self.assertIsInstance(paraphrases, list)
        self.assertEqual(len(paraphrases), 2)
    
    def test_augmentation_statistics(self):
        """Test augmentation statistics."""
        aug_source, aug_target = self.augmenter.augment_dataset(
            self.amharic_texts, self.oromiffa_texts, augmentation_factor=1.5
        )
        
        stats = self.augmenter.get_augmentation_stats(self.amharic_texts, aug_source)
        
        self.assertIsInstance(stats, dict)
        self.assertIn('augmentation_ratio', stats)
        self.assertIn('original_samples', stats)
        self.assertIn('augmented_samples', stats)


class TestIntegration(unittest.TestCase):
    """Test integration between data preparation components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Sample data
        self.amharic_texts = [
            "የሰላም እለት ነው።",
            "እግዚአብሔር ይመስገን።",
            "የሰማይ ንጉሥ ነው።"
        ]
        
        self.oromiffa_texts = [
            "Baga nagaan dhuftan!",
            "Waaqayoo galata isaaniif.",
            "Waaqayoo qabeenya isaaniif."
        ]
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end data preparation pipeline."""
        # 1. Preprocess texts
        preprocessor = TextPreprocessor()
        processed_amharic = [preprocessor.preprocess(text, 'amharic') for text in self.amharic_texts]
        processed_oromiffa = [preprocessor.preprocess(text, 'oromiffa') for text in self.oromiffa_texts]
        
        # 2. Build vocabulary
        vocab = Vocabulary(source_lang='amharic', target_lang='oromiffa', max_vocab_size=100)
        vocab.build_vocabulary(processed_amharic, processed_oromiffa)
        
        # 3. Create tokenizers
        source_tokenizer = Tokenizer(tokenization_type='word', vocab_size=100)
        target_tokenizer = Tokenizer(tokenization_type='word', vocab_size=100)
        source_tokenizer.fit(processed_amharic)
        target_tokenizer.fit(processed_oromiffa)
        
        # 4. Create data loader
        data_loader = DataLoader(
            source_texts=processed_amharic,
            target_texts=processed_oromiffa,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            batch_size=2
        )
        
        # 5. Test data loading
        batch = data_loader.get_sample_batch()
        self.assertIsInstance(batch, dict)
        self.assertIn('source', batch)
        self.assertIn('target', batch)
        
        # 6. Test augmentation
        augmenter = DataAugmentation()
        aug_source, aug_target = augmenter.augment_dataset(
            processed_amharic, processed_oromiffa, augmentation_factor=1.5
        )
        
        self.assertGreater(len(aug_source), len(processed_amharic))
        self.assertGreater(len(aug_target), len(processed_oromiffa))
        
        print("✓ End-to-end data preparation pipeline test passed!")


def run_all_tests():
    """Run all tests."""
    print("Running Data Preparation Tests...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestTextPreprocessor,
        TestTokenizer,
        TestVocabulary,
        TestDataLoader,
        TestDataAugmentation,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 All tests passed successfully!")
    else:
        print("\n❌ Some tests failed!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()
