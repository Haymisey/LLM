"""
Tests for Training Loop Implementation (Checkpoint 7)
====================================================

This module tests all training components including loss functions, metrics,
optimizers, training loops, checkpointing, and progress monitoring.
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from training.loss_functions import CrossEntropyLoss, LabelSmoothingLoss
from training.metrics import TranslationMetrics, TrainingMetrics
from training.optimizers import SGDOptimizer, AdamOptimizer, LearningRateScheduler
from training.training_loop import TrainingLoop, ValidationLoop
from training.progress_monitor import ProgressMonitor, TrainingLogger


class TestLossFunctions(unittest.TestCase):
    """Test loss functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 3
        self.vocab_size = 5
        
        # Test data
        self.logits = np.random.randn(self.batch_size, self.seq_len, self.vocab_size)
        self.targets = np.random.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.padding_mask = np.ones((self.batch_size, self.seq_len))
        self.padding_mask[0, -1] = 0  # One padding token
    
    def test_cross_entropy_loss_initialization(self):
        """Test CrossEntropyLoss initialization."""
        loss_fn = CrossEntropyLoss(ignore_index=0, reduction='mean')
        self.assertEqual(loss_fn.ignore_index, 0)
        self.assertEqual(loss_fn.reduction, 'mean')
    
    def test_cross_entropy_loss_forward(self):
        """Test CrossEntropyLoss forward pass."""
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(self.logits, self.targets, self.padding_mask)
        
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
        self.assertLess(loss, 10)  # Reasonable loss range
    
    def test_cross_entropy_loss_backward(self):
        """Test CrossEntropyLoss backward pass."""
        loss_fn = CrossEntropyLoss()
        loss_fn.forward(self.logits, self.targets, self.padding_mask)
        grad_logits = loss_fn.backward()
        
        self.assertEqual(grad_logits.shape, self.logits.shape)
        self.assertTrue(np.all(np.isfinite(grad_logits)))
    
    def test_label_smoothing_loss_initialization(self):
        """Test LabelSmoothingLoss initialization."""
        loss_fn = LabelSmoothingLoss(smoothing=0.1)
        self.assertEqual(loss_fn.smoothing, 0.1)
    
    def test_label_smoothing_loss_forward(self):
        """Test LabelSmoothingLoss forward pass."""
        loss_fn = LabelSmoothingLoss(smoothing=0.1)
        loss = loss_fn.forward(self.logits, self.targets, self.padding_mask)
        
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
    
    def test_label_smoothing_loss_backward(self):
        """Test LabelSmoothingLoss backward pass."""
        loss_fn = LabelSmoothingLoss(smoothing=0.1)
        loss_fn.forward(self.logits, self.targets, self.padding_mask)
        grad_logits = loss_fn.backward()
        
        self.assertEqual(grad_logits.shape, self.logits.shape)
        self.assertTrue(np.all(np.isfinite(grad_logits)))


class TestMetrics(unittest.TestCase):
    """Test metrics classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.seq_len = 3
        self.vocab_size = 5
        
        # Test data
        self.predictions = np.random.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.targets = np.random.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        self.logits = np.random.randn(self.batch_size, self.seq_len, self.vocab_size)
        self.padding_mask = np.ones((self.batch_size, self.seq_len))
        self.padding_mask[0, -1] = 0  # One padding token
    
    def test_translation_metrics_initialization(self):
        """Test TranslationMetrics initialization."""
        metrics = TranslationMetrics()
        self.assertIsInstance(metrics, TranslationMetrics)
    
    def test_translation_metrics_accuracy(self):
        """Test accuracy computation."""
        metrics = TranslationMetrics()
        
        # Test with identical predictions and targets
        identical_preds = self.targets.copy()
        accuracy = metrics.compute_accuracy(identical_preds, self.targets, self.padding_mask)
        self.assertEqual(accuracy, 1.0)
        
        # Test with different predictions
        different_preds = np.random.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        accuracy = metrics.compute_accuracy(different_preds, self.targets, self.padding_mask)
        self.assertLess(accuracy, 1.0)
    
    def test_translation_metrics_perplexity(self):
        """Test perplexity computation."""
        metrics = TranslationMetrics()
        perplexity = metrics.compute_perplexity(self.logits, self.targets, self.padding_mask)
        
        self.assertIsInstance(perplexity, float)
        self.assertGreater(perplexity, 0)
    
    def test_translation_metrics_bleu(self):
        """Test BLEU score computation."""
        metrics = TranslationMetrics()
        bleu = metrics.compute_bleu_score(self.predictions, self.targets, self.padding_mask)
        
        self.assertIsInstance(bleu, float)
        self.assertGreaterEqual(bleu, 0.0)
        self.assertLessEqual(bleu, 1.0)
    
    def test_training_metrics_initialization(self):
        """Test TrainingMetrics initialization."""
        metrics = TrainingMetrics()
        self.assertIsInstance(metrics, TrainingMetrics)
    
    def test_training_metrics_logging(self):
        """Test metrics logging."""
        metrics = TrainingMetrics()
        
        # Log some metrics
        for step in range(5):
            loss = np.random.random()
            lr = 0.001 * (0.9 ** step)
            grad_norm = np.random.random()
            
            metrics.log_metrics(step, loss, lr, grad_norm)
        
        stats = metrics.get_training_stats()
        self.assertEqual(stats['total_steps'], 5)
        self.assertIn('average_loss', stats)


class TestOptimizers(unittest.TestCase):
    """Test optimizer classes."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parameters = {
            'weight1': np.random.randn(10, 10) * 0.1,
            'weight2': np.random.randn(5, 10) * 0.1,
            'bias': np.random.randn(5) * 0.1
        }
        
        self.gradients = {
            'weight1': np.random.randn(10, 10) * 0.01,
            'weight2': np.random.randn(5, 10) * 0.01,
            'bias': np.random.randn(5) * 0.01
        }
    
    def test_sgd_optimizer_initialization(self):
        """Test SGDOptimizer initialization."""
        optimizer = SGDOptimizer(learning_rate=0.01, momentum=0.9)
        self.assertEqual(optimizer.learning_rate, 0.01)
        self.assertEqual(optimizer.momentum, 0.9)
    
    def test_sgd_optimizer_step(self):
        """Test SGDOptimizer step."""
        optimizer = SGDOptimizer(learning_rate=0.01, momentum=0.9)
        updated_params = optimizer.step(self.parameters, self.gradients)
        
        # Check that parameters were updated
        for name in self.parameters:
            self.assertFalse(np.array_equal(self.parameters[name], updated_params[name]))
    
    def test_adam_optimizer_initialization(self):
        """Test AdamOptimizer initialization."""
        optimizer = AdamOptimizer(learning_rate=0.001)
        self.assertEqual(optimizer.learning_rate, 0.001)
        self.assertEqual(optimizer.beta1, 0.9)
        self.assertEqual(optimizer.beta2, 0.999)
    
    def test_adam_optimizer_step(self):
        """Test AdamOptimizer step."""
        optimizer = AdamOptimizer(learning_rate=0.001)
        updated_params = optimizer.step(self.parameters, self.gradients)
        
        # Check that parameters were updated
        for name in self.parameters:
            self.assertFalse(np.array_equal(self.parameters[name], updated_params[name]))
    
    def test_learning_rate_scheduler(self):
        """Test LearningRateScheduler."""
        optimizer = AdamOptimizer(learning_rate=0.001)
        scheduler = LearningRateScheduler(optimizer, 'step', step_size=2, gamma=0.5)
        
        initial_lr = optimizer.get_learning_rate()
        
        # Step scheduler
        scheduler.step()
        scheduler.step()
        
        # Learning rate should decrease after step_size steps
        current_lr = optimizer.get_learning_rate()
        self.assertLess(current_lr, initial_lr)


class TestTrainingLoops(unittest.TestCase):
    """Test training and validation loops."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock model for testing
        class MockModel:
            def __init__(self):
                self.parameters = {
                    'weight': np.random.randn(10, 10) * 0.1,
                    'bias': np.random.randn(10) * 0.1
                }
                self.gradients = {
                    'weight': np.random.randn(10, 10) * 0.01,
                    'bias': np.random.randn(10) * 0.01
                }
            
            def forward(self, source, target, source_mask, target_mask, look_ahead_mask):
                batch_size, seq_len = source.shape
                vocab_size = 100
                logits = np.random.randn(batch_size, seq_len, vocab_size)
                return logits, None
            
            def backward(self, grad_output):
                return np.random.randn(*grad_output.shape), np.random.randn(*grad_output.shape)
            
            def get_parameters(self):
                return self.parameters
            
            def set_parameters(self, params):
                self.parameters = params
            
            def get_gradients(self):
                return self.gradients
            
            def zero_gradients(self):
                self.gradients = {k: np.zeros_like(v) for k, v in self.gradients.items()}
            
            def train(self):
                pass
            
            def eval(self):
                pass
        
        # Mock data loader
        class MockDataLoader:
            def __init__(self, num_batches=3):
                self.num_batches = num_batches
            
            def __iter__(self):
                for i in range(self.num_batches):
                    yield {
                        'source': np.random.randint(0, 100, (2, 10)),
                        'target': np.random.randint(0, 100, (2, 10)),
                        'source_padding_mask': np.ones((2, 10)),
                        'target_padding_mask': np.ones((2, 10)),
                        'look_ahead_mask': np.ones((2, 10, 10))
                    }
            
            def __len__(self):
                return self.num_batches
        
        self.model = MockModel()
        self.train_loader = MockDataLoader(3)
        self.val_loader = MockDataLoader(2)
        
        # Training components
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = AdamOptimizer(learning_rate=0.001)
        self.scheduler = LearningRateScheduler(self.optimizer, 'step', step_size=2)
        self.metrics = TranslationMetrics()
    
    def test_training_loop_initialization(self):
        """Test TrainingLoop initialization."""
        train_loop = TrainingLoop(self.model, self.loss_fn, self.optimizer, self.scheduler)
        self.assertIsInstance(train_loop, TrainingLoop)
    
    def test_training_loop_epoch(self):
        """Test training loop epoch."""
        train_loop = TrainingLoop(self.model, self.loss_fn, self.optimizer, self.scheduler)
        
        # Train for one epoch
        train_stats = train_loop.train_epoch(self.train_loader, 1, max_steps=2)
        
        self.assertIn('epoch', train_stats)
        self.assertIn('loss', train_stats)
        self.assertIn('time', train_stats)
    
    def test_validation_loop_initialization(self):
        """Test ValidationLoop initialization."""
        val_loop = ValidationLoop(self.model, self.metrics)
        self.assertIsInstance(val_loop, ValidationLoop)
    
    def test_validation_loop_validation(self):
        """Test validation loop."""
        val_loop = ValidationLoop(self.model, self.metrics)
        
        # Validate
        val_results = val_loop.validate(self.val_loader, 1)
        
        self.assertIn('epoch', val_results)
        self.assertIn('loss', val_results)
        self.assertIn('metrics', val_results)


class TestProgressMonitoring(unittest.TestCase):
    """Test progress monitoring and logging."""
    
    def test_progress_monitor_initialization(self):
        """Test ProgressMonitor initialization."""
        monitor = ProgressMonitor(total_epochs=10, log_interval=5)
        self.assertEqual(monitor.total_epochs, 10)
        self.assertEqual(monitor.log_interval, 5)
    
    def test_progress_monitor_training_cycle(self):
        """Test complete training cycle with progress monitor."""
        monitor = ProgressMonitor(total_epochs=2, log_interval=2)
        
        monitor.start_training()
        
        for epoch in range(2):
            monitor.start_epoch(epoch, 3)
            
            for batch in range(3):
                loss = 1.0 - epoch * 0.3 - batch * 0.1
                lr = 0.001 * (0.9 ** epoch)
                accuracy = 0.5 + epoch * 0.2 + batch * 0.05
                
                monitor.update_batch(batch, loss, lr, accuracy)
            
            monitor.end_epoch(loss, {'accuracy': accuracy})
        
        monitor.end_training()
        
        stats = monitor.get_training_stats()
        self.assertEqual(stats['total_epochs'], 2)
        self.assertGreater(stats['best_loss'], 0)
    
    def test_training_logger_initialization(self):
        """Test TrainingLogger initialization."""
        logger = TrainingLogger("test_logs", "test_experiment")
        self.assertEqual(logger.experiment_name, "test_experiment")
    
    def test_training_logger_logging(self):
        """Test training logger functionality."""
        logger = TrainingLogger("test_logs", "test_experiment")
        
        # Log configuration
        config = {'model_type': 'transformer', 'vocab_size': 1000}
        logger.log_config(config)
        
        # Log training steps
        for step in range(3):
            logger.log_training_step(0, step, 1.0 - step * 0.1, 0.001)
        
        # Log epoch
        logger.log_epoch(0, {'loss': 0.7, 'accuracy': 0.8})
        
        # Get experiment info
        info = logger.get_experiment_info()
        self.assertIn('experiment_id', info)
        self.assertIn('total_log_entries', info)
        
        # Clean up
        import shutil
        if os.path.exists("test_logs"):
            shutil.rmtree("test_logs")


def run_all_tests():
    """Run all training tests."""
    print("Running Training Loop Implementation Tests...")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestLossFunctions,
        TestMetrics,
        TestOptimizers,
        TestTrainingLoops,
        TestProgressMonitoring
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL TRAINING TESTS PASSED SUCCESSFULLY!")
        print("Checkpoint 7: Training Loop Implementation is COMPLETE!")
    else:
        print("❌ Some tests failed. Please check the output above.")
    
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()
