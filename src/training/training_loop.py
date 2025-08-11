"""
Training and Validation Loops
=============================

This module implements the main training and validation loops for the Transformer model.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional


class TrainingLoop:
    """
    Main training loop for the Transformer model.
    
    Handles forward pass, loss computation, backward pass, and optimization.
    """
    
    def __init__(self, model, loss_fn, optimizer, scheduler=None, 
                 max_grad_norm=1.0, accumulation_steps=1):
        """
        Initialize training loop.
        
        Args:
            model: Transformer model to train
            loss_fn: Loss function
            optimizer: Optimizer instance
            scheduler: Learning rate scheduler (optional)
            max_grad_norm (float): Maximum gradient norm for clipping
            accumulation_steps (int): Number of steps for gradient accumulation
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.accumulation_steps = accumulation_steps
        
        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.total_loss = 0.0
        self.running_loss = 0.0
        
        # Gradient accumulation
        self.accumulated_gradients = {}
        self.accumulation_count = 0
    
    def train_epoch(self, train_loader, epoch, max_steps=None):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            epoch (int): Current epoch number
            max_steps (int): Maximum training steps (for debugging)
        
        Returns:
            dict: Training statistics for this epoch
        """
        self.epoch_count = epoch
        self.model.train()
        
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        print(f"\nEpoch {epoch} - Training")
        print("=" * 50)
        
        for batch_idx, batch in enumerate(train_loader):
            if max_steps and self.step_count >= max_steps:
                break
            
            # Training step
            batch_loss = self.train_step(batch)
            
            epoch_loss += batch_loss
            num_batches += 1
            
            # Print progress
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx:3d}/{len(train_loader):3d} | "
                      f"Loss: {batch_loss:.4f} | "
                      f"LR: {self.optimizer.get_learning_rate():.6f}")
        
        # Finalize epoch
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        epoch_time = time.time() - start_time
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Learning Rate: {self.optimizer.get_learning_rate():.6f}")
        
        return {
            'epoch': epoch,
            'loss': avg_epoch_loss,
            'time': epoch_time,
            'num_batches': num_batches,
            'learning_rate': self.optimizer.get_learning_rate()
        }
    
    def train_step(self, batch):
        """
        Perform one training step.
        
        Args:
            batch (dict): Batch data with source, target, and masks
        
        Returns:
            float: Loss value for this step
        """
        # Forward pass
        source = batch['source']
        target = batch['target']
        source_padding_mask = batch['source_padding_mask']
        target_padding_mask = batch['target_padding_mask']
        look_ahead_mask = batch['look_ahead_mask']
        
        # Forward pass through model
        logits, _ = self.model.forward(source, target, 
                                      source_padding_mask, 
                                      target_padding_mask, 
                                      look_ahead_mask)
        
        # Compute loss
        loss = self.loss_fn.forward(logits, target, target_padding_mask)
        
        # Backward pass
        grad_logits = self.loss_fn.backward()
        
        # Backward pass through model
        grad_source, grad_target = self.model.backward(grad_logits)
        
        # Get model gradients
        model_gradients = self.model.get_gradients()
        
        # Gradient accumulation
        if self.accumulation_steps > 1:
            self._accumulate_gradients(model_gradients)
            self.accumulation_count += 1
            
            if self.accumulation_count < self.accumulation_steps:
                return loss
        
        # Apply gradient clipping
        if self.max_grad_norm > 0:
            model_gradients = self._clip_gradients(model_gradients)
        
        # Update model parameters
        updated_params = self.optimizer.step(self.model.get_parameters(), model_gradients)
        self.model.set_parameters(updated_params)
        
        # Reset gradients
        self.optimizer.zero_grad()
        self.model.zero_gradients()
        
        # Reset accumulation
        if self.accumulation_steps > 1:
            self.accumulated_gradients = {}
            self.accumulation_count = 0
        
        # Update statistics
        self.step_count += 1
        self.total_loss += loss
        self.running_loss = 0.9 * self.running_loss + 0.1 * loss
        
        return loss
    
    def _accumulate_gradients(self, gradients):
        """Accumulate gradients across multiple steps."""
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            if name not in self.accumulated_gradients:
                self.accumulated_gradients[name] = np.zeros_like(grad)
            
            self.accumulated_gradients[name] += grad
    
    def _clip_gradients(self, gradients):
        """Clip gradients to prevent exploding gradients."""
        total_norm = 0.0
        
        # Compute total norm
        for grad in gradients.values():
            if grad is not None:
                total_norm += np.sum(grad ** 2)
        
        total_norm = np.sqrt(total_norm)
        
        # Clip if necessary
        if total_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / total_norm
            for name in gradients:
                if gradients[name] is not None:
                    gradients[name] *= clip_coef
        
        return gradients
    
    def get_training_stats(self):
        """Get current training statistics."""
        return {
            'step_count': self.step_count,
            'epoch_count': self.epoch_count,
            'total_loss': self.total_loss,
            'running_loss': self.running_loss,
            'learning_rate': self.optimizer.get_learning_rate()
        }
    
    def reset_stats(self):
        """Reset training statistics."""
        self.total_loss = 0.0
        self.running_loss = 0.0


class ValidationLoop:
    """
    Validation loop for evaluating model performance.
    
    Runs inference without gradient computation and computes validation metrics.
    """
    
    def __init__(self, model, metrics, max_eval_steps=None):
        """
        Initialize validation loop.
        
        Args:
            model: Transformer model to evaluate
            metrics: Metrics instance for evaluation
            max_eval_steps (int): Maximum evaluation steps (for debugging)
        """
        self.model = model
        self.metrics = metrics
        self.max_eval_steps = max_eval_steps
    
    def validate(self, val_loader, epoch):
        """
        Perform validation.
        
        Args:
            val_loader: DataLoader for validation data
            epoch (int): Current epoch number
        
        Returns:
            dict: Validation results
        """
        self.model.eval()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        print(f"\nEpoch {epoch} - Validation")
        print("=" * 50)
        
        for batch_idx, batch in enumerate(val_loader):
            if self.max_eval_steps and batch_idx >= self.max_eval_steps:
                break
            
            # Validation step
            batch_loss, batch_metrics = self.validate_step(batch)
            
            total_loss += batch_loss
            num_batches += 1
            
            # Print progress
            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx:3d}/{len(val_loader):3d} | "
                      f"Loss: {batch_loss:.4f}")
        
        # Finalize validation
        avg_loss = total_loss / max(num_batches, 1)
        val_time = time.time() - start_time
        final_metrics = self.metrics.get_metrics()
        
        print(f"\nValidation Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Time: {val_time:.2f}s")
        print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
        print(f"  Perplexity: {final_metrics['perplexity']:.4f}")
        print(f"  BLEU: {final_metrics['bleu']:.4f}")
        
        return {
            'epoch': epoch,
            'loss': avg_loss,
            'time': val_time,
            'num_batches': num_batches,
            'metrics': final_metrics
        }
    
    def validate_step(self, batch):
        """
        Perform one validation step.
        
        Args:
            batch (dict): Batch data with source, target, and masks
        
        Returns:
            tuple: (loss, metrics_dict)
        """
        # Forward pass
        source = batch['source']
        target = batch['target']
        source_padding_mask = batch['source_padding_mask']
        target_padding_mask = batch['target_padding_mask']
        look_ahead_mask = batch['look_ahead_mask']
        
        # Forward pass through model (no gradients)
        logits, _ = self.model.forward(source, target, 
                                      source_padding_mask, 
                                      target_padding_mask, 
                                      look_ahead_mask)
        
        # Compute loss (we need to create a loss function instance)
        from .loss_functions import CrossEntropyLoss
        loss_fn = CrossEntropyLoss()
        loss = loss_fn.forward(logits, target, target_padding_mask)
        
        # Get predictions
        predictions = np.argmax(logits, axis=-1)
        
        # Update metrics
        self.metrics.update(predictions, target, logits, target_padding_mask)
        
        return loss, self.metrics.get_metrics()
    
    def generate_samples(self, val_loader, num_samples=3):
        """
        Generate sample translations for qualitative evaluation.
        
        Args:
            val_loader: DataLoader for validation data
            num_samples (int): Number of samples to generate
        
        Returns:
            list: List of generated samples
        """
        self.model.eval()
        samples = []
        
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break
            
            source = batch['source'][0]  # Take first sample
            target = batch['target'][0]
            
            # Generate translation
            generated = self.model.generate(source, max_length=50)
            
            samples.append({
                'source': source,
                'target': target,
                'generated': generated
            })
        
        return samples


def test_training_loop():
    """Test the training and validation loops."""
    print("Testing Training and Validation Loops...")
    
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
        
        def generate(self, source, max_length=50):
            return np.random.randint(0, 100, max_length)
    
    # Mock data loader
    class MockDataLoader:
        def __init__(self, num_batches=5):
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
    
    # Test components
    model = MockModel()
    loss_fn = CrossEntropyLoss()
    optimizer = AdamOptimizer(learning_rate=0.001)
    scheduler = LearningRateScheduler(optimizer, 'step', step_size=2)
    metrics = TranslationMetrics()
    
    # Test training loop
    print("\n--- Training Loop ---")
    train_loop = TrainingLoop(model, loss_fn, optimizer, scheduler)
    
    train_loader = MockDataLoader(3)
    train_stats = train_loop.train_epoch(train_loop, 1, max_steps=2)
    print(f"Training stats: {train_stats}")
    
    # Test validation loop
    print("\n--- Validation Loop ---")
    val_loop = ValidationLoop(model, metrics)
    
    val_loader = MockDataLoader(2)
    val_results = val_loop.validate(val_loader, 1)
    print(f"Validation results: {val_results}")
    
    print("\n✓ Training and validation loops tested successfully!")


if __name__ == "__main__":
    # Import required for testing
    from .loss_functions import CrossEntropyLoss
    from .optimizers import AdamOptimizer, LearningRateScheduler
    from .metrics import TranslationMetrics
    
    test_training_loop()
