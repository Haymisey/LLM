"""
Training and Translation Metrics
===============================

This module implements metrics for monitoring training progress and
evaluating translation quality during training and validation.
"""

import numpy as np
from collections import defaultdict


class TranslationMetrics:
    """
    Metrics for evaluating translation quality.
    
    Implements BLEU score, accuracy, and other translation-specific metrics.
    """
    
    def __init__(self):
        """Initialize TranslationMetrics."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_bleu = 0.0
        self.total_accuracy = 0.0
        self.total_perplexity = 0.0
        self.total_samples = 0
        self.batch_count = 0
        
        # For BLEU calculation
        self.reference_ngrams = defaultdict(int)
        self.candidate_ngrams = defaultdict(int)
        self.clipped_ngrams = defaultdict(int)
        self.total_candidate_length = 0
        self.total_reference_length = 0
    
    def compute_accuracy(self, predictions, targets, padding_mask=None):
        """
        Compute token-level accuracy.
        
        Args:
            predictions (np.ndarray): Predicted token IDs [batch_size, seq_len]
            targets (np.ndarray): Target token IDs [batch_size, seq_len]
            padding_mask (np.ndarray): Boolean mask for valid tokens [batch_size, seq_len]
        
        Returns:
            float: Accuracy score
        """
        correct = (predictions == targets).astype(float)
        
        if padding_mask is not None:
            # Apply padding mask
            correct = correct * padding_mask.astype(float)
            total_tokens = np.sum(padding_mask)
        else:
            total_tokens = predictions.size
        
        if total_tokens > 0:
            accuracy = np.sum(correct) / total_tokens
        else:
            accuracy = 0.0
        
        return accuracy
    
    def compute_perplexity(self, logits, targets, padding_mask=None):
        """
        Compute perplexity (exponential of cross-entropy loss).
        
        Args:
            logits (np.ndarray): Model predictions [batch_size, seq_len, vocab_size]
            targets (np.ndarray): Target token IDs [batch_size, seq_len]
            padding_mask (np.ndarray): Boolean mask for valid tokens [batch_size, seq_len]
        
        Returns:
            float: Perplexity score
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        
        # Get target probabilities
        target_probs = np.take_along_axis(probs, targets[..., np.newaxis], axis=-1).squeeze(-1)
        log_probs = np.log(target_probs)
        
        # Apply padding mask if provided
        if padding_mask is not None:
            valid_mask = padding_mask.astype(bool)
            log_probs = log_probs * valid_mask
            num_valid_tokens = np.sum(valid_mask)
        else:
            num_valid_tokens = predictions.size
        
        if num_valid_tokens > 0:
            avg_log_prob = np.sum(log_probs) / num_valid_tokens
            perplexity = np.exp(-avg_log_prob)
        else:
            perplexity = float('inf')
        
        return perplexity
    
    def compute_bleu_score(self, predictions, targets, padding_mask=None, n_grams=4):
        """
        Compute BLEU score for translation quality.
        
        Args:
            predictions (np.ndarray): Predicted token IDs [batch_size, seq_len]
            targets (np.ndarray): Target token IDs [batch_size, seq_len]
            padding_mask (np.ndarray): Boolean mask for valid tokens [batch_size, seq_len]
            n_grams (int): Maximum n-gram order to consider
        
        Returns:
            float: BLEU score
        """
        batch_size, seq_len = predictions.shape
        
        total_bleu = 0.0
        valid_samples = 0
        
        for i in range(batch_size):
            # Get valid tokens for this sample
            if padding_mask is not None:
                valid_mask = padding_mask[i].astype(bool)
                pred_tokens = predictions[i][valid_mask]
                target_tokens = targets[i][valid_mask]
            else:
                pred_tokens = predictions[i]
                target_tokens = targets[i]
            
            if len(pred_tokens) == 0 or len(target_tokens) == 0:
                continue
            
            # Convert token IDs to strings for n-gram computation
            pred_str = ' '.join(map(str, pred_tokens))
            target_str = ' '.join(map(str, target_tokens))
            
            # Compute n-gram precision for this sample
            sample_bleu = self._compute_sample_bleu(pred_str, target_str, n_grams)
            total_bleu += sample_bleu
            valid_samples += 1
        
        if valid_samples > 0:
            bleu_score = total_bleu / valid_samples
        else:
            bleu_score = 0.0
        
        return bleu_score
    
    def _compute_sample_bleu(self, prediction, reference, n_grams):
        """
        Compute BLEU score for a single prediction-reference pair.
        
        Args:
            prediction (str): Predicted translation
            reference (str): Reference translation
            n_grams (int): Maximum n-gram order
        
        Returns:
            float: BLEU score for this sample
        """
        # Simple n-gram implementation
        pred_words = prediction.split()
        ref_words = reference.split()
        
        if len(pred_words) == 0:
            return 0.0
        
        # Compute brevity penalty
        bp = min(1.0, len(pred_words) / len(ref_words)) if len(ref_words) > 0 else 0.0
        
        # Compute n-gram precision
        total_precision = 0.0
        total_ngrams = 0
        
        for n in range(1, min(n_grams + 1, len(pred_words) + 1)):
            pred_ngrams = self._get_ngrams(pred_words, n)
            ref_ngrams = self._get_ngrams(ref_words, n)
            
            if len(pred_ngrams) == 0:
                continue
            
            # Count matches
            matches = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    matches += 1
            
            precision = matches / len(pred_ngrams) if len(pred_ngrams) > 0 else 0.0
            total_precision += np.log(precision + 1e-10)
            total_ngrams += 1
        
        if total_ngrams == 0:
            return 0.0
        
        avg_precision = total_precision / total_ngrams
        bleu = bp * np.exp(avg_precision)
        
        return bleu
    
    def _get_ngrams(self, words, n):
        """Get n-grams from a list of words."""
        if n > len(words):
            return []
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def update(self, predictions, targets, logits=None, padding_mask=None):
        """
        Update metrics with a new batch.
        
        Args:
            predictions (np.ndarray): Predicted token IDs [batch_size, seq_len]
            targets (np.ndarray): Target token IDs [batch_size, seq_len]
            logits (np.ndarray): Model predictions [batch_size, seq_len, vocab_size]
            padding_mask (np.ndarray): Boolean mask for valid tokens [batch_size, seq_len]
        """
        batch_size = predictions.shape[0]
        
        # Compute metrics for this batch
        accuracy = self.compute_accuracy(predictions, targets, padding_mask)
        perplexity = self.compute_perplexity(logits, targets, padding_mask) if logits is not None else 0.0
        bleu = self.compute_bleu_score(predictions, targets, padding_mask)
        
        # Update running totals
        self.total_accuracy += accuracy * batch_size
        self.total_perplexity += perplexity * batch_size
        self.total_bleu += bleu * batch_size
        self.total_samples += batch_size
        self.batch_count += 1
    
    def get_metrics(self):
        """Get current metrics."""
        if self.total_samples == 0:
            return {
                'accuracy': 0.0,
                'perplexity': 0.0,
                'bleu': 0.0,
                'total_samples': 0,
                'batch_count': 0
            }
        
        return {
            'accuracy': self.total_accuracy / self.total_samples,
            'perplexity': self.total_perplexity / self.total_samples,
            'bleu': self.total_bleu / self.total_samples,
            'total_samples': self.total_samples,
            'batch_count': self.batch_count
        }


class TrainingMetrics:
    """
    Metrics for monitoring training progress.
    
    Tracks loss, learning rate, gradient norms, and other training statistics.
    """
    
    def __init__(self):
        """Initialize TrainingMetrics."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.total_grad_norm = 0.0
        self.learning_rates = []
        self.loss_history = []
        self.grad_norm_history = []
        self.batch_count = 0
        self.step_count = 0
        
        # Performance metrics
        self.best_loss = float('inf')
        self.worst_loss = float('-inf')
        self.loss_std = 0.0
    
    def update_loss(self, loss):
        """Update loss metrics."""
        self.total_loss += loss
        self.loss_history.append(loss)
        self.batch_count += 1
        
        # Update best/worst loss
        if loss < self.best_loss:
            self.best_loss = loss
        if loss > self.worst_loss:
            self.worst_loss = loss
    
    def update_grad_norm(self, grad_norm):
        """Update gradient norm metrics."""
        self.total_grad_norm += grad_norm
        self.grad_norm_history.append(grad_norm)
    
    def update_learning_rate(self, lr):
        """Update learning rate history."""
        self.learning_rates.append(lr)
    
    def compute_gradient_statistics(self, gradients):
        """
        Compute gradient statistics.
        
        Args:
            gradients (list): List of gradient arrays
        
        Returns:
            dict: Gradient statistics
        """
        if not gradients:
            return {'norm': 0.0, 'mean': 0.0, 'std': 0.0}
        
        # Flatten all gradients
        flat_grads = []
        for grad in gradients:
            if grad is not None:
                flat_grads.extend(grad.flatten())
        
        if not flat_grads:
            return {'norm': 0.0, 'mean': 0.0, 'std': 0.0}
        
        flat_grads = np.array(flat_grads)
        
        # Compute statistics
        grad_norm = np.linalg.norm(flat_grads)
        grad_mean = np.mean(flat_grads)
        grad_std = np.std(flat_grads)
        
        # Update running totals
        self.update_grad_norm(grad_norm)
        
        return {
            'norm': grad_norm,
            'mean': grad_mean,
            'std': grad_std
        }
    
    def get_training_stats(self):
        """Get current training statistics."""
        if self.batch_count == 0:
            return {
                'average_loss': 0.0,
                'best_loss': float('inf'),
                'worst_loss': float('-inf'),
                'average_grad_norm': 0.0,
                'current_lr': 0.0,
                'total_batches': 0,
                'total_steps': 0
            }
        
        # Compute loss statistics
        loss_array = np.array(self.loss_history)
        loss_std = np.std(loss_array) if len(loss_array) > 1 else 0.0
        
        return {
            'average_loss': self.total_loss / self.batch_count,
            'best_loss': self.best_loss,
            'worst_loss': self.worst_loss,
            'loss_std': loss_std,
            'average_grad_norm': self.total_grad_norm / max(len(self.grad_norm_history), 1),
            'current_lr': self.learning_rates[-1] if self.learning_rates else 0.0,
            'total_batches': self.batch_count,
            'total_steps': self.step_count
        }
    
    def log_metrics(self, step, loss, lr, grad_norm=None):
        """
        Log metrics for a training step.
        
        Args:
            step (int): Current training step
            loss (float): Current loss value
            lr (float): Current learning rate
            grad_norm (float): Current gradient norm
        """
        self.step_count = step + 1  # Increment step count
        self.update_loss(loss)
        self.update_learning_rate(lr)
        
        if grad_norm is not None:
            self.update_grad_norm(grad_norm)
    
    def get_progress_summary(self):
        """Get a summary of training progress."""
        stats = self.get_training_stats()
        
        return {
            'step': self.step_count,
            'batch': self.batch_count,
            'loss': {
                'current': stats['average_loss'],
                'best': stats['best_loss'],
                'worst': stats['worst_loss'],
                'std': stats['loss_std']
            },
            'gradients': {
                'norm': stats['average_grad_norm']
            },
            'learning_rate': stats['current_lr']
        }


def test_metrics():
    """Test the metrics classes."""
    print("Testing Metrics...")
    
    # Test data
    batch_size, seq_len, vocab_size = 2, 3, 5
    predictions = np.random.randint(0, vocab_size, (batch_size, seq_len))
    targets = np.random.randint(0, vocab_size, (batch_size, seq_len))
    logits = np.random.randn(batch_size, seq_len, vocab_size)
    padding_mask = np.ones((batch_size, seq_len))
    padding_mask[0, -1] = 0  # One padding token
    
    # Test TranslationMetrics
    print("\n--- TranslationMetrics ---")
    trans_metrics = TranslationMetrics()
    trans_metrics.update(predictions, targets, logits, padding_mask)
    
    metrics = trans_metrics.get_metrics()
    print(f"Translation metrics: {metrics}")
    
    # Test TrainingMetrics
    print("\n--- TrainingMetrics ---")
    train_metrics = TrainingMetrics()
    
    # Simulate training steps
    for step in range(5):
        loss = np.random.random() * 2.0
        lr = 0.001 * (0.9 ** step)
        grad_norm = np.random.random() * 1.0
        
        train_metrics.log_metrics(step, loss, lr, grad_norm)
    
    stats = train_metrics.get_training_stats()
    print(f"Training stats: {stats}")
    
    progress = train_metrics.get_progress_summary()
    print(f"Progress summary: {progress}")
    
    print("\n✓ Metrics tested successfully!")


if __name__ == "__main__":
    test_metrics()
