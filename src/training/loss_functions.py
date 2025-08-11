"""
Loss Functions for Transformer Training
======================================

This module implements loss functions specifically designed for sequence-to-sequence
translation tasks, including cross-entropy loss and label smoothing.
"""

import numpy as np


class CrossEntropyLoss:
    """
    Cross-Entropy Loss for sequence-to-sequence translation.
    
    Computes the negative log-likelihood loss between predicted logits
    and target token IDs, with optional padding mask support.
    """
    
    def __init__(self, ignore_index=0, reduction='mean'):
        """
        Initialize CrossEntropyLoss.
        
        Args:
            ignore_index (int): Token ID to ignore in loss computation (e.g., PAD token)
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.reset()
    
    def reset(self):
        """Reset loss statistics."""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.batch_count = 0
    
    def forward(self, logits, targets, padding_mask=None):
        """
        Compute forward pass of cross-entropy loss.
        
        Args:
            logits (np.ndarray): Model predictions [batch_size, seq_len, vocab_size]
            targets (np.ndarray): Target token IDs [batch_size, seq_len]
            padding_mask (np.ndarray): Boolean mask for valid tokens [batch_size, seq_len]
        
        Returns:
            float: Computed loss value
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        
        # Compute negative log-likelihood
        target_probs = np.take_along_axis(probs, targets[..., np.newaxis], axis=-1).squeeze(-1)
        log_probs = np.log(target_probs)
        
        # Apply padding mask if provided
        if padding_mask is not None:
            # Invert mask so True = valid token, False = padding
            valid_mask = padding_mask.astype(bool)
            log_probs = log_probs * valid_mask
        
        # Apply ignore index mask
        ignore_mask = (targets != self.ignore_index).astype(float)
        if padding_mask is not None:
            ignore_mask = ignore_mask * valid_mask.astype(float)
        
        # Compute loss for valid tokens only
        masked_log_probs = log_probs * ignore_mask
        loss = -np.sum(masked_log_probs)
        
        # Count valid tokens
        num_valid_tokens = np.sum(ignore_mask)
        
        # Apply reduction
        if self.reduction == 'mean' and num_valid_tokens > 0:
            loss = loss / num_valid_tokens
        elif self.reduction == 'sum':
            pass  # Already summed
        elif self.reduction == 'none':
            loss = masked_log_probs
        
        # Store for backward pass
        self.logits = logits
        self.targets = targets
        self.padding_mask = padding_mask
        self.ignore_mask = ignore_mask
        self.probs = probs
        
        # Update statistics
        self.total_loss += float(loss)
        self.total_tokens += int(num_valid_tokens)
        self.batch_count += 1
        
        return loss
    
    def backward(self, grad_output=1.0):
        """
        Compute gradients for backward pass.
        
        Args:
            grad_output (float): Gradient from upstream layers
        
        Returns:
            np.ndarray: Gradients with respect to logits
        """
        batch_size, seq_len, vocab_size = self.logits.shape
        
        # Create one-hot encoding of targets
        targets_one_hot = np.zeros((batch_size, seq_len, vocab_size))
        np.put_along_axis(targets_one_hot, self.targets[..., np.newaxis], 1, axis=-1)
        
        # Compute gradients: dL/dlogits = (probs - targets_one_hot) * grad_output
        grad_logits = (self.probs - targets_one_hot) * grad_output
        
        # Apply masks
        if self.padding_mask is not None:
            valid_mask = self.padding_mask.astype(bool)
            grad_logits = grad_logits * valid_mask[..., np.newaxis]
        
        grad_logits = grad_logits * self.ignore_mask[..., np.newaxis]
        
        return grad_logits
    
    def get_statistics(self):
        """Get loss statistics."""
        return {
            'total_loss': self.total_loss,
            'total_tokens': self.total_tokens,
            'batch_count': self.batch_count,
            'average_loss': self.total_loss / max(self.batch_count, 1),
            'per_token_loss': self.total_loss / max(self.total_tokens, 1)
        }


class LabelSmoothingLoss:
    """
    Label Smoothing Loss for improved generalization.
    
    Applies label smoothing to prevent overconfidence and improve
    model generalization during training.
    """
    
    def __init__(self, smoothing=0.1, ignore_index=0, reduction='mean'):
        """
        Initialize LabelSmoothingLoss.
        
        Args:
            smoothing (float): Label smoothing factor (0.0 = no smoothing)
            ignore_index (int): Token ID to ignore in loss computation
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.reset()
    
    def reset(self):
        """Reset loss statistics."""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.batch_count = 0
    
    def forward(self, logits, targets, padding_mask=None):
        """
        Compute forward pass of label smoothing loss.
        
        Args:
            logits (np.ndarray): Model predictions [batch_size, seq_len, vocab_size]
            targets (np.ndarray): Target token IDs [batch_size, seq_len]
            padding_mask (np.ndarray): Boolean mask for valid tokens [batch_size, seq_len]
        
        Returns:
            float: Computed loss value
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        
        # Create smoothed targets
        smoothed_targets = np.full((batch_size, seq_len, vocab_size), 
                                 self.smoothing / (vocab_size - 1))
        
        # Set correct target probability
        np.put_along_axis(smoothed_targets, targets[..., np.newaxis], 
                          1.0 - self.smoothing, axis=-1)
        
        # Compute KL divergence loss
        log_probs = np.log(probs)
        loss = -np.sum(smoothed_targets * log_probs, axis=-1)
        
        # Apply padding mask if provided
        if padding_mask is not None:
            valid_mask = padding_mask.astype(bool)
            loss = loss * valid_mask
        
        # Apply ignore index mask
        ignore_mask = (targets != self.ignore_index).astype(float)
        if padding_mask is not None:
            ignore_mask = ignore_mask * valid_mask.astype(float)
        
        # Compute loss for valid tokens only
        masked_loss = loss * ignore_mask
        
        # Apply reduction
        if self.reduction == 'mean':
            num_valid_tokens = np.sum(ignore_mask)
            if num_valid_tokens > 0:
                loss = np.sum(masked_loss) / num_valid_tokens
            else:
                loss = 0.0
        elif self.reduction == 'sum':
            loss = np.sum(masked_loss)
        elif self.reduction == 'none':
            loss = masked_loss
        
        # Store for backward pass
        self.logits = logits
        self.targets = targets
        self.padding_mask = padding_mask
        self.ignore_mask = ignore_mask
        self.probs = probs
        self.smoothed_targets = smoothed_targets
        
        # Update statistics
        self.total_loss += float(loss)
        self.total_tokens += int(np.sum(ignore_mask))
        self.batch_count += 1
        
        return loss
    
    def backward(self, grad_output=1.0):
        """
        Compute gradients for backward pass.
        
        Args:
            grad_output (float): Gradient from upstream layers
        
        Returns:
            np.ndarray: Gradients with respect to logits
        """
        batch_size, seq_len, vocab_size = self.logits.shape
        
        # Compute gradients: dL/dlogits = (probs - smoothed_targets) * grad_output
        grad_logits = (self.probs - self.smoothed_targets) * grad_output
        
        # Apply masks
        if self.padding_mask is not None:
            valid_mask = self.padding_mask.astype(bool)
            grad_logits = grad_logits * valid_mask[..., np.newaxis]
        
        grad_logits = grad_logits * self.ignore_mask[..., np.newaxis]
        
        return grad_logits
    
    def get_statistics(self):
        """Get loss statistics."""
        return {
            'total_loss': self.total_loss,
            'total_tokens': self.total_tokens,
            'batch_count': self.batch_count,
            'average_loss': self.total_loss / max(self.batch_count, 1),
            'per_token_loss': self.total_loss / max(self.total_tokens, 1),
            'smoothing_factor': self.smoothing
        }


def test_loss_functions():
    """Test the loss functions."""
    print("Testing Loss Functions...")
    
    # Test data
    batch_size, seq_len, vocab_size = 2, 3, 5
    logits = np.random.randn(batch_size, seq_len, vocab_size)
    targets = np.random.randint(0, vocab_size, (batch_size, seq_len))
    padding_mask = np.ones((batch_size, seq_len))
    padding_mask[0, -1] = 0  # One padding token
    
    # Test CrossEntropyLoss
    print("\n--- CrossEntropyLoss ---")
    ce_loss = CrossEntropyLoss()
    loss_value = ce_loss.forward(logits, targets, padding_mask)
    grad_logits = ce_loss.backward()
    
    print(f"Loss value: {loss_value:.4f}")
    print(f"Gradient shape: {grad_logits.shape}")
    print(f"Statistics: {ce_loss.get_statistics()}")
    
    # Test LabelSmoothingLoss
    print("\n--- LabelSmoothingLoss ---")
    ls_loss = LabelSmoothingLoss(smoothing=0.1)
    loss_value = ls_loss.forward(logits, targets, padding_mask)
    grad_logits = ls_loss.backward()
    
    print(f"Loss value: {loss_value:.4f}")
    print(f"Gradient shape: {grad_logits.shape}")
    print(f"Statistics: {ls_loss.get_statistics()}")
    
    print("\n✓ Loss functions tested successfully!")


if __name__ == "__main__":
    test_loss_functions()
