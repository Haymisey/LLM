"""
Model Checkpointing and Early Stopping
======================================

This module implements checkpointing and early stopping mechanisms
for training the Transformer model.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Optional


class ModelCheckpointer:
    """
    Manages model checkpointing during training.
    
    Saves and loads model states, optimizer states, and training metadata.
    """
    
    def __init__(self, save_dir: str, save_freq: int = 1, max_checkpoints: int = 5):
        """
        Initialize model checkpointer.
        
        Args:
            save_dir (str): Directory to save checkpoints
            save_freq (int): Save checkpoint every N epochs
            max_checkpoints (int): Maximum number of checkpoints to keep
        """
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.max_checkpoints = max_checkpoints
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Track saved checkpoints
        self.saved_checkpoints = []
        self.best_metric = float('-inf')
        self.best_checkpoint = None
    
    def save_checkpoint(self, epoch: int, model, optimizer, scheduler=None, 
                       metrics: Optional[Dict[str, Any]] = None, 
                       is_best: bool = False, **kwargs) -> str:
        """
        Save a training checkpoint.
        
        Args:
            epoch (int): Current epoch number
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Learning rate scheduler to save (optional)
            metrics (dict): Training/validation metrics (optional)
            is_best (bool): Whether this is the best checkpoint so far
            **kwargs: Additional data to save
        
        Returns:
            str: Path to saved checkpoint
        """
        checkpoint_data = {
            'epoch': epoch,
            'model_state': model.get_parameters(),
            'optimizer_state': self._get_optimizer_state(optimizer),
            'scheduler_state': self._get_scheduler_state(scheduler) if scheduler else None,
            'metrics': metrics or {},
            'is_best': is_best,
            'timestamp': self._get_timestamp(),
            **kwargs
        }
        
        # Create checkpoint filename
        if is_best:
            filename = f"best_model_epoch_{epoch:04d}.npz"
        else:
            filename = f"checkpoint_epoch_{epoch:04d}.npz"
        
        checkpoint_path = os.path.join(self.save_dir, filename)
        
        # Save checkpoint
        self._save_npz(checkpoint_path, checkpoint_data)
        
        # Track saved checkpoint
        self.saved_checkpoints.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'is_best': is_best,
            'metrics': metrics or {}
        })
        
        # Update best checkpoint if necessary
        if is_best:
            self.best_checkpoint = checkpoint_path
            if metrics and 'loss' in metrics:
                self.best_metric = -metrics['loss']  # Lower loss is better
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        print(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer, 
                       scheduler=None, **kwargs) -> Dict[str, Any]:
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Learning rate scheduler to load state into (optional)
            **kwargs: Additional data to load
        
        Returns:
            dict: Loaded checkpoint data
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint data
        checkpoint_data = self._load_npz(checkpoint_path)
        
        # Restore model state
        if 'model_state' in checkpoint_data:
            model.set_parameters(checkpoint_data['model_state'])
        
        # Restore optimizer state
        if 'optimizer_state' in checkpoint_data:
            self._restore_optimizer_state(optimizer, checkpoint_data['optimizer_state'])
        
        # Restore scheduler state
        if scheduler and 'scheduler_state' in checkpoint_data:
            self._restore_scheduler_state(scheduler, checkpoint_data['scheduler_state'])
        
        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"  Epoch: {checkpoint_data.get('epoch', 'Unknown')}")
        print(f"  Metrics: {checkpoint_data.get('metrics', {})}")
        
        return checkpoint_data
    
    def load_best_checkpoint(self, model, optimizer, scheduler=None, **kwargs) -> Dict[str, Any]:
        """
        Load the best checkpoint based on metrics.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Learning rate scheduler to load state into (optional)
            **kwargs: Additional data to load
        
        Returns:
            dict: Loaded checkpoint data
        """
        if not self.best_checkpoint:
            raise ValueError("No best checkpoint available")
        
        return self.load_checkpoint(self.best_checkpoint, model, optimizer, scheduler, **kwargs)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint."""
        if not self.saved_checkpoints:
            return None
        
        # Sort by epoch and return latest
        latest = max(self.saved_checkpoints, key=lambda x: x['epoch'])
        return latest['path']
    
    def resume_training(self, model, optimizer, scheduler=None, **kwargs) -> Dict[str, Any]:
        """
        Resume training from the latest checkpoint.
        
        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Learning rate scheduler to load state into (optional)
            **kwargs: Additional data to load
        
        Returns:
            dict: Loaded checkpoint data
        """
        latest_checkpoint = self.get_latest_checkpoint()
        if not latest_checkpoint:
            print("No checkpoint found to resume from")
            return {}
        
        return self.load_checkpoint(latest_checkpoint, model, optimizer, scheduler, **kwargs)
    
    def _get_optimizer_state(self, optimizer) -> Dict[str, Any]:
        """Extract optimizer state for saving."""
        if hasattr(optimizer, 'get_state'):
            return optimizer.get_state()
        
        # Default state extraction
        state = {
            'learning_rate': optimizer.get_learning_rate(),
            'step_count': getattr(optimizer, 'step_count', 0)
        }
        
        # Add optimizer-specific state
        if hasattr(optimizer, 'velocity'):
            state['velocity'] = optimizer.velocity
        if hasattr(optimizer, 'm'):
            state['m'] = optimizer.m
        if hasattr(optimizer, 'v'):
            state['v'] = optimizer.v
        
        return state
    
    def _restore_optimizer_state(self, optimizer, state: Dict[str, Any]):
        """Restore optimizer state from checkpoint."""
        if hasattr(optimizer, 'set_state'):
            optimizer.set_state(state)
            return
        
        # Default state restoration
        if 'learning_rate' in state:
            optimizer.set_learning_rate(state['learning_rate'])
        
        if 'step_count' in state:
            optimizer.step_count = state['step_count']
        
        # Restore optimizer-specific state
        if 'velocity' in state and hasattr(optimizer, 'velocity'):
            optimizer.velocity = state['velocity']
        if 'm' in state and hasattr(optimizer, 'm'):
            optimizer.m = state['m']
        if 'v' in state and hasattr(optimizer, 'v'):
            optimizer.v = state['v']
    
    def _get_scheduler_state(self, scheduler) -> Dict[str, Any]:
        """Extract scheduler state for saving."""
        if hasattr(scheduler, 'get_state'):
            return scheduler.get_state()
        
        return {
            'step_count': getattr(scheduler, 'step_count', 0),
            'initial_lr': getattr(scheduler, 'initial_lr', 0.0)
        }
    
    def _restore_scheduler_state(self, scheduler, state: Dict[str, Any]):
        """Restore scheduler state from checkpoint."""
        if hasattr(scheduler, 'set_state'):
            scheduler.set_state(state)
            return
        
        if 'step_count' in state:
            scheduler.step_count = state['step_count']
        if 'initial_lr' in state:
            scheduler.initial_lr = state['initial_lr']
    
    def _save_npz(self, filepath: str, data: Dict[str, Any]):
        """Save checkpoint data as NPZ file."""
        # Convert data to numpy arrays for saving
        save_data = {}
        for key, value in data.items():
            if value is not None:
                if isinstance(value, dict):
                    # Recursively save nested dictionaries
                    for nested_key, nested_value in value.items():
                        if nested_value is not None:
                            save_data[f"{key}_{nested_key}"] = nested_value
                else:
                    save_data[key] = value
        
        np.savez_compressed(filepath, **save_data)
    
    def _load_npz(self, filepath: str) -> Dict[str, Any]:
        """Load checkpoint data from NPZ file."""
        data = np.load(filepath, allow_pickle=True)
        
        # Reconstruct original data structure
        checkpoint_data = {}
        for key in data.keys():
            if '_' in key and key.count('_') > 1:
                # Handle nested dictionary keys
                parts = key.split('_', 1)
                main_key = parts[0]
                nested_key = parts[1]
                
                if main_key not in checkpoint_data:
                    checkpoint_data[main_key] = {}
                checkpoint_data[main_key][nested_key] = data[key].item()
            else:
                checkpoint_data[key] = data[key].item()
        
        return checkpoint_data
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        if len(self.saved_checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by epoch and keep only the most recent ones
        self.saved_checkpoints.sort(key=lambda x: x['epoch'])
        
        # Remove old checkpoints (but keep the best one)
        checkpoints_to_remove = self.saved_checkpoints[:-self.max_checkpoints]
        
        for checkpoint in checkpoints_to_remove:
            if not checkpoint['is_best']:  # Don't remove the best checkpoint
                try:
                    os.remove(checkpoint['path'])
                    self.saved_checkpoints.remove(checkpoint)
                    print(f"Removed old checkpoint: {checkpoint['path']}")
                except OSError:
                    pass
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about saved checkpoints."""
        return {
            'save_dir': self.save_dir,
            'total_checkpoints': len(self.saved_checkpoints),
            'best_checkpoint': self.best_checkpoint,
            'best_metric': self.best_metric,
            'checkpoints': [
                {
                    'path': cp['path'],
                    'epoch': cp['epoch'],
                    'is_best': cp['is_best'],
                    'metrics': cp['metrics']
                }
                for cp in self.saved_checkpoints
            ]
        }


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.
    
    Monitors validation metrics and stops training when no improvement is seen.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', monitor: str = 'loss'):
        """
        Initialize early stopping.
        
        Args:
            patience (int): Number of epochs to wait for improvement
            min_delta (float): Minimum change to qualify as improvement
            mode (str): 'min' for minimizing metrics, 'max' for maximizing
            monitor (str): Metric to monitor ('loss', 'accuracy', 'bleu', etc.)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        
        # State variables
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0
    
    def __call__(self, current_metric: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            current_metric (float): Current value of monitored metric
            epoch (int): Current epoch number
        
        Returns:
            bool: True if training should stop
        """
        if self.mode == 'min':
            improved = current_metric < (self.best_metric - self.min_delta)
        else:
            improved = current_metric > (self.best_metric + self.min_delta)
        
        if improved:
            self.best_metric = current_metric
            self.counter = 0
            self.best_epoch = epoch
            print(f"Early stopping: {self.monitor} improved to {current_metric:.4f}")
        else:
            self.counter += 1
            print(f"Early stopping: {self.monitor} did not improve for {self.counter} epochs")
        
        if self.counter >= self.patience:
            self.should_stop = True
            print(f"Early stopping triggered after {self.patience} epochs without improvement")
            print(f"Best {self.monitor}: {self.best_metric:.4f} at epoch {self.best_epoch}")
        
        return self.should_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.should_stop = False
    
    def get_best_metric(self) -> float:
        """Get the best metric value seen so far."""
        return self.best_metric
    
    def get_best_epoch(self) -> int:
        """Get the epoch with the best metric."""
        return self.best_epoch
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing."""
        return {
            'best_metric': self.best_metric,
            'counter': self.counter,
            'should_stop': self.should_stop,
            'best_epoch': self.best_epoch
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restore state from checkpoint."""
        self.best_metric = state.get('best_metric', self.best_metric)
        self.counter = state.get('counter', 0)
        self.should_stop = state.get('should_stop', False)
        self.best_epoch = state.get('best_epoch', 0)


def test_checkpointing():
    """Test the checkpointing and early stopping functionality."""
    print("Testing Checkpointing and Early Stopping...")
    
    # Mock model and optimizer for testing
    class MockModel:
        def __init__(self):
            self.parameters = {
                'weight': np.random.randn(10, 10) * 0.1,
                'bias': np.random.randn(10) * 0.1
            }
        
        def get_parameters(self):
            return self.parameters
        
        def set_parameters(self, params):
            self.parameters = params
    
    class MockOptimizer:
        def __init__(self):
            self.learning_rate = 0.001
            self.step_count = 0
        
        def get_learning_rate(self):
            return self.learning_rate
        
        def set_learning_rate(self, lr):
            self.learning_rate = lr
    
    # Test ModelCheckpointer
    print("\n--- ModelCheckpointer ---")
    checkpointer = ModelCheckpointer("test_checkpoints", save_freq=1, max_checkpoints=3)
    
    model = MockModel()
    optimizer = MockOptimizer()
    
    # Save some checkpoints
    for epoch in range(5):
        metrics = {'loss': 1.0 - epoch * 0.1, 'accuracy': 0.5 + epoch * 0.1}
        is_best = epoch == 4  # Last epoch is best
        
        checkpoint_path = checkpointer.save_checkpoint(
            epoch, model, optimizer, metrics=metrics, is_best=is_best
        )
    
    # Get checkpoint info
    info = checkpointer.get_checkpoint_info()
    print(f"Checkpoint info: {info}")
    
    # Test EarlyStopping
    print("\n--- EarlyStopping ---")
    early_stopping = EarlyStopping(patience=3, mode='min', monitor='loss')
    
    # Simulate training with improving then worsening loss
    losses = [1.0, 0.9, 0.8, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    
    for epoch, loss in enumerate(losses):
        should_stop = early_stopping(loss, epoch)
        print(f"Epoch {epoch}: Loss = {loss:.3f}, Should stop = {should_stop}")
        
        if should_stop:
            break
    
    print(f"Best loss: {early_stopping.get_best_metric():.3f} at epoch {early_stopping.get_best_epoch()}")
    
    # Clean up test files
    import shutil
    if os.path.exists("test_checkpoints"):
        shutil.rmtree("test_checkpoints")
    
    print("\n✓ Checkpointing and early stopping tested successfully!")


if __name__ == "__main__":
    test_checkpointing()
