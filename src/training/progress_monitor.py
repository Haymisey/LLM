"""
Training Progress Monitoring and Logging
========================================

This module implements progress monitoring and logging for training.
"""

import time
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime


class ProgressMonitor:
    """
    Monitors and displays training progress in real-time.
    
    Provides progress bars, statistics, and performance metrics.
    """
    
    def __init__(self, total_epochs: int, log_interval: int = 10):
        """
        Initialize progress monitor.
        
        Args:
            total_epochs (int): Total number of epochs to train
            log_interval (int): Log progress every N batches
        """
        self.total_epochs = total_epochs
        self.log_interval = log_interval
        
        # Training state
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.start_time = None
        
        # Performance metrics
        self.epoch_times = []
        self.batch_times = []
        self.loss_history = []
        self.learning_rates = []
        
        # Progress tracking
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.best_bleu = 0.0
    
    def start_training(self):
        """Start training timer."""
        self.start_time = time.time()
        print(f"\n🚀 Starting Training - {self.total_epochs} epochs")
        print("=" * 60)
    
    def start_epoch(self, epoch: int, num_batches: int):
        """Start a new epoch."""
        self.current_epoch = epoch
        self.current_batch = 0
        self.total_batches = num_batches
        self.epoch_start_time = time.time()
        
        print(f"\n📚 Epoch {epoch}/{self.total_epochs}")
        print(f"   Batches: {num_batches}")
        print("-" * 40)
    
    def update_batch(self, batch_idx: int, loss: float, learning_rate: float, 
                    accuracy: Optional[float] = None, bleu: Optional[float] = None):
        """Update batch progress."""
        self.current_batch = batch_idx + 1
        
        # Store metrics
        self.loss_history.append(loss)
        self.learning_rates.append(learning_rate)
        
        # Update best metrics
        if loss < self.best_loss:
            self.best_loss = loss
        if accuracy and accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
        if bleu and bleu > self.best_bleu:
            self.best_bleu = bleu
        
        # Log progress
        if batch_idx % self.log_interval == 0 or batch_idx == self.total_batches - 1:
            self._log_batch_progress(batch_idx, loss, learning_rate, accuracy, bleu)
    
    def end_epoch(self, epoch_loss: float, epoch_metrics: Optional[Dict[str, Any]] = None):
        """End current epoch."""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # Calculate epoch statistics
        avg_loss = sum(self.loss_history[-self.total_batches:]) / self.total_batches
        avg_lr = sum(self.learning_rates[-self.total_batches:]) / self.total_batches
        
        print(f"\n📊 Epoch {self.current_epoch} Summary")
        print(f"   Average Loss: {avg_loss:.4f}")
        print(f"   Best Loss: {self.best_loss:.4f}")
        if epoch_metrics:
            print(f"   Accuracy: {epoch_metrics.get('accuracy', 0.0):.4f}")
            print(f"   BLEU: {epoch_metrics.get('bleu', 0.0):.4f}")
        print(f"   Learning Rate: {avg_lr:.6f}")
        print(f"   Time: {epoch_time:.2f}s")
        print("-" * 40)
    
    def end_training(self):
        """End training and display final summary."""
        total_time = time.time() - self.start_time
        
        print(f"\n🎉 Training Complete!")
        print("=" * 60)
        print(f"Total Time: {total_time:.2f}s")
        import numpy as np
        print(f"Average Epoch Time: {np.mean(self.epoch_times):.2f}s")
        print(f"Best Loss: {self.best_loss:.4f}")
        print(f"Best Accuracy: {self.best_accuracy:.4f}")
        print(f"Best BLEU: {self.best_bleu:.4f}")
        print("=" * 60)
    
    def _log_batch_progress(self, batch_idx: int, loss: float, learning_rate: float,
                           accuracy: Optional[float], bleu: Optional[float]):
        """Log batch progress."""
        progress = (batch_idx + 1) / self.total_batches
        bar_length = 30
        filled_length = int(bar_length * progress)
        
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        log_str = f"  [{bar}] {progress*100:5.1f}% | "
        log_str += f"Batch {batch_idx+1:3d}/{self.total_batches:3d} | "
        log_str += f"Loss: {loss:.4f} | "
        log_str += f"LR: {learning_rate:.6f}"
        
        if accuracy is not None:
            log_str += f" | Acc: {accuracy:.4f}"
        if bleu is not None:
            log_str += f" | BLEU: {bleu:.4f}"
        
        print(log_str)
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        return {
            'total_epochs': self.total_epochs,
            'current_epoch': self.current_epoch,
            'total_batches': self.total_batches,
            'current_batch': self.current_batch,
            'best_loss': self.best_loss,
            'best_accuracy': self.best_accuracy,
            'best_bleu': self.best_bleu,
            'epoch_times': self.epoch_times,
            'loss_history': self.loss_history,
            'learning_rates': self.learning_rates
        }
    
    def reset(self):
        """Reset progress monitor."""
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.start_time = None
        self.epoch_times = []
        self.batch_times = []
        self.loss_history = []
        self.learning_rates = []
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.best_bleu = 0.0


class TrainingLogger:
    """
    Logs training progress and results to files.
    
    Supports JSON logging and experiment tracking.
    """
    
    def __init__(self, log_dir: str, experiment_name: str = "transformer_training"):
        """
        Initialize training logger.
        
        Args:
            log_dir (str): Directory to save logs
            experiment_name (str): Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.experiment_id = self._generate_experiment_id()
        
        # Create log directory
        self.experiment_dir = os.path.join(log_dir, self.experiment_id)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Log files
        self.training_log_path = os.path.join(self.experiment_dir, "training_log.json")
        self.config_path = os.path.join(self.experiment_dir, "config.json")
        self.metrics_path = os.path.join(self.experiment_dir, "metrics.json")
        
        # Initialize logs
        self.training_log = []
        self.metrics_log = []
        self.config = {}
    
    def log_config(self, config: Dict[str, Any]):
        """Log training configuration."""
        self.config = config
        self.config['experiment_id'] = self.experiment_id
        self.config['experiment_name'] = self.experiment_name
        self.config['start_time'] = datetime.now().isoformat()
        
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"📝 Configuration logged to: {self.config_path}")
    
    def log_training_step(self, epoch: int, batch: int, loss: float, 
                         learning_rate: float, **kwargs):
        """Log a training step."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'batch': batch,
            'loss': loss,
            'learning_rate': learning_rate,
            **kwargs
        }
        
        self.training_log.append(log_entry)
        
        # Save to file periodically
        if len(self.training_log) % 100 == 0:
            self._save_training_log()
    
    def log_epoch(self, epoch: int, epoch_stats: Dict[str, Any]):
        """Log epoch statistics."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'type': 'epoch_summary',
            **epoch_stats
        }
        
        self.training_log.append(log_entry)
        self._save_training_log()
    
    def log_validation(self, epoch: int, validation_results: Dict[str, Any]):
        """Log validation results."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'type': 'validation',
            **validation_results
        }
        
        self.training_log.append(log_entry)
        self._save_training_log()
    
    def log_metrics(self, metrics: Dict[str, Any]):
        """Log general metrics."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        
        self.metrics_log.append(log_entry)
        
        # Save metrics
        with open(self.metrics_path, 'w') as f:
            json.dump(self.metrics_log, f, indent=2)
    
    def log_sample_translation(self, epoch: int, source: str, target: str, 
                              generated: str, metrics: Optional[Dict[str, Any]] = None):
        """Log sample translation for qualitative evaluation."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'epoch': epoch,
            'type': 'sample_translation',
            'source': source,
            'target': target,
            'generated': generated,
            'metrics': metrics or {}
        }
        
        self.training_log.append(log_entry)
        self._save_training_log()
    
    def get_experiment_info(self) -> Dict[str, Any]:
        """Get experiment information."""
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'log_dir': self.experiment_dir,
            'config_path': self.config_path,
            'training_log_path': self.training_log_path,
            'metrics_path': self.metrics_path,
            'total_log_entries': len(self.training_log),
            'total_metrics': len(self.metrics_log)
        }
    
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.experiment_name}_{timestamp}"
    
    def _save_training_log(self):
        """Save training log to file."""
        with open(self.training_log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
    
    def load_logs(self) -> Dict[str, Any]:
        """Load all logs from files."""
        logs = {}
        
        # Load training log
        if os.path.exists(self.training_log_path):
            with open(self.training_log_path, 'r') as f:
                logs['training'] = json.load(f)
        
        # Load metrics
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                logs['metrics'] = json.load(f)
        
        # Load config
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                logs['config'] = json.load(f)
        
        return logs
    
    def export_summary(self, output_path: str):
        """Export training summary to a file."""
        summary = {
            'experiment_info': self.get_experiment_info(),
            'config': self.config,
            'training_summary': {
                'total_epochs': max([entry.get('epoch', 0) for entry in self.training_log]),
                'total_steps': len([entry for entry in self.training_log if entry.get('type') != 'epoch_summary']),
                'final_loss': self.training_log[-1].get('loss') if self.training_log else None,
                'best_loss': min([entry.get('loss', float('inf')) for entry in self.training_log if 'loss' in entry])
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Training summary exported to: {output_path}")


def test_progress_monitor():
    """Test the progress monitoring and logging functionality."""
    print("Testing Progress Monitor and Logger...")
    
    # Test ProgressMonitor
    print("\n--- ProgressMonitor ---")
    monitor = ProgressMonitor(total_epochs=3, log_interval=2)
    
    monitor.start_training()
    
    for epoch in range(3):
        monitor.start_epoch(epoch, 5)
        
        for batch in range(5):
            loss = 1.0 - epoch * 0.2 - batch * 0.02
            lr = 0.001 * (0.9 ** epoch)
            accuracy = 0.5 + epoch * 0.1 + batch * 0.02
            
            monitor.update_batch(batch, loss, lr, accuracy)
            time.sleep(0.1)  # Simulate training time
        
        monitor.end_epoch(loss, {'accuracy': accuracy})
    
    monitor.end_training()
    
    # Test TrainingLogger
    print("\n--- TrainingLogger ---")
    logger = TrainingLogger("test_logs", "test_experiment")
    
    # Log configuration
    config = {
        'model_type': 'transformer',
        'vocab_size': 1000,
        'd_model': 512,
        'n_heads': 8
    }
    logger.log_config(config)
    
    # Log some training steps
    for step in range(5):
        logger.log_training_step(0, step, 1.0 - step * 0.1, 0.001)
    
    # Log epoch
    logger.log_epoch(0, {'loss': 0.5, 'accuracy': 0.8})
    
    # Log validation
    logger.log_validation(0, {'val_loss': 0.6, 'val_accuracy': 0.75})
    
    # Get experiment info
    info = logger.get_experiment_info()
    print(f"Experiment info: {info}")
    
    # Export summary
    logger.export_summary("test_summary.json")
    
    # Clean up test files
    import shutil
    if os.path.exists("test_logs"):
        shutil.rmtree("test_logs")
    if os.path.exists("test_summary.json"):
        os.remove("test_summary.json")
    
    print("\n✓ Progress monitor and logger tested successfully!")


if __name__ == "__main__":
    import numpy as np
    test_progress_monitor()
