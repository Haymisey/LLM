"""
Training Loop Implementation Module
==================================

This module implements the complete training pipeline for the Transformer model:
- Loss functions and metrics
- Training and validation loops
- Gradient computation and optimization
- Model checkpointing and early stopping
- Learning rate scheduling
- Training progress monitoring
"""

from .loss_functions import CrossEntropyLoss, LabelSmoothingLoss
from .metrics import TranslationMetrics, TrainingMetrics
from .optimizers import AdamOptimizer, SGDOptimizer, LearningRateScheduler
from .training_loop import TrainingLoop, ValidationLoop
from .checkpointing import ModelCheckpointer, EarlyStopping
from .progress_monitor import ProgressMonitor, TrainingLogger

__all__ = [
    'CrossEntropyLoss',
    'LabelSmoothingLoss',
    'TranslationMetrics',
    'TrainingMetrics',
    'AdamOptimizer',
    'SGDOptimizer',
    'LearningRateScheduler',
    'TrainingLoop',
    'ValidationLoop',
    'ModelCheckpointer',
    'EarlyStopping',
    'ProgressMonitor',
    'TrainingLogger'
]
