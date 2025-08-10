"""
Neural Networks Package

This package implements neural networks from scratch for educational purposes.
It includes all the core components needed to build and train neural networks.

Components:
- Layer: Dense/fully connected layers
- Activations: ReLU, Sigmoid, Tanh, Softmax
- Loss Functions: MSE, Cross-Entropy, Binary Cross-Entropy
- Optimizers: SGD, Adam, RMSprop
- Neuron: Individual neuron implementation

Example:
    from neural_nets import Layer, Activation_ReLU, Loss_MSE, Optimizer_SGD
    
    # Build a simple network
    network = [
        Layer(2, 4),
        Activation_ReLU(),
        Layer(4, 1)
    ]
"""

from .layer import Layer
from .activations import (
    Activation_ReLU,
    Activation_Sigmoid,
    Activation_Tanh,
    Activation_Softmax
)
from .loss import (
    Loss_MSE,
    Loss_CrossEntropy,
    Loss_BinaryCrossEntropy
)
from .optimizer import (
    Optimizer_SGD,
    Optimizer_Adam,
    Optimizer_RMSprop
)
from .neuron import Neuron

__version__ = "1.0.0"
__author__ = "Transformer Workshop Team"

__all__ = [
    'Layer',
    'Activation_ReLU',
    'Activation_Sigmoid',
    'Activation_Tanh',
    'Activation_Softmax',
    'Loss_MSE',
    'Loss_CrossEntropy',
    'Loss_BinaryCrossEntropy',
    'Optimizer_SGD',
    'Optimizer_Adam',
    'Optimizer_RMSprop',
    'Neuron'
]
