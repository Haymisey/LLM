"""
Neural Network Module

This module provides the core components for building and training neural networks.
It includes layers, activations, loss functions, and optimizers.

Example usage:
    from neural_nets.layer import Layer
    from neural_nets.activations import Activation_ReLU
    from neural_nets.loss import Loss_MSE
    from neural_nets.optimizer import Optimizer_SGD
    
    # Create a simple network
    network = [
        Layer(n_inputs=2, n_neurons=4),
        Activation_ReLU(),
        Layer(n_inputs=4, n_neurons=1)
    ]
    
    # Define loss and optimizer
    loss_function = Loss_MSE()
    optimizer = Optimizer_SGD(learning_rate=0.1)
    
    # Training loop would go here...
"""

# Import all the main components
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

# Version information
__version__ = "1.0.0"
__author__ = "Transformer Workshop Team"

# Export all classes
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