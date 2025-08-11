# Checkpoint 1: Neural Networks & Backpropagation ✅

## Overview

Checkpoint 1 implements a complete neural network framework from scratch, including all the core components needed to build and train neural networks. This serves as the foundation for understanding how neural networks work before moving on to more complex architectures like RNNs, LSTMs, and Transformers.

## What's Implemented

### 🧠 Core Components

#### 1. **Layer Class** (`src/neural_nets/layer.py`)
- **Dense/Fully Connected Layers**: Implements the basic building block of neural networks
- **Forward Pass**: Computes weighted sum of inputs plus bias
- **Backward Pass**: Calculates gradients for weights, biases, and inputs
- **Proper Shape Handling**: Automatically handles matrix dimensions for batch processing

#### 2. **Activation Functions** (`src/neural_nets/activations.py`)
- **ReLU**: Rectified Linear Unit - most common activation for hidden layers
- **Sigmoid**: S-shaped function for binary classification outputs
- **Tanh**: Hyperbolic tangent for outputs in range [-1, 1]
- **Softmax**: Multi-class classification output normalization
- **All with Backpropagation**: Each activation includes proper gradient calculation

#### 3. **Loss Functions** (`src/neural_nets/loss.py`)
- **MSE (Mean Squared Error)**: For regression problems
- **Cross-Entropy**: For multi-class classification
- **Binary Cross-Entropy**: For binary classification
- **Proper Gradient Calculation**: All loss functions compute gradients for backpropagation

#### 4. **Optimizers** (`src/neural_nets/optimizer.py`)
- **SGD**: Stochastic Gradient Descent - basic but effective
- **Adam**: Adaptive learning rates with momentum (recommended for most cases)
- **RMSprop**: Adaptive learning rates without momentum
- **Parameter Updates**: All optimizers properly update weights and biases

#### 5. **Neuron Class** (`src/neural_nets/neuron.py`)
- **Individual Neuron**: Single neuron implementation for educational purposes
- **Forward & Backward Pass**: Complete gradient computation
- **Building Block**: Foundation for understanding how layers work

### 🚀 Training Examples

#### **XOR Problem** (`src/neural_nets/training_examples.py`)
- Classic non-linear classification problem
- Demonstrates the need for hidden layers
- Shows how neural networks can learn complex patterns

#### **Linear Regression**
- Simple regression problem: y = 2x + 1 + noise
- Single-layer network learns the linear relationship
- Demonstrates basic gradient descent

#### **Binary Classification**
- Two classes separated by a line (y = x)
- Multi-layer network with ReLU and Sigmoid activations
- Shows how neural networks can learn decision boundaries

#### **Multi-Class Classification**
- Three classes in concentric circles
- Demonstrates softmax activation and cross-entropy loss
- Shows how networks can learn complex spatial patterns

## How to Use

### 1. **Run Tests** (Recommended First Step)
```bash
cd tests
python test_neural_nets.py
```

This will verify that all components are working correctly.

### 2. **Run Training Examples**
```bash
cd src/neural_nets
python training_examples.py
```

This will train networks on all four example problems and show results.

### 3. **Build Your Own Network**
```python
from neural_nets import Layer, Activation_ReLU, Loss_MSE, Optimizer_Adam

# Create a simple network
network = [
    Layer(n_inputs=10, n_neurons=64),
    Activation_ReLU(),
    Layer(n_inputs=64, n_neurons=32),
    Activation_ReLU(),
    Layer(n_inputs=32, n_neurons=1)
]

# Define loss and optimizer
loss_function = Loss_MSE()
optimizer = Optimizer_Adam(learning_rate=0.01)

# Training loop
for epoch in range(1000):
    # Forward pass
    output = inputs
    for component in network:
        output = component.forward(output)
    
    # Calculate loss
    loss = loss_function.forward(output, targets)
    
    # Backward pass
    loss_function.backward(output, targets)
    dvalues = loss_function.dinputs
    
    for component in reversed(network):
        component.backward(dvalues)
        dvalues = component.dinputs
    
    # Update weights
    for component in network:
        if isinstance(component, Layer):
            optimizer.update_params(component)
```

## Key Concepts Demonstrated

### 🔄 **Forward Pass**
- Data flows through the network layer by layer
- Each layer applies weights, biases, and activation functions
- Output shape is determined by layer dimensions

### ⬅️ **Backward Pass (Backpropagation)**
- Gradients flow backward through the network
- Each layer computes gradients for its parameters
- Chain rule is applied to propagate gradients

### 📚 **Loss Functions**
- Measure how well the network is performing
- Provide gradients to guide weight updates
- Different loss functions for different problem types

### 🎯 **Optimization**
- Various algorithms to update network parameters
- Learning rate controls update step size
- Adaptive methods adjust learning rates automatically

## What You'll Learn

1. **Mathematical Foundations**: How neural networks compute outputs and gradients
2. **Implementation Details**: How to code each component from scratch
3. **Training Process**: Complete forward/backward pass implementation
4. **Problem Solving**: How to apply neural networks to different types of problems
5. **Debugging Skills**: Understanding what can go wrong and how to fix it

## Next Steps

After completing Checkpoint 1, you'll be ready for:
- **Checkpoint 2**: RNN/LSTM Implementation
- **Checkpoint 3**: Attention Mechanism
- **Checkpoint 4**: Transformer Core Components

## Troubleshooting

### Common Issues:
1. **Shape Mismatches**: Ensure input/output dimensions match between layers
2. **Gradient Explosion**: Use gradient clipping or lower learning rates
3. **Slow Convergence**: Try different optimizers or adjust learning rates
4. **Overfitting**: Add regularization or reduce network capacity

### Debugging Tips:
1. Print shapes at each layer during forward pass
2. Check that gradients are reasonable (not NaN or extremely large)
3. Verify loss is decreasing over time
4. Test with simple problems first (like XOR)

## Files Structure

```
src/neural_nets/
├── __init__.py              # Package initialization
├── layer.py                 # Dense layer implementation
├── activations.py           # Activation functions
├── loss.py                  # Loss functions
├── optimizer.py             # Optimization algorithms
├── neuron.py                # Individual neuron
├── network.py               # Main module exports
└── training_examples.py     # Training examples

tests/
└── test_neural_nets.py     # Comprehensive tests
```

## Success Criteria ✅

- [x] Basic feedforward neural network
- [x] Backpropagation implementation
- [x] Simple training examples
- [x] Multiple activation functions
- [x] Multiple loss functions
- [x] Multiple optimizers
- [x] Comprehensive testing
- [x] Educational examples
- [x] Proper documentation

**Checkpoint 1 is now complete!** 🎉

You have a fully functional neural network framework that you can use to understand the fundamentals before moving on to more complex architectures. This solid foundation will make understanding RNNs, LSTMs, and Transformers much easier.
