import sys
import os
import numpy as np

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neural_nets.layer import Layer
from neural_nets.activations import Activation_ReLU, Activation_Sigmoid, Activation_Tanh, Activation_Softmax
from neural_nets.loss import Loss_MSE, Loss_CrossEntropy, Loss_BinaryCrossEntropy
from neural_nets.optimizer import Optimizer_SGD, Optimizer_Adam, Optimizer_RMSprop
from neural_nets.neuron import Neuron

def test_layer():
    """Test the Layer class forward and backward passes."""
    print("Testing Layer class...")
    
    # Test forward pass
    layer = Layer(n_inputs=3, n_neurons=2)
    inputs = np.array([[1.0, 2.0, 3.0]])
    output = layer.forward(inputs)
    
    assert output.shape == (1, 2), f"Expected shape (1, 2), got {output.shape}"
    assert layer.inputs is not None, "Inputs not stored for backpropagation"
    assert layer.output is not None, "Output not stored"
    
    # Test backward pass
    dvalues = np.array([[0.1, 0.2]])
    layer.backward(dvalues)
    
    assert layer.dweights.shape == (3, 2), f"Expected dweights shape (3, 2), got {layer.dweights.shape}"
    assert layer.dbiases.shape == (1, 2), f"Expected dbiases shape (1, 2), got {layer.dbiases.shape}"
    assert layer.dinputs.shape == (1, 3), f"Expected dinputs shape (1, 3), got {layer.dinputs.shape}"
    
    print("✓ Layer class tests passed!")

def test_activations():
    """Test all activation functions."""
    print("Testing activation functions...")
    
    test_input = np.array([[0.5, -0.2, 0.1], [-0.8, 0.4, -0.9]])
    
    # Test ReLU
    relu = Activation_ReLU()
    relu_output = relu.forward(test_input)
    relu.backward(np.ones_like(test_input))
    
    assert np.all(relu_output >= 0), "ReLU should output non-negative values"
    assert relu.dinputs.shape == test_input.shape, "ReLU dinputs shape mismatch"
    
    # Test Sigmoid
    sigmoid = Activation_Sigmoid()
    sigmoid_output = sigmoid.forward(test_input)
    sigmoid.backward(np.ones_like(test_input))
    
    assert np.all(sigmoid_output >= 0) and np.all(sigmoid_output <= 1), "Sigmoid should output values in [0,1]"
    assert sigmoid.dinputs.shape == test_input.shape, "Sigmoid dinputs shape mismatch"
    
    # Test Tanh
    tanh = Activation_Tanh()
    tanh_output = tanh.forward(test_input)
    tanh.backward(np.ones_like(test_input))
    
    assert np.all(tanh_output >= -1) and np.all(tanh_output <= 1), "Tanh should output values in [-1,1]"
    assert tanh.dinputs.shape == test_input.shape, "Tanh dinputs shape mismatch"
    
    # Test Softmax
    softmax = Activation_Softmax()
    softmax_output = softmax.forward(test_input)
    softmax.backward(np.ones_like(test_input))
    
    assert np.allclose(np.sum(softmax_output, axis=1), 1.0), "Softmax outputs should sum to 1"
    assert softmax.dinputs.shape == test_input.shape, "Softmax dinputs shape mismatch"
    
    print("✓ Activation functions tests passed!")

def test_loss_functions():
    """Test all loss functions."""
    print("Testing loss functions...")
    
    # Test MSE
    mse = Loss_MSE()
    y_pred = np.array([[0.1, 0.8]])
    y_true = np.array([[0.0, 1.0]])
    
    loss = mse.forward(y_pred, y_true)
    mse.backward(y_pred, y_true)
    
    assert loss > 0, "MSE loss should be positive"
    assert mse.dinputs.shape == y_pred.shape, "MSE dinputs shape mismatch"
    
    # Test Cross-Entropy
    ce = Loss_CrossEntropy()
    y_pred_ce = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    y_true_ce = np.array([1, 0])
    
    ce_loss = ce.forward(y_pred_ce, y_true_ce)
    ce.backward(y_pred_ce, y_true_ce)
    
    assert ce_loss > 0, "Cross-entropy loss should be positive"
    assert ce.dinputs.shape == y_pred_ce.shape, "Cross-entropy dinputs shape mismatch"
    
    # Test Binary Cross-Entropy
    bce = Loss_BinaryCrossEntropy()
    y_pred_bce = np.array([[0.1], [0.8], [0.3]])
    y_true_bce = np.array([[0], [1], [0]])
    
    bce_loss = bce.forward(y_pred_bce, y_true_bce)
    bce.backward(y_pred_bce, y_true_bce)
    
    assert bce_loss > 0, "Binary cross-entropy loss should be positive"
    assert bce.dinputs.shape == y_pred_bce.shape, "Binary cross-entropy dinputs shape mismatch"
    
    print("✓ Loss functions tests passed!")

def test_optimizers():
    """Test all optimizers."""
    print("Testing optimizers...")
    
    # Create a mock layer for testing
    class MockLayer:
        def __init__(self):
            self.weights = np.array([[0.1, 0.2], [0.3, 0.4]])
            self.biases = np.array([[0.01, 0.02]])
            self.dweights = np.array([[0.01, 0.02], [0.03, 0.04]])
            self.dbiases = np.array([[0.001, 0.002]])
    
    layer = MockLayer()
    original_weights = layer.weights.copy()
    original_biases = layer.biases.copy()
    
    # Test SGD
    sgd = Optimizer_SGD(learning_rate=0.1)
    sgd.update_params(layer)
    
    assert not np.array_equal(layer.weights, original_weights), "SGD should update weights"
    assert not np.array_equal(layer.biases, original_biases), "SGD should update biases"
    
    # Reset for next test
    layer.weights = original_weights.copy()
    layer.biases = original_biases.copy()
    
    # Test Adam
    adam = Optimizer_Adam(learning_rate=0.1)
    adam.update_params(layer)
    
    assert not np.array_equal(layer.weights, original_weights), "Adam should update weights"
    assert not np.array_equal(layer.biases, original_biases), "Adam should update biases"
    
    # Reset for next test
    layer.weights = original_weights.copy()
    layer.biases = original_biases.copy()
    
    # Test RMSprop
    rmsprop = Optimizer_RMSprop(learning_rate=0.1)
    rmsprop.update_params(layer)
    
    assert not np.array_equal(layer.weights, original_weights), "RMSprop should update weights"
    assert not np.array_equal(layer.biases, original_biases), "RMSprop should update biases"
    
    print("✓ Optimizers tests passed!")

def test_neuron():
    """Test the Neuron class."""
    print("Testing Neuron class...")
    
    neuron = Neuron(n_inputs=3)
    inputs = np.array([1.0, 2.0, 3.0])
    
    # Test forward pass
    output = neuron.forward(inputs)
    assert neuron.inputs is not None, "Inputs not stored for backpropagation"
    assert neuron.output is not None, "Output not stored"
    
    # Test backward pass
    dvalues = 0.5
    neuron.backward(dvalues)
    
    assert neuron.dweights.shape == (3,), f"Expected dweights shape (3,), got {neuron.dweights.shape}"
    assert neuron.dbias is not None, "dbias not calculated"
    assert neuron.dinputs.shape == (3,), f"Expected dinputs shape (3,), got {neuron.dinputs.shape}"
    
    print("✓ Neuron class tests passed!")

def test_simple_network():
    """Test a simple neural network with forward and backward passes."""
    print("Testing simple neural network...")
    
    # Create a simple network: 2 inputs -> 3 hidden -> 1 output
    network = [
        Layer(n_inputs=2, n_neurons=3),
        Activation_ReLU(),
        Layer(n_inputs=3, n_neurons=1)
    ]
    
    # Test forward pass
    inputs = np.array([[1.0, 2.0]])
    output = inputs
    for component in network:
        output = component.forward(output)
    
    assert output.shape == (1, 1), f"Expected output shape (1, 1), got {output.shape}"
    
    # Test backward pass
    loss_function = Loss_MSE()
    y_true = np.array([[0.5]])
    
    loss = loss_function.forward(output, y_true)
    loss_function.backward(output, y_true)
    dvalues = loss_function.dinputs
    
    for component in reversed(network):
        component.backward(dvalues)
        dvalues = component.dinputs
    
    # Check that gradients were calculated
    for component in network:
        if isinstance(component, Layer):
            assert hasattr(component, 'dweights'), "Layer missing dweights"
            assert hasattr(component, 'dbiases'), "Layer missing dbiases"
            assert hasattr(component, 'dinputs'), "Layer missing dinputs"
    
    print("✓ Simple network tests passed!")

def run_all_tests():
    """Run all tests."""
    print("Running Neural Network Tests")
    print("=" * 40)
    
    try:
        test_layer()
        test_activations()
        test_loss_functions()
        test_optimizers()
        test_neuron()
        test_simple_network()
        
        print("\n" + "=" * 40)
        print("🎉 All tests passed! Checkpoint 1 is complete.")
        print("=" * 40)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    run_all_tests()
