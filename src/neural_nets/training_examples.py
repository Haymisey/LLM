import numpy as np
from layer import Layer
from activations import Activation_ReLU, Activation_Sigmoid, Activation_Tanh
from loss import Loss_MSE, Loss_CrossEntropy, Loss_BinaryCrossEntropy
from optimizer import Optimizer_SGD, Optimizer_Adam, Optimizer_RMSprop

def train_xor_problem():
    """
    Train a neural network to solve the XOR problem.
    """
    print("=== Training XOR Problem ===")
    
    # XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Network architecture
    network = [
        Layer(n_inputs=2, n_neurons=4),
        Activation_ReLU(),
        Layer(n_inputs=4, n_neurons=1),
        Activation_Sigmoid()
    ]
    
    # Loss and optimizer
    loss_function = Loss_MSE()
    optimizer = Optimizer_Adam(learning_rate=0.01)
    
    # Training loop
    epochs = 2000
    for epoch in range(epochs):
        total_loss = 0
        for x_sample, y_true_sample in zip(X, y):
            # Reshape inputs
            x_sample = np.reshape(x_sample, (1, -1))
            y_true_sample = np.reshape(y_true_sample, (1, -1))
            
            # Forward pass
            output = x_sample
            for component in network:
                output = component.forward(output)
            
            # Calculate loss
            total_loss += loss_function.forward(output, y_true_sample)
            
            # Backward pass
            loss_function.backward(output, y_true_sample)
            dvalues = loss_function.dinputs
            
            for component in reversed(network):
                component.backward(dvalues)
                dvalues = component.dinputs
            
            # Update weights
            for component in network:
                if isinstance(component, Layer):
                    optimizer.update_params(component)
        
        if epoch % 500 == 0:
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    # Test the trained network
    print("\n--- XOR Results ---")
    for x_sample, y_true_sample in zip(X, y):
        output = x_sample.reshape(1, -1)
        for component in network:
            output = component.forward(output)
        predicted = output.flatten()[0]
        true_val = y_true_sample.flatten()[0]
        print(f"Input: {x_sample} -> Predicted: {predicted:.4f} (True: {true_val})")

def train_linear_regression():
    """
    Train a neural network for linear regression.
    """
    print("\n=== Training Linear Regression ===")
    
    # Generate synthetic data: y = 2x + 1 + noise
    np.random.seed(42)
    X = np.random.rand(100, 1) * 10
    y = 2 * X + 1 + np.random.randn(100, 1) * 0.5
    
    # Network architecture (simple linear regression)
    network = [
        Layer(n_inputs=1, n_neurons=1)
    ]
    
    # Loss and optimizer
    loss_function = Loss_MSE()
    optimizer = Optimizer_SGD(learning_rate=0.01)
    
    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        total_loss = 0
        for x_sample, y_true_sample in zip(X, y):
            # Reshape inputs
            x_sample = np.reshape(x_sample, (1, -1))
            y_true_sample = np.reshape(y_true_sample, (1, -1))
            
            # Forward pass
            output = x_sample
            for component in network:
                output = component.forward(output)
            
            # Calculate loss
            total_loss += loss_function.forward(output, y_true_sample)
            
            # Backward pass
            loss_function.backward(output, y_true_sample)
            dvalues = loss_function.dinputs
            
            for component in reversed(network):
                component.backward(dvalues)
                dvalues = component.dinputs
            
            # Update weights
            for component in network:
                if isinstance(component, Layer):
                    optimizer.update_params(component)
        
        if epoch % 200 == 0:
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    # Test the trained network
    print("\n--- Linear Regression Results ---")
    test_x = np.array([[0], [5], [10]])
    for x_sample in test_x:
        output = x_sample.reshape(1, -1)
        for component in network:
            output = component.forward(output)
        predicted = output.flatten()[0]
        expected = 2 * x_sample.flatten()[0] + 1
        print(f"Input: {x_sample.flatten()[0]} -> Predicted: {predicted:.4f} (Expected: {expected:.4f})")

def train_binary_classification():
    """
    Train a neural network for binary classification.
    """
    print("\n=== Training Binary Classification ===")
    
    # Generate synthetic data: two classes separated by a line
    np.random.seed(42)
    n_samples = 200
    
    # Class 0: points below y = x
    class0_x = np.random.rand(n_samples//2, 2) * 2
    class0_y = class0_x[:, 1] < class0_x[:, 0]
    
    # Class 1: points above y = x
    class1_x = np.random.rand(n_samples//2, 2) * 2
    class1_y = class1_x[:, 1] >= class1_x[:, 0]
    
    # Combine data
    X = np.vstack([class0_x, class1_x])
    y = np.vstack([np.zeros((n_samples//2, 1)), np.ones((n_samples//2, 1))])
    
    # Network architecture
    network = [
        Layer(n_inputs=2, n_neurons=8),
        Activation_ReLU(),
        Layer(n_inputs=8, n_neurons=4),
        Activation_ReLU(),
        Layer(n_inputs=4, n_neurons=1),
        Activation_Sigmoid()
    ]
    
    # Loss and optimizer
    loss_function = Loss_BinaryCrossEntropy()
    optimizer = Optimizer_Adam(learning_rate=0.01)
    
    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        total_loss = 0
        for x_sample, y_true_sample in zip(X, y):
            # Reshape inputs
            x_sample = np.reshape(x_sample, (1, -1))
            y_true_sample = np.reshape(y_true_sample, (1, -1))
            
            # Forward pass
            output = x_sample
            for component in network:
                output = component.forward(output)
            
            # Calculate loss
            total_loss += loss_function.forward(output, y_true_sample)
            
            # Backward pass
            loss_function.backward(output, y_true_sample)
            dvalues = loss_function.dinputs
            
            for component in reversed(network):
                component.backward(dvalues)
                dvalues = component.dinputs
            
            # Update weights
            for component in network:
                if isinstance(component, Layer):
                    optimizer.update_params(component)
        
        if epoch % 200 == 0:
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    # Test the trained network
    print("\n--- Binary Classification Results ---")
    test_points = np.array([[0.5, 0.3], [0.5, 0.7], [1.5, 0.5], [1.5, 1.5]])
    for x_sample in test_points:
        output = x_sample.reshape(1, -1)
        for component in network:
            output = component.forward(output)
        predicted = output.flatten()[0]
        predicted_class = 1 if predicted > 0.5 else 0
        expected_class = 1 if x_sample[1] >= x_sample[0] else 0
        print(f"Input: {x_sample} -> Predicted: {predicted:.4f} (Class: {predicted_class}, Expected: {expected_class})")

def train_multi_class_classification():
    """
    Train a neural network for multi-class classification.
    """
    print("\n=== Training Multi-Class Classification ===")
    
    # Generate synthetic data: three classes in a circle pattern
    np.random.seed(42)
    n_samples = 300
    
    # Class 0: center
    class0_x = np.random.randn(n_samples//3, 2) * 0.5
    
    # Class 1: outer ring
    angles = np.random.rand(n_samples//3) * 2 * np.pi
    radii = 2 + np.random.randn(n_samples//3) * 0.3
    class1_x = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Class 2: middle ring
    angles = np.random.rand(n_samples//3) * 2 * np.pi
    radii = 1 + np.random.randn(n_samples//3) * 0.3
    class2_x = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    
    # Combine data
    X = np.vstack([class0_x, class1_x, class2_x])
    y = np.array([0] * (n_samples//3) + [1] * (n_samples//3) + [2] * (n_samples//3))
    
    # Network architecture
    network = [
        Layer(n_inputs=2, n_neurons=16),
        Activation_ReLU(),
        Layer(n_inputs=16, n_neurons=8),
        Activation_ReLU(),
        Layer(n_inputs=8, n_neurons=3)
    ]
    
    # Loss and optimizer
    loss_function = Loss_CrossEntropy()
    optimizer = Optimizer_Adam(learning_rate=0.01)
    
    # Training loop
    epochs = 1000
    for epoch in range(epochs):
        total_loss = 0
        for x_sample, y_true_sample in zip(X, y):
            # Reshape inputs
            x_sample = np.reshape(x_sample, (1, -1))
            
            # Forward pass
            output = x_sample
            for component in network:
                output = component.forward(output)
            
            # Calculate loss
            total_loss += loss_function.forward(output, np.array([y_true_sample]))
            
            # Backward pass
            loss_function.backward(output, np.array([y_true_sample]))
            dvalues = loss_function.dinputs
            
            for component in reversed(network):
                component.backward(dvalues)
                dvalues = component.dinputs
            
            # Update weights
            for component in network:
                if isinstance(component, Layer):
                    optimizer.update_params(component)
        
        if epoch % 200 == 0:
            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    # Test the trained network
    print("\n--- Multi-Class Classification Results ---")
    test_points = np.array([[0, 0], [2, 0], [1, 1], [0, 2]])
    for x_sample in test_points:
        output = x_sample.reshape(1, -1)
        for component in network:
            output = component.forward(output)
        predicted_class = np.argmax(output)
        print(f"Input: {x_sample} -> Predicted Class: {predicted_class}")

if __name__ == "__main__":
    print("Neural Network Training Examples")
    print("=" * 50)
    
    # Train all examples
    train_xor_problem()
    train_linear_regression()
    train_binary_classification()
    train_multi_class_classification()
    
    print("\n" + "=" * 50)
    print("All training examples completed!")
