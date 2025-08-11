import numpy as np

class Activation_ReLU:
    """
    ReLU (Rectified Linear Unit) activation function.
    """
    def forward(self, inputs):
        """
        Calculates the ReLU activation and stores the input for backpropagation.
        """
        # Store the original inputs for use in the backward pass
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        """
        Calculates the gradient for the ReLU function.
        """
        # Create a copy of the incoming gradient
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

class Activation_Sigmoid:
    """
    Sigmoid activation function.
    """
    def forward(self, inputs):
        """
        Calculates the sigmoid activation and stores the input for backpropagation.
        """
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output
    
    def backward(self, dvalues):
        """
        Calculates the gradient for the sigmoid function.
        """
        self.dinputs = dvalues * self.output * (1 - self.output)

class Activation_Tanh:
    """
    Hyperbolic tangent activation function.
    """
    def forward(self, inputs):
        """
        Calculates the tanh activation and stores the input for backpropagation.
        """
        self.inputs = inputs
        self.output = np.tanh(inputs)
        return self.output
    
    def backward(self, dvalues):
        """
        Calculates the gradient for the tanh function.
        """
        self.dinputs = dvalues * (1 - self.output**2)

class Activation_Softmax:
    """
    Softmax activation function for multi-class classification.
    """
    def forward(self, inputs):
        """
        Calculates the softmax activation and stores the input for backpropagation.
        """
        self.inputs = inputs
        
        # Subtract the maximum value to prevent overflow
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        
        # Normalize
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output
    
    def backward(self, dvalues):
        """
        Calculates the gradient for the softmax function.
        """
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            
            # Calculate Jacobian matrix of the output
            jacobian_m = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            # Calculate sample-wise gradient
            self.dinputs[index] = np.dot(jacobian_m, single_dvalues)

# --- Verification ---
if __name__ == '__main__':
    print("=== Testing Activation Functions ===\n")
    
    # Test ReLU
    print("--- ReLU Test ---")
    relu = Activation_ReLU()
    test_input = np.array([[0.5, -0.2, 0.1], [-0.8, 0.4, -0.9]])
    relu_output = relu.forward(test_input)
    print("Input:", test_input)
    print("ReLU Output:", relu_output)
    
    # Test Sigmoid
    print("\n--- Sigmoid Test ---")
    sigmoid = Activation_Sigmoid()
    sigmoid_output = sigmoid.forward(test_input)
    print("Input:", test_input)
    print("Sigmoid Output:", sigmoid_output)
    
    # Test Tanh
    print("\n--- Tanh Test ---")
    tanh = Activation_Tanh()
    tanh_output = tanh.forward(test_input)
    print("Input:", test_input)
    print("Tanh Output:", tanh_output)
    
    # Test Softmax
    print("\n--- Softmax Test ---")
    softmax = Activation_Softmax()
    softmax_output = softmax.forward(test_input)
    print("Input:", test_input)
    print("Softmax Output:", softmax_output)
    print("Sum of softmax outputs:", np.sum(softmax_output, axis=1))