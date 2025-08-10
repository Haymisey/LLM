import numpy as np

class Neuron:
    """
    Represents a single neuron in a neural network.
    """
    def __init__(self, n_inputs):
        """
        Initializes the neuron.

        Args:
            n_inputs (int): The number of inputs the neuron will receive.
        """
        # Initialize weights with small random numbers.
        # Shape will be (n_inputs, 1)
        self.weights = 0.01 * np.random.randn(n_inputs)

        # Initialize bias to zero.
        self.bias = 0

    def forward(self, inputs):
        """
        Performs the forward pass calculation of the neuron.
        output = sum(inputs * weights) + bias

        Args:
            inputs (np.array): The input values. Should be a 1D array of shape (n_inputs,).

        Returns:
            float: The output of the neuron's calculation (before activation).
        """
        # Store inputs for backpropagation
        self.inputs = inputs
        
        # Calculate the weighted sum of inputs and add the bias.
        # np.dot is the dot product (e.g., x1*w1 + x2*w2 + ...)
        self.output = np.dot(self.weights, inputs) + self.bias
        return self.output
    
    def backward(self, dvalues):
        """
        Performs the backward pass to calculate gradients.
        
        Args:
            dvalues (float): Gradient from the next layer.
        """
        # Gradient on weights
        self.dweights = dvalues * self.inputs
        
        # Gradient on bias
        self.dbias = dvalues
        
        # Gradient on inputs
        self.dinputs = dvalues * self.weights

# --- Verification (let's test our neuron) ---
if __name__ == '__main__':
    # A neuron that accepts 3 inputs
    n_inputs = 3
    neuron = Neuron(n_inputs)

    # Some example inputs
    inputs = np.array([1.0, 2.0, 3.0])

    # Calculate the output
    output = neuron.forward(inputs)

    print("Neuron created with:")
    print("Weights:", neuron.weights)
    print("Bias:", neuron.bias)
    print("\nInputs to the neuron:", inputs)
    print("\nOutput from the neuron (forward pass):", output)
    
    # Test backward pass
    dvalues = 0.5
    neuron.backward(dvalues)
    
    print("\n--- Backward Pass Test ---")
    print("Incoming gradient (dvalues):", dvalues)
    print("Gradient on weights:", neuron.dweights)
    print("Gradient on bias:", neuron.dbias)
    print("Gradient on inputs:", neuron.dinputs)