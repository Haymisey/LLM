import numpy as np

class Layer:
    """
    Represents a layer of neurons in a neural network.
    """
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        Performs the forward pass and stores the inputs for backpropagation.
        """
        # Store inputs for use in the backward pass
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        """
        Performs the backward pass to calculate gradients.
        """
        # Gradient on parameters (weights and biases)
        # We transpose self.inputs to align shapes for matrix multiplication
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # Gradient on values to be passed to the previous layer
        # We transpose self.weights for shape alignment
        self.dinputs = np.dot(dvalues, self.weights.T)

# --- Verification (let's test our layer's backward pass) ---
if __name__ == '__main__':
    # Create a layer
    # Use a fixed seed for random numbers to make the output predictable
    np.random.seed(0)
    layer = Layer(n_inputs=3, n_neurons=4)

    # Create some example inputs
    inputs = np.array([[1.0, 2.0, 3.0]]) # A batch with one sample

    # --- Forward Pass ---
    layer.forward(inputs)

    # --- Backward Pass ---
    # An example incoming gradient (dvalues) from a later layer
    dvalues = np.array([[0.1, 0.2, 0.3, 0.4]])

    layer.backward(dvalues)

    print("--- Layer Backward Pass Test ---")
    print("Inputs (from forward pass):\n", layer.inputs)
    print("\nWeights:\n", layer.weights)
    print("\nIncoming Gradient (dvalues):\n", dvalues)
    
    print("\n--- Calculated Gradients ---")
    print("dweights (gradient for weights):\n", layer.dweights)
    print("\ndbiases (gradient for biases):\n", layer.dbiases)
    print("\ndinputs (gradient to pass back):\n", layer.dinputs)