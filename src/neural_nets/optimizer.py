import numpy as np

class Optimizer_SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer.
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update_params(self, layer):
        """
        Updates the parameters of a given layer.
        """
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases

class Optimizer_Adam:
    """
    Adam optimizer with momentum and adaptive learning rates.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = None
        self.m_biases = None
        self.v_weights = None
        self.v_biases = None
        self.t = 0
    
    def update_params(self, layer):
        """
        Updates the parameters of a given layer using Adam.
        """
        # Initialize momentum and velocity if not done yet
        if self.m_weights is None:
            self.m_weights = np.zeros_like(layer.weights)
            self.m_biases = np.zeros_like(layer.biases)
            self.v_weights = np.zeros_like(layer.weights)
            self.v_biases = np.zeros_like(layer.biases)
        
        # Increment time step
        self.t += 1
        
        # Update momentum for weights
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * layer.dweights
        self.m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * layer.dbiases
        
        # Update velocity for weights
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * layer.dweights**2
        self.v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * layer.dbiases**2
        
        # Bias correction
        m_weights_corrected = self.m_weights / (1 - self.beta1**self.t)
        m_biases_corrected = self.m_biases / (1 - self.beta1**self.t)
        v_weights_corrected = self.v_weights / (1 - self.beta2**self.t)
        v_biases_corrected = self.v_biases / (1 - self.beta2**self.t)
        
        # Update parameters
        layer.weights += -self.learning_rate * m_weights_corrected / (np.sqrt(v_weights_corrected) + self.epsilon)
        layer.biases += -self.learning_rate * m_biases_corrected / (np.sqrt(v_biases_corrected) + self.epsilon)

class Optimizer_RMSprop:
    """
    RMSprop optimizer with adaptive learning rates.
    """
    def __init__(self, learning_rate=0.001, rho=0.9, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.v_weights = None
        self.v_biases = None
    
    def update_params(self, layer):
        """
        Updates the parameters of a given layer using RMSprop.
        """
        # Initialize velocity if not done yet
        if self.v_weights is None:
            self.v_weights = np.zeros_like(layer.weights)
            self.v_biases = np.zeros_like(layer.biases)
        
        # Update velocity for weights
        self.v_weights = self.rho * self.v_weights + (1 - self.rho) * layer.dweights**2
        self.v_biases = self.rho * self.v_biases + (1 - self.rho) * layer.dbiases**2
        
        # Update parameters
        layer.weights += -self.learning_rate * layer.dweights / (np.sqrt(self.v_weights) + self.epsilon)
        layer.biases += -self.learning_rate * layer.dbiases / (np.sqrt(self.v_biases) + self.epsilon)

# --- Verification ---
if __name__ == '__main__':
    print("=== Testing Optimizers ===\n")
    
    # Create a mock layer for testing
    class MockLayer:
        def __init__(self):
            self.weights = np.array([[0.1, 0.2], [0.3, 0.4]])
            self.biases = np.array([[0.01, 0.02]])
            self.dweights = np.array([[0.01, 0.02], [0.03, 0.04]])
            self.dbiases = np.array([[0.001, 0.002]])
    
    layer = MockLayer()
    
    print("Initial weights:", layer.weights)
    print("Initial biases:", layer.biases)
    print("Gradients - weights:", layer.dweights)
    print("Gradients - biases:", layer.dbiases)
    
    # Test SGD
    print("\n--- SGD Test ---")
    sgd = Optimizer_SGD(learning_rate=0.1)
    sgd.update_params(layer)
    print("After SGD update:")
    print("Weights:", layer.weights)
    print("Biases:", layer.biases)
    
    # Reset for next test
    layer.weights = np.array([[0.1, 0.2], [0.3, 0.4]])
    layer.biases = np.array([[0.01, 0.02]])
    
    # Test Adam
    print("\n--- Adam Test ---")
    adam = Optimizer_Adam(learning_rate=0.1)
    adam.update_params(layer)
    print("After Adam update:")
    print("Weights:", layer.weights)
    print("Biases:", layer.biases)
    
    # Reset for next test
    layer.weights = np.array([[0.1, 0.2], [0.3, 0.4]])
    layer.biases = np.array([[0.01, 0.02]])
    
    # Test RMSprop
    print("\n--- RMSprop Test ---")
    rmsprop = Optimizer_RMSprop(learning_rate=0.1)
    rmsprop.update_params(layer)
    print("After RMSprop update:")
    print("Weights:", layer.weights)
    print("Biases:", layer.biases)