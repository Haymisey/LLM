import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LSTMCell:
    """
    A single cell of a Long Short-Term Memory network.
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W = 0.01 * np.random.randn(input_size + hidden_size, 4 * hidden_size)
        self.b = np.zeros((1, 4 * hidden_size))

    def forward(self, xt, h_prev, C_prev):
        combined_input = np.hstack((h_prev, xt))
        gates = np.dot(combined_input, self.W) + self.b
        f = sigmoid(gates[:, 0:self.hidden_size])
        i = sigmoid(gates[:, self.hidden_size:2*self.hidden_size])
        o = sigmoid(gates[:, 2*self.hidden_size:3*self.hidden_size])
        C_candidate = np.tanh(gates[:, 3*self.hidden_size:])
        C_next = f * C_prev + i * C_candidate
        h_next = o * np.tanh(C_next)
        return h_next, C_next

# NEW CLASS: LSTMLayer
class LSTMLayer:
    """
    A full layer of an LSTM network that processes sequences.
    """
    def __init__(self, input_size, hidden_size):
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, X):
        """
        Performs a forward pass for an entire input sequence.

        Args:
            X (np.array): Input data for the whole sequence,
                          shape (sequence_length, input_size).

        Returns:
            (np.array, np.array): The hidden states for each step and the final cell state.
        """
        sequence_length, _ = X.shape
        hidden_size = self.cell.hidden_size

        # Initialize the first hidden state and cell state to zeros
        h_prev = np.zeros((1, hidden_size))
        C_prev = np.zeros((1, hidden_size))

        # List to store all the hidden states from each time step
        outputs = []

        # Loop through each time step
        for t in range(sequence_length):
            xt = X[t:t+1, :]
            
            # Use the cell to compute the next states
            h_next, C_next = self.cell.forward(xt, h_prev, C_prev)
            
            outputs.append(h_next)
            
            # Update states for the next iteration
            h_prev = h_next
            C_prev = C_next

        # We return all the hidden states and the final cell state
        return np.vstack(outputs), C_next

# --- NEW Verification (for the LSTMLayer) ---
if __name__ == '__main__':
    np.random.seed(0)

    # Config
    sequence_length = 5
    input_size = 10
    hidden_size = 20

    # Create an LSTM Layer
    lstm_layer = LSTMLayer(input_size, hidden_size)

    # Dummy input sequence
    X = np.random.randn(sequence_length, input_size)

    # Forward pass for the entire sequence
    all_hidden_states, final_cell_state = lstm_layer.forward(X)

    print("--- LSTMLayer Test ---")
    print("Input sequence shape (X):", X.shape)
    print("\nShape of all hidden states:", all_hidden_states.shape)
    print("Shape of the final cell state:", final_cell_state.shape)
    print("\nThis confirms the layer correctly processes a sequence,")
    print("providing a hidden state for each step and the final memory state.")