import numpy as np

class RNNCell:
    """
    A single cell of a Recurrent Neural Network.
    """
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_xh = 0.01 * np.random.randn(input_size, hidden_size)
        self.W_hh = 0.01 * np.random.randn(hidden_size, hidden_size)
        self.b_h = np.zeros((1, hidden_size))

    def forward(self, xt, h_prev):
        h_next = np.tanh(np.dot(xt, self.W_xh) + np.dot(h_prev, self.W_hh) + self.b_h)
        return h_next

# NEW CLASS: RNNLayer
class RNNLayer:
    """
    A full layer of a Recurrent Neural Network that processes sequences.
    """
    def __init__(self, input_size, hidden_size):
        """
        Initializes the RNNLayer, which uses an RNNCell internally.
        """
        self.cell = RNNCell(input_size, hidden_size)

    def forward(self, X):
        """
        Performs a forward pass for an entire input sequence.

        Args:
            X (np.array): Input data for the whole sequence,
                          shape (sequence_length, input_size).

        Returns:
            np.array: The hidden states for each time step,
                      shape (sequence_length, hidden_size).
        """
        # Get the dimensions from the input data
        sequence_length, _ = X.shape
        hidden_size = self.cell.hidden_size

        # Initialize the first hidden state to zeros
        h_prev = np.zeros((1, hidden_size))

        # List to store all the hidden states from each time step
        outputs = []

        # Loop through each time step in the sequence
        for t in range(sequence_length):
            # Get the input for the current time step
            xt = X[t:t+1, :] # Shape (1, input_size)

            # Use the cell to compute the next hidden state
            h_next = self.cell.forward(xt, h_prev)

            # Store the result
            outputs.append(h_next)

            # Update the previous hidden state for the next iteration
            h_prev = h_next

        # Stack the outputs into a single numpy array
        return np.vstack(outputs)


# --- NEW Verification (for the RNNLayer) ---
if __name__ == '__main__':
    np.random.seed(0) # for predictable results

    # Configuration
    sequence_length = 5 # e.g., a sentence with 5 words
    input_size = 10     # each word is a 10-dim vector
    hidden_size = 20    # the size of our memory

    # Create an RNN Layer
    rnn_layer = RNNLayer(input_size, hidden_size)

    # Create a dummy input sequence
    X = np.random.randn(sequence_length, input_size)

    # Perform a forward pass for the entire sequence
    all_hidden_states = rnn_layer.forward(X)

    print("--- RNNLayer Test ---")
    print("Input sequence shape (X):", X.shape)
    print("\nShape of all hidden states:", all_hidden_states.shape)
    print("\nThis shows we have a 20-dimensional hidden state for each of the 5 time steps.")