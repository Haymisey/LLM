import numpy as np
import sys
import os

# --- Add the project root to the Python path ---
# This allows us to import from the 'neural_nets' directory
# (A more robust solution would be to make 'src' a package)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.rnn_lstm.lstm import LSTMLayer
from src.neural_nets.layer import Layer

# --- Configuration ---
np.random.seed(0)
sequence_length = 4  # "I love this film" -> 4 words
embedding_size = 10  # Each word is represented by a 10-dim vector
hidden_size = 20     # LSTM memory size
output_size = 2      # Final output: [prob_negative, prob_positive]

# --- Build the Model ---
# 1. The LSTM layer to read the sequence
lstm = LSTMLayer(input_size=embedding_size, hidden_size=hidden_size)

# 2. The final classification layer (a simple feedforward layer)
#    It takes the final hidden state from the LSTM as its input.
output_layer = Layer(n_inputs=hidden_size, n_neurons=output_size)


# --- Create a Dummy Input Sequence ---
# This represents a single sentence, e.g., "I love this film"
# Shape: (sequence_length, embedding_size)
sentence_sequence = np.random.randn(sequence_length, embedding_size)

print("--- Sentiment Analysis Example ---")
print(f"Input sentence shape: {sentence_sequence.shape}\n")

# --- Forward Pass ---

# 1. Process the sequence through the LSTM layer
all_h, _ = lstm.forward(sentence_sequence)
print(f"LSTM produced {all_h.shape[0]} hidden states, each of size {all_h.shape[1]}.")

# 2. We only care about the *last* hidden state, as it summarizes the whole sentence.
#    The last hidden state is at index -1.
final_hidden_state = all_h[-1:, :] # Shape: (1, hidden_size)
print(f"Using the final hidden state as a sentence summary. Shape: {final_hidden_state.shape}")

# 3. Pass the summary through the final classification layer
final_prediction = output_layer.forward(final_hidden_state)
print(f"Final prediction layer output shape: {final_prediction.shape}\n")

print("Final raw prediction (logits):", final_prediction)
print("\nThis raw output would then be passed to a Softmax function")
print("to get the final probabilities for [negative, positive].")