import numpy as np

class Loss_MSE:
    """
    Mean Squared Error Loss function.
    """
    def forward(self, y_pred, y_true):
        """
        Calculates the Mean Squared Error loss.
        """
        # Calculate the sample-wise loss
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
        
        # Calculate the mean loss over the batch
        data_loss = np.mean(sample_losses)
        return data_loss

    def backward(self, y_pred, y_true):
        """
        Calculates the gradient of the loss with respect to the predictions.
        """
        # Number of samples in the batch
        samples = len(y_pred)
        # Number of outputs in each sample
        outputs = len(y_pred[0])

        # Calculate the gradient
        self.dinputs = -2 * (y_true - y_pred) / outputs
        # Normalize gradient by number of samples
        self.dinputs = self.dinputs / samples

class Loss_CrossEntropy:
    """
    Cross-Entropy Loss function for multi-class classification.
    """
    def forward(self, y_pred, y_true):
        """
        Calculates the Cross-Entropy loss.
        """
        # Number of samples
        samples = len(y_pred)
        
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate sample-wise losses
        if len(y_true.shape) == 1:
            # If y_true contains class indices
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            # If y_true contains one-hot encoded values
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        
        # Calculate negative log likelihood
        negative_log_likelihoods = -np.log(correct_confidences)
        
        # Return mean loss
        return np.mean(negative_log_likelihoods)
    
    def backward(self, y_pred, y_true):
        """
        Calculates the gradient of the loss with respect to the predictions.
        """
        # Number of samples
        samples = len(y_pred)
        
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate gradient
        if len(y_true.shape) == 1:
            # If y_true contains class indices
            self.dinputs = y_pred_clipped.copy()
            self.dinputs[range(samples), y_true] -= 1
        elif len(y_true.shape) == 2:
            # If y_true contains one-hot encoded values
            self.dinputs = y_pred_clipped - y_true
        
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Loss_BinaryCrossEntropy:
    """
    Binary Cross-Entropy Loss function for binary classification.
    """
    def forward(self, y_pred, y_true):
        """
        Calculates the Binary Cross-Entropy loss.
        """
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate sample-wise losses
        sample_losses = -(y_true * np.log(y_pred_clipped) + 
                         (1 - y_true) * np.log(1 - y_pred_clipped))
        
        # Return mean loss
        return np.mean(sample_losses)
    
    def backward(self, y_pred, y_true):
        """
        Calculates the gradient of the loss with respect to the predictions.
        """
        # Clip data to prevent division by 0
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        # Calculate gradient
        self.dinputs = -(y_true / y_pred_clipped - 
                        (1 - y_true) / (1 - y_pred_clipped))
        
        # Normalize gradient
        self.dinputs = self.dinputs / len(y_pred)

# --- Verification ---
if __name__ == '__main__':
    print("=== Testing Loss Functions ===\n")
    
    # Test MSE Loss
    print("--- MSE Loss Test ---")
    loss_function = Loss_MSE()
    y_pred = np.array([[0.1, 0.8]])
    y_true = np.array([[0.0, 1.0]])
    loss = loss_function.forward(y_pred, y_true)
    loss_function.backward(y_pred, y_true)
    print("Predicted values:", y_pred)
    print("True values:", y_true)
    print("Calculated Loss:", loss)
    print("Gradient:", loss_function.dinputs)
    
    # Test Cross-Entropy Loss
    print("\n--- Cross-Entropy Loss Test ---")
    ce_loss = Loss_CrossEntropy()
    y_pred_ce = np.array([[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]])
    y_true_ce = np.array([1, 0])  # Class indices
    ce_loss_value = ce_loss.forward(y_pred_ce, y_true_ce)
    ce_loss.backward(y_pred_ce, y_true_ce)
    print("Predicted values:", y_pred_ce)
    print("True class indices:", y_true_ce)
    print("Calculated Loss:", ce_loss_value)
    print("Gradient:", ce_loss.dinputs)
    
    # Test Binary Cross-Entropy Loss
    print("\n--- Binary Cross-Entropy Loss Test ---")
    bce_loss = Loss_BinaryCrossEntropy()
    y_pred_bce = np.array([[0.1], [0.8], [0.3]])
    y_true_bce = np.array([[0], [1], [0]])
    bce_loss_value = bce_loss.forward(y_pred_bce, y_true_bce)
    bce_loss.backward(y_pred_bce, y_true_bce)
    print("Predicted values:", y_pred_bce)
    print("True values:", y_true_bce)
    print("Calculated Loss:", bce_loss_value)
    print("Gradient:", bce_loss.dinputs)