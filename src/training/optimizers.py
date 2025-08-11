"""
Optimizers and Learning Rate Scheduling
======================================

This module implements optimization algorithms and learning rate scheduling
for training the Transformer model.
"""

import numpy as np


class SGDOptimizer:
    """
    Stochastic Gradient Descent optimizer with momentum.
    
    Implements SGD with optional momentum and weight decay.
    """
    
    def __init__(self, learning_rate=0.01, momentum=0.0, weight_decay=0.0):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate (float): Initial learning rate
            momentum (float): Momentum factor (0.0 = no momentum)
            weight_decay (float): Weight decay factor (L2 regularization)
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}  # Momentum velocity for each parameter
        self.step_count = 0
    
    def step(self, parameters, gradients):
        """
        Perform optimization step.
        
        Args:
            parameters (dict): Dictionary of parameter arrays
            gradients (dict): Dictionary of gradient arrays
        
        Returns:
            dict: Updated parameters
        """
        self.step_count += 1
        
        updated_params = {}
        
        for name, param in parameters.items():
            if name not in gradients or gradients[name] is None:
                updated_params[name] = param.copy()
                continue
            
            grad = gradients[name]
            
            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Initialize velocity if not exists
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(param)
            
            # Update velocity with momentum
            if self.momentum > 0:
                self.velocity[name] = self.momentum * self.velocity[name] + grad
                update = self.velocity[name]
            else:
                update = grad
            
            # Update parameter
            updated_params[name] = param - self.learning_rate * update
        
        return updated_params
    
    def zero_grad(self):
        """Reset gradients and velocity."""
        self.velocity = {}
    
    def get_learning_rate(self):
        """Get current learning rate."""
        return self.learning_rate
    
    def set_learning_rate(self, lr):
        """Set learning rate."""
        self.learning_rate = lr


class AdamOptimizer:
    """
    Adam optimizer with adaptive learning rates.
    
    Implements Adam algorithm with bias correction and adaptive moment estimation.
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate (float): Initial learning rate
            beta1 (float): Exponential decay rate for first moment estimates
            beta2 (float): Exponential decay rate for second moment estimates
            epsilon (float): Small constant for numerical stability
            weight_decay (float): Weight decay factor (L2 regularization)
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # First and second moment estimates
        self.m = {}  # First moment (mean)
        self.v = {}  # Second moment (variance)
        self.step_count = 0
    
    def step(self, parameters, gradients):
        """
        Perform optimization step.
        
        Args:
            parameters (dict): Dictionary of parameter arrays
            gradients (dict): Dictionary of gradient arrays
        
        Returns:
            dict: Updated parameters
        """
        self.step_count += 1
        
        updated_params = {}
        
        for name, param in parameters.items():
            if name not in gradients or gradients[name] is None:
                updated_params[name] = param.copy()
                continue
            
            grad = gradients[name]
            
            # Apply weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            
            # Initialize moment estimates if not exists
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)
            
            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[name] / (1 - self.beta1 ** self.step_count)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[name] / (1 - self.beta2 ** self.step_count)
            
            # Update parameter
            update = m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_params[name] = param - self.learning_rate * update
        
        return updated_params
    
    def zero_grad(self):
        """Reset moment estimates."""
        self.m = {}
        self.v = {}
    
    def get_learning_rate(self):
        """Get current learning rate."""
        return self.learning_rate
    
    def set_learning_rate(self, lr):
        """Set learning rate."""
        self.learning_rate = lr


class LearningRateScheduler:
    """
    Learning rate scheduler for training optimization.
    
    Implements various learning rate scheduling strategies.
    """
    
    def __init__(self, optimizer, scheduler_type='constant', **kwargs):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: Optimizer instance to schedule
            scheduler_type (str): Type of scheduler ('constant', 'step', 'exponential', 'cosine')
            **kwargs: Scheduler-specific parameters
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        self.initial_lr = optimizer.learning_rate
        self.step_count = 0
        
        # Scheduler-specific parameters
        if scheduler_type == 'step':
            self.step_size = kwargs.get('step_size', 30)
            self.gamma = kwargs.get('gamma', 0.1)
        elif scheduler_type == 'exponential':
            self.gamma = kwargs.get('gamma', 0.95)
        elif scheduler_type == 'cosine':
            self.T_max = kwargs.get('T_max', 100)
            self.eta_min = kwargs.get('eta_min', 0.0)
        elif scheduler_type == 'warmup_cosine':
            self.warmup_steps = kwargs.get('warmup_steps', 4000)
            self.max_steps = kwargs.get('max_steps', 100000)
            self.eta_min = kwargs.get('eta_min', 0.0)
    
    def step(self):
        """Update learning rate based on scheduler."""
        self.step_count += 1
        
        if self.scheduler_type == 'constant':
            lr = self.initial_lr
        
        elif self.scheduler_type == 'step':
            lr = self.initial_lr * (self.gamma ** (self.step_count // self.step_size))
        
        elif self.scheduler_type == 'exponential':
            lr = self.initial_lr * (self.gamma ** self.step_count)
        
        elif self.scheduler_type == 'cosine':
            lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                 (1 + np.cos(np.pi * self.step_count / self.T_max)) / 2
        
        elif self.scheduler_type == 'warmup_cosine':
            if self.step_count < self.warmup_steps:
                # Linear warmup
                lr = self.initial_lr * self.step_count / self.warmup_steps
            else:
                # Cosine annealing
                progress = (self.step_count - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                lr = self.eta_min + (self.initial_lr - self.eta_min) * \
                     (1 + np.cos(np.pi * progress)) / 2
        
        else:
            lr = self.initial_lr
        
        # Update optimizer learning rate
        self.optimizer.set_learning_rate(lr)
        return lr
    
    def get_last_lr(self):
        """Get the last computed learning rate."""
        return self.optimizer.get_learning_rate()
    
    def get_learning_rate(self):
        """Get current learning rate."""
        return self.optimizer.get_learning_rate()
    
    def reset(self):
        """Reset scheduler state."""
        self.step_count = 0
        self.optimizer.set_learning_rate(self.initial_lr)


def test_optimizers():
    """Test the optimizers and scheduler."""
    print("Testing Optimizers and Scheduler...")
    
    # Test data
    param_names = ['weight1', 'weight2', 'bias']
    parameters = {
        'weight1': np.random.randn(10, 10) * 0.1,
        'weight2': np.random.randn(5, 10) * 0.1,
        'bias': np.random.randn(5) * 0.1
    }
    
    gradients = {
        'weight1': np.random.randn(10, 10) * 0.01,
        'weight2': np.random.randn(5, 10) * 0.01,
        'bias': np.random.randn(5) * 0.01
    }
    
    # Test SGD
    print("\n--- SGD Optimizer ---")
    sgd = SGDOptimizer(learning_rate=0.01, momentum=0.9)
    updated_params = sgd.step(parameters, gradients)
    
    print(f"Original weight1 norm: {np.linalg.norm(parameters['weight1']):.6f}")
    print(f"Updated weight1 norm: {np.linalg.norm(updated_params['weight1']):.6f}")
    print(f"Gradient norm: {np.linalg.norm(gradients['weight1']):.6f}")
    
    # Test Adam
    print("\n--- Adam Optimizer ---")
    adam = AdamOptimizer(learning_rate=0.001)
    updated_params = adam.step(parameters, gradients)
    
    print(f"Original weight1 norm: {np.linalg.norm(parameters['weight1']):.6f}")
    print(f"Updated weight1 norm: {np.linalg.norm(updated_params['weight1']):.6f}")
    
    # Test Learning Rate Scheduler
    print("\n--- Learning Rate Scheduler ---")
    scheduler = LearningRateScheduler(
        adam, 
        scheduler_type='warmup_cosine',
        warmup_steps=5,
        max_steps=20
    )
    
    print("Learning rate progression:")
    for step in range(10):
        lr = scheduler.step()
        print(f"  Step {step + 1}: {lr:.6f}")
    
    print("\n✓ Optimizers and scheduler tested successfully!")


if __name__ == "__main__":
    test_optimizers()
