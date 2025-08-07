"""
NeuralPredictor - Performance Surrogate Model

A lightweight neural network that:
- Learns to predict model performance from parameters
- Accelerates optimization by reducing expensive evaluations
- Uses Numba-optimized operations for efficiency
"""

import numpy as np
from numba import njit

@njit(parallel=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@njit(parallel=True)
def forward(X, W1, W2):
    H = np.tanh(X @ W1)
    out = sigmoid(H @ W2)
    return H, out

@njit(parallel=True)
def train_njit(X, y, W1, W2, lr, epochs):
    batch_size = min(64, X.shape[0])
    loss_prev = 1e9

    for epoch in range(epochs):
        idx = np.random.choice(X.shape[0], batch_size, replace=False)
        Xb, yb = X[idx], y[idx]

        H, out = forward(Xb, W1, W2)
        error = out - yb
        loss = np.mean(error**2)

        if abs(loss_prev - loss) < 1e-6:
            break
        loss_prev = loss

        grad_W2 = H.T @ (error * out * (1 - out))
        grad_hidden = (error @ W2.T) * (1 - H**2)
        grad_W1 = Xb.T @ grad_hidden

        W2 -= lr * grad_W2
        W1 -= lr * grad_W1
        lr *= 0.99

    return W1, W2, lr

@njit(parallel=True)
def predict_njit(X, W1, W2):
    H = np.tanh(X @ W1)
    return sigmoid(H @ W2)

class NeuralPredictor:
    def __init__(self, param_space):
        """
        Initialize the prediction model.
        
        Args:
            param_space (dict): Parameter space definition used to determine:
                                - Input layer size (based on parameter types)
                                - Normalization ranges
        """
        input_size = 0
        for cfg in param_space.values():
            if cfg['type'] == 'categorical':
                input_size += len(cfg['options'])
            else:
                input_size += 1

        # Architecture: 
        # Input Layer -> 16-neuron Hidden Layer (tanh) -> Output (sigmoid)
        self.W1 = np.random.randn(input_size, 16) * 0.01  # Input->Hidden weights
        self.W2 = np.random.randn(16, 1) * 0.01           # Hidden->Output weights
        self.lr = 0.01  # Initial learning rate


    def train(self, X, y, epochs=5):
        """
        Train the predictor on observed data.
        
        Args:
            X (ndarray): Normalized parameter vectors (n_samples, n_features)
            y (ndarray): Corresponding performance scores (n_samples, 1)
            epochs (int): Training iterations (default: 5)
        
        Features:
            - Mini-batch training (batch_size=64)
            - Adaptive learning rate decay
            - Early stopping on loss plateau
        """
        # Converts y to 2D if needed
        # Uses Numba-optimized training loop
        # Returns updated weights
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        self.W1, self.W2, self.lr = train_njit(X, y, self.W1, self.W2, self.lr, epochs)

    def predict(self, X):
        """
        Predict performance for new parameters.
        
        Args:
            X (ndarray): Normalized parameter vectors
            
        Returns:
            ndarray: Predicted scores (0-1 range)
        
        Note:
            Uses forward pass only (no training overhead)
        """
        # Numba-accelerated prediction
        # Returns sigmoid-activated outputs
        return predict_njit(X, self.W1, self.W2)
