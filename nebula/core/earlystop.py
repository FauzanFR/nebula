"""
EarlyStopOracle

Intelligently detects optimization convergence by:
- Tracking score gradient trends
- Adjusting patience dynamically
- Preventing premature termination
"""

import numpy as np

class EarlyStopOracle:
    def __init__(self, patience=3, window=5, grad_thresh=0.01):
        self.scores = []
        self.patience = patience
        self.grad_thresh = grad_thresh
        self.window = window
        self.max_patience = 6
        self.total_stagnant = 0

    def should_stop(self, new_score):
        """
        Determine whether optimization should terminate.
        
        Args:
            new_score: Latest evaluation score
            
        Returns:
            bool: True if optimization should stop
        """
        self.scores.append(new_score)
        if len(self.scores) < self.window:
            return False

        grad = np.gradient(self.scores[-self.window:]).mean()
        if grad < self.grad_thresh * max(self.scores):
            self.patience -= 1
            self.total_stagnant += 1
        else:
            self.patience = min(3, self.patience + 0.5)
            self.total_stagnant = 0

        return self.patience <= 0 or self.total_stagnant >= self.max_patience