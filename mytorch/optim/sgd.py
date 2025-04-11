"""This module implements the Stochastic Gradient Descent (SGD) optimizer.It supports both vanilla SGD and SGD with momentum."""

import numpy as np

class SGD:
    """
    Stochastic Gradient Descent Optimizer
    Supports both vanilla SGD and SGD with momentum
    
    Updates weights using the formula:
    - Vanilla SGD: w = w - lr * dw
    - With momentum: v = mu * v + dw
                    w = w - lr * v
    """
    def __init__(self, model, lr=0.1, momentum=0):
        """
        Initialize SGD optimizer with learning rate and momentum
        Args:
            model: Neural network model with .layers attribute
            lr: Learning rate (step size) for gradient descent
            momentum: Momentum factor (mu) in range [0,1]
                     0 = vanilla SGD
                     >0 = SGD with momentum
        """
        self.l = model.layers
        self.L = len(model.layers)
        self.lr = lr
        self.mu = momentum
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f")
                    for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f")
                    for i in range(self.L)]

    def step(self):
        """
        Update weights and biases using SGD with momentum
        
        For each layer with parameters:
        1. If mu=0: Vanilla SGD
           w = w - lr * dw
           b = b - lr * db
        
        2. If mu>0: SGD with momentum 
           v_w = mu * v_w + dw    # Update velocity
           v_b = mu * v_b + db
           w = w - lr * v_w       # Update parameters
           b = b - lr * v_b
        """
        layer_idx = 0
        for i in range(self.L):
            # skip non-linear layers like ReLU
            if not hasattr(self.l[i], "W"):
                continue 
            if self.mu == 0:
                 # Standard SGD update
                self.l[i].W -= self.lr * self.l[i].dLdW
                self.l[i].b -= self.lr * self.l[i].dLdb

            else:
                 # SGD with momentum update
                self.v_W[layer_idx] = self.mu * self.v_W[layer_idx] +self.l[i].dLdW
                self.v_b[layer_idx] = self.mu * self.v_b[layer_idx] +self.l[i].dLdb

                self.l[i].W -=  self.lr * self.v_W[layer_idx]
                self.l[i].b -=  self.lr * self.v_b[layer_idx]
            layer_idx += 1

