"""This module contains loss functions for training neural networks."""

import numpy as np

class MSELoss:
    """
    Mean Squared Error Loss
    Commonly used for regression problems
    L = (1/N*C) * Σ(y - ŷ)²
    where N is batch size, C is number of features
    """
    def forward(self, A, Y):
        """
          Calculate the Mean Squared error
    Args:
        A: Output of the model of shape (N, C)
            N = batch size, C = number of features
        Y: Ground-truth values of shape (N, C)
    Returns:
        L: MSE Loss (scalar)

        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]

        se = (A - Y) ** 2 # Shape: (N, C)

        i_n = np.ones(self.N).reshape(self.N,1)
        i_c = np.ones(self.C).reshape(self.C,1)
        # Sum of squared errors
        sse = i_n.T@se@i_c
        # Normalize by N*C
        mse = sse/(self.N*self.C)
         # Extract scalar value
        L=mse[0][0]

        return L

    def backward(self):
        """
        Compute gradient of MSE loss with respect to model outputs
        Returns:
            dLdA: Gradient of shape (N, C)
                 dL/dA = 2(A-Y)/(N*C)
        """
        dLdA = 2*(self.A-self.Y)/(self.N*self.C)  

        return dLdA


class CrossEntropyLoss:
    """
    Cross Entropy Loss with Softmax
    Commonly used for classification problems
    L = -(1/N) * Σ y_i * log(softmax(x_i))
    where N is batch size, y_i are true labels
    """
    def forward(self, A, Y):
        """
        Calculate Cross Entropy loss with softmax activation
        Args:
            A: Model logits of shape (N, C)
               N = batch size, C = number of classes
            Y: One-hot encoded targets of shape (N, C)
        Returns:
            L: Cross Entropy Loss (scalar)
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]

        Ones_C = np.ones((self.C,1),dtype='f')
        Ones_N = np.ones((self.N,1),dtype='f')
        # Compute softmax probabilities
        # exp(x_i) / Σexp(x_j)
        self.softmax = np.exp(A)/np.sum(np.exp(A),axis=1,keepdims=True)
        # Compute cross entropy
        # -Σ y_i * log(softmax_i)
        crossentropy = -np.dot(Y* np.log(self.softmax),Ones_C)
        #Y->NxC, log(softmax)->NxC,product->NxC(Element-wise), Ones_C->Cx1,crossentropy->Nx1

        # Average over batch
        sum_crossentropy = Ones_N.T @ crossentropy 
        L = sum_crossentropy / self.N

        return L[0][0]

    def backward(self):
        """
        Compute gradient of Cross Entropy loss with respect to logits
        The gradient simplifies to: dL/dA = (softmax(A) - Y)/N
        Returns:
            dLdA: Gradient of shape (N, C)
        """
        dLdA = (self.softmax - self.Y)/self.N

        return dLdA