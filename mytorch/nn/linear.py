"""This code is part of the mytorch library. It implements a Linear (Fully Connected) Layer."""

import numpy as np

class Linear:
    """
    Linear (Fully Connected) Layer
    Applies linear transformation: Y = XW^T + b
    Where:
        X: Input matrix (N x in_features)
        W: Weight matrix (out_features x in_features)
        b: Bias vector (out_features x 1)
        Y: Output matrix (N x out_features)
    """
    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize linear layer parameters
        Args:
            in_features: Number of input features (C0)
            out_features: Number of output features (C1)
            debug: Boolean to store intermediate values
        """
        self.W = np.zeros(shape=(out_features,in_features))
        self.b = np.zeros(shape=(out_features,1)) 

        self.debug = debug

    def forward(self, A):
        """
        Forward pass of linear layer
        Args:
            A: Input matrix of shape (N, C0) where:
               N = batch size
               C0 = number of input features
        Returns:
            Z: Output matrix of shape (N, C1) where:
               C1 = number of output features
        
        Note: Z = AW^T + b where W^T is the transpose of W
        """
        self.A = A
        self.N = self.A.shape[0]  #store the batch size of input
        # Think how will self.Ones helps in the calculations and uncomment below
        Z = np.dot(self.A,self.W.T)+self.b.T

        return Z

    def backward(self, dLdZ):
        """
        Backward pass of linear layer
        Args:
            dLdZ: Gradient of loss with respect to output Z, shape (N, C1)
        Returns:
            dLdA: Gradient of loss with respect to input A, shape (N, C0)
        
        Computes:
            dL/dW = dL/dZ · A      [Matrix: (C1 x N) · (N x C0) -> (C1 x C0)]
            dL/db = sum(dL/dZ)     [Vector: sum over N -> (C1 x 1)]
            dL/dA = dL/dZ · W      [Matrix: (N x C1) · (C1 x C0) -> (N x C0)]
        """
        
        self.dLdW = np.dot(dLdZ.T,self.A)
        self.dLdb = np.sum(dLdZ, axis=0).reshape(-1, 1)
        dLdA = np.dot(dLdZ,self.W)
        if self.debug:
            
            self.dLdA = dLdA

        return dLdA
