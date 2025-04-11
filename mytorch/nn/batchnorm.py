"""This code is part of the mytorch library. It implements a Batch Normalization layer for 1D data."""

import numpy as np

class BatchNorm1d:
    """
    Batch Normalization Layer (1D)
    Normalizes input features to have zero mean and unit variance across batch dimension
    Then applies learnable scale (BW) and shift (Bb) parameters
    """
    def __init__(self, num_features, alpha=0.9):
        """
        Args:
            num_features: Number of input features (D)
            alpha: Exponential moving average factor for running statistics
        """
        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        Forward pass for batch normalization
        Args:
            Z: Input array of shape (N, D) where:
               N = batch size
               D = number of features
            eval: Boolean indicating if in evaluation mode
        Returns:
            BZ: Batch normalized output of shape (N, D)
        """
        self.Z = Z
        self.N = Z.shape[0]
        self.C = Z.shape[1]
        self.M = np.mean(Z, axis=0)
        self.V = np.var(Z, axis=0)

        if eval == False:
            # training mode
            # Normalize
            self.NZ = (Z-self.M)/np.sqrt(self.V+self.eps)
            # Scale and shift
            self.BZ = self.BW*self.NZ+self.Bb

            self.running_M = self.alpha*self.running_M + (1-self.alpha)*self.M
            self.running_V = self.alpha*self.running_V + (1-self.alpha)*self.V
        else:
            # inference mode
            self.NZ = (Z-self.running_M)/np.sqrt(self.running_V+self.eps)
            self.BZ = self.BW*self.NZ+self.Bb

        return self.BZ

    def backward(self, dLdBZ):
        """
         Backward pass for batch normalization.
    Args:
        dLdZ: Gradient of loss with respect to batch normalized output
    Returns:
        dLdA: Gradient of loss with respect to input
        """
        self.dLdBW =np.sum(dLdBZ*self.NZ, axis=0)
        self.dLdBb = np.sum(dLdBZ, axis=0)

        dLdNZ = dLdBZ*self.BW
        dLdV = -0.5*np.sum(dLdNZ*(self.Z-self.M)*(self.V+self.eps)**(-3/2), axis=0)


        term1 = -np.sum(dLdNZ / np.sqrt(self.V + self.eps), axis=0)

        term2 = -np.sum(dLdNZ * (self.Z - self.M) * 2 * np.mean(self.Z - self.M, axis=0) / (self.V + self.eps) ** (3/2),axis=0)

        dLdM = term1 + term2


        dLdZ = dLdNZ*(1/np.sqrt(self.V+self.eps)) + dLdV*2*(self.Z-self.M)/self.N + dLdM/self.N

        return dLdZ
