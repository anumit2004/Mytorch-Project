"""This module contains various activation functions used in neural networks.
"""
import numpy as np
import scipy

class Identity:
    """
    Identity activation function: f(x) = x
    Simply passes the input through unchanged
    Derivative is always 1
    Used primarily for testing or as a placeholder
    """
    def forward(self, Z):
        """
        Forward pass for Identity activation
        Args:
            Z: Input array of any shape
        Returns:
            A: Same array as input (identity function)
        """
        self.A = Z

        return self.A

    def backward(self, dLdA):
        """
        Backward pass for Identity activation
        Derivative of identity function is 1 everywhere
        Args:
            dLdA: Gradient of loss with respect to output
        Returns:
            dLdZ: Same as input gradient (since derivative is 1)
        """
        dAdZ = np.ones(self.A.shape, dtype="f")
        dLdZ = dLdA * dAdZ

        return dLdZ


class Sigmoid:
    """
    Sigmoid activation function: f(x) = 1/(1 + e^(-x))
    Squashes input to range [0,1]
    Commonly used in binary classification output layers
    """
    def forward(self, Z):
        """
        Forward pass applies sigmoid function element-wise
        Args:
            Z: Input array
        Returns:
            A: Output array with values in range [0,1]
        """
        self.Z=Z
        self.A = 1 / (1+np.exp(-Z))

        return self.A
    
   #Define 'backward' function
    def backward(self, dLdA):
        """
        Backward pass for sigmoid
        Derivative: f'(x) = f(x)(1 - f(x))
        Args:
            dLdA: Gradient of loss with respect to output
        Returns:
            dLdZ: Gradient with respect to input
        """
        dAdZ = self.A * (1 - self.A)
        dLdZ = dLdA * dAdZ

        return dLdZ

class Tanh:
    """
    Hyperbolic tangent activation: f(x) = tanh(x)
    Squashes input to range [-1,1]
    Zero-centered, making it useful in hidden layers
    """
    def forward(self,Z):
        """
        Forward pass applies tanh function element-wise
        Args:
            Z: Input array
        Returns:
            A: Output array with values in range [-1,1]
        """
        self.Z=Z
        self.A=np.tanh(Z)
        
        return self.A
    
    def backward(self,dLdA):
        """
        Backward pass for tanh
        Derivative: f'(x) = 1 - tanh²(x)
        Args:
            dLdA: Gradient of loss with respect to output
        Returns:
            dLdZ: Gradient with respect to input
        """
        dLdZ=dLdA * (1 - np.tanh(self.Z)**2)
    
        return dLdZ


class ReLU:
    """
    Rectified Linear Unit: f(x) = max(0,x)
    Most commonly used activation in modern networks
    Helps prevent vanishing gradient problem
    """
    def forward(self,Z):
        """
        Forward pass keeps positive values, zeros out negatives
        Args:
            Z: Input array
        Returns:
            A: Output array with negative values set to 0
        """
        self.Z=Z
        self.A=np.maximum(0,Z)
        
        return self.A
    
    def backward(self,dLdA):
        """
        Backward pass for ReLU
        Derivative: 1 if x > 0, 0 if x <= 0
        Args:
            dLdA: Gradient of loss with respect to output
        Returns:
            dLdZ: Gradient with respect to input
        """
        dLdZ = dLdA * (self.Z > 0).astype(float)            
        return dLdZ

class GELU:
    """
    Gaussian Error Linear Unit
    Smooth approximation of ReLU that applies stochastic regularization
    Used in modern transformers like BERT, GPT
    f(x) = 0.5x(1 + erf(x/√2))
    """
    def forward(self,Z):
        """
        Forward pass for GELU activation
        Args:
            Z: Input array
        Returns:
            A: Output array after GELU transformation
        """
        self.Z=Z
        self.erf_term = scipy.special.erf(Z/np.sqrt(2))
        self.A = 0.5 * Z * (1 + self.erf_term)
        
        return self.A 
    
    def backward(self,dLdA):
        """
        Backward pass for GELU
        Derivative combines gaussian and error function terms
        Args:
            dLdA: Gradient of loss with respect to output
        Returns:
            dLdZ: Gradient with respect to input
        """
        constant = 1/np.sqrt(2*np.pi)
        gaussian = np.exp(-0.5 * self.Z**2)
        dAdZ = 0.5 * (1 + self.erf_term) + constant * self.Z * gaussian
        dLdZ=dLdA*dAdZ
        return dLdZ
class Softmax:
    """
    Softmax activation function
    Converts logits to probability distribution
    Commonly used in classification output layers
    Note: Acts on entire rows, not element-wise
    """

    def forward(self, Z):
        """
        Forward pass computes softmax probabilities
        Args:
            Z: Input logits array of shape (N, C)
        Returns:
            A: Probability distribution over C classes for each sample
        """
        self.A = np.zeros(Z.shape, dtype="f")
        for i in range(Z.shape[0]):
            expZ = np.exp(Z[i])
            sumExpZ = np.sum(expZ)
            self.A[i] = expZ / sumExpZ


        return self.A
    
    def backward(self, dLdA):
        """
        Backward pass for softmax
        Requires special handling due to interdependence between outputs
        Args:
            dLdA: Gradient of loss with respect to output probabilities
        Returns:
            dLdZ: Gradient with respect to input logits
        """
        N = dLdA.shape[0]
        C = dLdA.shape[1]
        # dLdA is of shape (N, C) and self.A is of shape (N, C)
        dLdZ = np.zeros_like(dLdA, dtype="f")

        for i in range(dLdA.shape[0]):
            a = self.A[i]
        # Compute Jacobian matrix for this example
            diag_a = np.diag(a)
            outer_a = np.outer(a, a)
            jacobian = diag_a - outer_a
        
        # Multiply by incoming gradient
            dLdZ[i] = np.dot(dLdA[i], jacobian)

        return dLdZ
