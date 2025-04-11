"""This code is part of the mytorch library. It implements a Multi-Layer Perceptron (MLP) with various configurations."""

import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU

class MLP0:

    def __init__(self, debug=False):
        """
        Initialize a single linear layer of shape (2,3).
        Use Relu activations for the layer.
        """

        self.layers = [Linear(2, 3), ReLU()]
        self.debug = debug

    def forward(self, A0):
        """
        Forward pass through MLP0
        Args:
            A0: input matrix (n x 2)
        Returns:
            A1: output matrix (n x 3)
        """

        Z0 = self.layers[0].forward(A0)  # Linear Layer
        A1 = self.layers[1].forward(Z0)

        if self.debug:
            self.A0 = A0
            self.Z0 = Z0
            self.A1 = A1

        return A1

    def backward(self, dLdA1):
        """
        Backward pass through MLP0
        Args:
            dLdA1: gradient of loss w.r.t. A1 (n x 3)
        Returns:
            dLdA0: gradient of loss w.r.t. A0 (n x 2)
        """       
        dLdZ0 = self.layers[1].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug:

            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP1:

    def __init__(self, debug=False):
        """
        Initialize 2 linear layers. Layer 1 of shape (2,3) and Layer 2 of shape (3, 2).
        Use Relu activations for both the layers.
        Implement it on the same lines(in a list) as MLP0
        """

        self.layers = [Linear(2,3),ReLU(),Linear(3,2),ReLU()]
        self.debug = debug

    def forward(self, A0):
        """
         Forward pass through all layers
        Args:
            A0: input matrix (n x 2)
        Returns:
            A2: output matrix (n x 2).
        """

        Z0 = self.layers[0].forward(A0)
        A1 = self.layers[1].forward(Z0)

        Z1 = self.layers[2].forward(A1)
        A2 = self.layers[3].forward(Z1)

        if self.debug:
            self.A0 = A0
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return A2

    def backward(self, dLdA2):
        """
         Backward pass through all layers
        Args:
            dLdA2: gradient of loss w.r.t. A2 (n x 2)
        Returns:
            dLdA0: gradient of loss w.r.t. A0 (n x 2)
        """

        dLdZ1 = self.layers[3].backward(dLdA2)
        dLdA1 = self.layers[2].backward(dLdZ1)

        dLdZ0 = self.layers[1].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug:

            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1

            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP4:
    def __init__(self, debug=False):
        """
         Initialize 4 hidden layers and an output layer with ReLU activations
        Layer structure:
        Input (2) → Linear(2,4) → ReLU → Linear(4,8) → ReLU → Linear(8,8) → ReLU 
        → Linear(8,4) → ReLU → Linear(4,2))
        """

        # List of Hidden and activation Layers in the correct order
        self.layers = [
            Linear(2, 4), ReLU(),    # First hidden layer
            Linear(4, 8), ReLU(),    # Second hidden layer
            Linear(8, 8), ReLU(),    # Third hidden layer
            Linear(8, 4), ReLU(),    # Fourth hidden layer
            Linear(4, 2), ReLU()             # Output layer
        ]
        self.debug = debug

    def forward(self, A0):
        """
        Forward pass through all layers
        Args:
            A0: input matrix (n x 2)
        Returns:
            A5: output matrix (n x 2).
        """

        if self.debug:
            self.A = [A0]
             # Initialize list with input
        A= A0

        for layer in self.layers:

            A = layer.forward(A)

            if self.debug:
                self.A.append(A)
        return A

    def backward(self, dLdA):
        """
         Backward pass through all layers
        Args:
            dLdA: gradient of loss w.r.t. output (n x 2)
        Returns:
            dLdA0: gradient of loss w.r.t. input (n x 2)
        """

        if self.debug:

            self.dLdA = [dLdA]

        for layer in reversed(self.layers):

            dLdA = layer.backward(dLdA)

            if self.debug:

                self.dLdA.insert(0, dLdA)

        return dLdA
