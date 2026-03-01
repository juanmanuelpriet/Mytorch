import numpy as np
import math as mt
from scipy.special import erf


class Identity:
    def forward(self, Z):
        self.A = Z
        return self.A

    def backward(self, dLdA):
        return dLdA


class Sigmoid:
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        return dLdA * self.A * (1 - self.A)


class Tanh:
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):
        return dLdA * (1 - self.A**2)


class ReLU:
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dLdA):
        return dLdA * (self.Z > 0)


class GeLU:
    def forward(self, Z):
        self.Z = Z
        self.Phi = 0.5 * (1 + erf(Z / np.sqrt(2)))
        self.A = Z * self.Phi
        return self.A

    def backward(self, dLdA):
        phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * self.Z**2)
        dAdZ = self.Phi + self.Z * phi
        return dLdA * dAdZ


class SoftMax:
    def forward(self, Z):
        E = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.A = E / np.sum(E, axis=1, keepdims=True)
        return self.A

    def backward(self, dLdA):
        # Implementación eficiente de dLdA @ Jacobian
        # dL/dZ_i = A_i * (dL/dA_i - \sum_j dL/dA_j * A_j)
        return self.A * (dLdA - np.sum(dLdA * self.A, axis=1, keepdims=True))
