import numpy as np


class Linear:
    def __init__(self, in_features, out_features, debug=False):
        # Inicialización de He/Xavier simplificada para "entrenar full"
        limit = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * limit
        self.b = np.zeros((out_features, 1))

        self.dLdW = None
        self.dLdb = None
        self.debug = debug

    def forward(self, A):
        self.A = A
        self.N = A.shape[0]
        # Z = A @ W.T + b.T
        Z = self.A @ self.W.T + self.b.T
        return Z

    def backward(self, dLdZ):
        dLdA = dLdZ @ self.W
        self.dLdW = dLdZ.T @ self.A / self.N
        self.dLdb = np.sum(dLdZ, axis=0, keepdims=True).T / self.N
        return dLdA
