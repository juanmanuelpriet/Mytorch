import numpy as np
from mytorch.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

        # Buffers para m (momento 1) y v (momento 2)
        self.m_W = [np.zeros_like(l.W) for l in self.layers]
        self.v_W = [np.zeros_like(l.W) for l in self.layers]
        self.m_b = [np.zeros_like(l.b) for l in self.layers]
        self.v_b = [np.zeros_like(l.b) for l in self.layers]

    def step(self):
        self.t += 1
        for i, layer in enumerate(self.layers):
            # Gradientes actuales
            dw = layer.dLdW
            db = layer.dLdb

            # Actualizar momento 1
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * dw
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db

            # Actualizar momento 2
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (dw**2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db**2)

            # Corrección de sesgo
            m_W_hat = self.m_W[i] / (1 - self.beta1**self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_W_hat = self.v_W[i] / (1 - self.beta2**self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)

            # Actualizar parámetros
            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
