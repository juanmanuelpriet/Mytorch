import numpy as np
from mytorch.optim.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, model, lr=0.1, momentum=0):
        super().__init__(model, lr)
        self.mu = momentum
        self.v_W = [np.zeros_like(l.W) for l in self.layers]
        self.v_b = [np.zeros_like(l.b) for l in self.layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            if self.mu == 0:
                layer.W -= self.lr * layer.dLdW
                layer.b -= self.lr * layer.dLdb
            else:
                self.v_W[i] = self.mu * self.v_W[i] + layer.dLdW
                self.v_b[i] = self.mu * self.v_b[i] + layer.dLdb
                layer.W -= self.lr * self.v_W[i]
                layer.b -= self.lr * self.v_b[i]
