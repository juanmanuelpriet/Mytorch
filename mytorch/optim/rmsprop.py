import numpy as np
from mytorch.optim.optimizer import Optimizer

class RMSprop(Optimizer):
    def __init__(self, model, lr=0.01, alpha=0.99, eps=1e-8):
        super().__init__(model)
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        
        # Cache para el promedio móvil de los gradientes al cuadrado
        self.v_W = [np.zeros(layer.W.shape) for layer in self.layers]
        self.v_b = [np.zeros(layer.b.shape) for layer in self.layers]

    def step(self):
        for i, layer in enumerate(self.layers):
            # Actualizar cache para W
            self.v_W[i] = self.alpha * self.v_W[i] + (1 - self.alpha) * (layer.dLdW**2)
            layer.W -= self.lr * layer.dLdW / (np.sqrt(self.v_W[i]) + self.eps)
            
            # Actualizar cache para b
            self.v_b[i] = self.alpha * self.v_b[i] + (1 - self.alpha) * (layer.dLdb**2)
            layer.b -= self.lr * layer.dLdb / (np.sqrt(self.v_b[i]) + self.eps)
