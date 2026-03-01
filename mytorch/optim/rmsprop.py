import numpy as np
from mytorch.optim.optimizer import Optimizer

# ==============================================================================
# CAPÍTULO: RMSPROP (ROOT MEAN SQUARE PROPAGATION)
# ==============================================================================
# 
# Problema que resuelve: 
#   En el descenso de gradiente estándar, si la superficie del error es muy 
#   empinada en una dirección y muy plana en otra, oscilamos sin avanzar.
# 
# Solución de RMSprop:
#   Divide el gradiente por la raíz de un promedio móvil de los gradientes 
#   cuadrados. Esto normaliza la magnitud de la actualización, permitiendo 
#   avanzar de forma estable incluso en terrenos accidentados.
# ==============================================================================

class RMSprop(Optimizer):
    def __init__(self, model, lr=0.01, alpha=0.99, eps=1e-8):
        """
        Args:
            alpha (float): Factor de decaimiento para el promedio móvil.
            eps (float): Epsilon para evitar divisiones infinitas.
        """
        super().__init__(model, lr)
        self.alpha = alpha
        self.eps = eps
        
        # ----------------------------------------------------------------------
        # CACHE DE ENERGÍA (Momento de orden 2)
        # ----------------------------------------------------------------------
        # Guardamos la magnitud histórica de los gradientes.
        # ----------------------------------------------------------------------
        self.v_W = [np.zeros_like(layer.W) for layer in self.layers]
        self.v_b = [np.zeros_like(layer.b) for layer in self.layers]

    def step(self):
        """
        Regla de Actualización:
        1. E_grad^2 = alpha * E_grad^2 + (1 - alpha) * grad^2
        2. w = w - lr * grad / sqrt(E_grad^2 + eps)
        """
        for i, layer in enumerate(self.layers):
            # A. Actualizar caché para los Pesos (W)
            self.v_W[i] = self.alpha * self.v_W[i] + (1 - self.alpha) * (layer.dLdW**2)
            # Escalamos el paso: a mayor varianza histórica, paso más pequeño.
            layer.W -= self.lr * layer.dLdW / (np.sqrt(self.v_W[i]) + self.eps)
            
            # B. Actualizar caché para los Sesgos (b)
            self.v_b[i] = self.alpha * self.v_b[i] + (1 - self.alpha) * (layer.dLdb**2)
            layer.b -= self.lr * layer.dLdb / (np.sqrt(self.v_b[i]) + self.eps)
