import numpy as np
from mytorch.optim.optimizer import Optimizer

# ==============================================================================
# CAPÍTULO: DESCENSO DE GRADIENTE ESTOCÁSTICO (SGD)
# ==============================================================================
# 
# El algoritmo más fundamental de optimización.
# 
# Concepto:
#   Imagina que estás en una montaña con mucha niebla. Solo puedes sentir la 
#   pendiente bajo tus pies. El SGD te dice que camines siempre hacia abajo 
#   con pasos de tamaño 'lr'.
# 
# Momentum (Inercia):
#   Añadimos una "bola de nieve" que rueda. Si el gradiente apunta siempre en la 
#   misma dirección, la velocidad aumenta, ayudando a salir de mínimos locales 
#   y valles planos.
# ==============================================================================

class SGD(Optimizer):
    def __init__(self, model, lr=0.1, momentum=0):
        """
        Args:
            momentum (float): Factor de fricción (0 a 1). 
                              0.9 es un valor estándar.
        """
        super().__init__(model, lr)
        self.mu = momentum
        
        # ----------------------------------------------------------------------
        # BUFFER DE VELOCIDAD
        # ----------------------------------------------------------------------
        # Necesitamos recordar la velocidad anterior para cada parámetro.
        # ----------------------------------------------------------------------
        self.v_W = [np.zeros_like(l.W) for l in self.layers]
        self.v_b = [np.zeros_like(l.b) for l in self.layers]

    def step(self):
        """
        Regla de Actualización:
        1. v = mu * v + gradiente
        2. parámetro = parámetro - lr * v
        """
        for i, layer in enumerate(self.layers):
            if self.mu == 0:
                # Vanilla SGD (Sin inercia)
                layer.W -= self.lr * layer.dLdW
                layer.b -= self.lr * layer.dLdb
            else:
                # SGD con Momentum
                # Actualizamos la velocidad acumulada
                self.v_W[i] = self.mu * self.v_W[i] + layer.dLdW
                self.v_b[i] = self.mu * self.v_b[i] + layer.dLdb
                
                # Aplicamos el paso proporcional a la velocidad
                layer.W -= self.lr * self.v_W[i]
                layer.b -= self.lr * self.v_b[i]
