import numpy as np
from mytorch.optim.optimizer import Optimizer

# ==============================================================================
# CAPÍTULO: ADAM (ADAPTIVE MOMENT ESTIMATION)
# ==============================================================================
# 
# El "Estado del Arte" en optimizadores.
# 
# Problema que resuelve: 
#   El SGD usa el mismo 'lr' para todos los parámetros. Pero algunos gradientes 
#   pueden ser enormes y otros diminutos. Adam adapta la tasa de aprendizaje 
#   individualmente para cada peso.
# 
# Concepto:
#   Combina Momentum (Momento de orden 1) con RMSprop (Momento de orden 2).
#   - m: Promedio móvil del gradiente (Dirección).
#   - v: Promedio móvil del gradiente al cuadrado (Incertidumbre/Escala).
# ==============================================================================

class Adam(Optimizer):
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Args:
            beta1 (float): Decaimiento para el primer momento (Momento lineal).
            beta2 (float): Decaimiento para el segundo momento (Varianza).
            eps (float): Factor de suavizado para no dividir por cero.
        """
        super().__init__(model, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0 # Contador de pasos para la corrección de sesgo

        # ----------------------------------------------------------------------
        # CACHES DE MOMENTOS
        # ----------------------------------------------------------------------
        self.m_W = [np.zeros_like(l.W) for l in self.layers]
        self.v_W = [np.zeros_like(l.W) for l in self.layers]
        self.m_b = [np.zeros_like(l.b) for l in self.layers]
        self.v_b = [np.zeros_like(l.b) for l in self.layers]

    def step(self):
        """
        Regla de Actualización Adam:
        1. Actualizar momentos m y v.
        2. Corregir sesgo inicial (Bias Correction).
        3. Actualizar parámetro: w = w - lr * m_hat / (sqrt(v_hat) + eps)
        """
        self.t += 1
        for i, layer in enumerate(self.layers):
            dw = layer.dLdW
            db = layer.dLdb

            # A. Actualizar momento 1 (Media de los gradientes)
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * dw
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db

            # B. Actualizar momento 2 (Media de los gradientes al cuadrado)
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (dw**2)
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db**2)

            # C. CORRECCIÓN DE SESGO
            # Al inicio m y v son 0, lo que sesga los promedios hacia abajo.
            # Dividiendo por (1 - beta^t) logramos que los primeros pasos sean significativos.
            m_W_hat = self.m_W[i] / (1 - self.beta1**self.t)
            m_b_hat = self.m_b[i] / (1 - self.beta1**self.t)
            v_W_hat = self.v_W[i] / (1 - self.beta2**self.t)
            v_b_hat = self.v_b[i] / (1 - self.beta2**self.t)

            # D. ACTUALIZACIÓN FINAL
            # Dividimos por la raíz de v_hat para normalizar el tamaño del paso.
            # Los pesos con gradientes volátiles se frenan, los estables se aceleran.
            layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
            layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
