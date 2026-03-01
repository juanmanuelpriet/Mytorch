import numpy as np
import math as mt
from scipy.special import erf

# ==============================================================================
# CAPÍTULO: FUNCIONES DE ACTIVACIÓN (NO LINEALIDAD)
# ==============================================================================
# 
# Problema que resuelven: 
#   Si solo tuviéramos capas lineales, el modelo completo sería equivalente a 
#   una sola matriz gigante (Producto de matrices). No podríamos aprender 
#   nada más complejo que una línea recta. Las activaciones introducen 
#   "curvatura" permitiendo aproximar cualquier función.
# 
# Rol en el Pipeline:
#   Se aplican inmediatamente después de la suma ponderada (Z). Deciden si 
#   una neurona debe "disparar" o no.
# ==============================================================================

class Identity:
    """
    Función Identidad: f(z) = z. 
    Usada principalmente en capas de salida para problemas de regresión.
    """
    def forward(self, Z):
        self.A = Z
        return self.A

    def backward(self, dLdA):
        # La derivada de z respecto a z es 1.
        return dLdA


class Sigmoid:
    """
    Sigmoide: f(z) = 1 / (1 + e^-z)
    Mapea cualquier valor real al rango (0, 1). 
    Intuición: Representa una probabilidad.
    """
    def forward(self, Z):
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        """
        Derivada: f'(z) = f(z) * (1 - f(z))
        """
        # dL/dZ = dL/dA * dA/dZ
        return dLdA * self.A * (1 - self.A)


class Tanh:
    """
    Tangente Hiperbólica: f(z) = tanh(z)
    Mapea al rango (-1, 1). Centrada en cero, lo que ayuda a la convergencia.
    """
    def forward(self, Z):
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dLdA):
        """
        Derivada: f'(z) = 1 - tanh^2(z)
        """
        return dLdA * (1 - self.A**2)


class ReLU:
    """
    Rectified Linear Unit: f(z) = max(0, z)
    La función más popular en Deep Learning. Elimina el gradiente desvaneciente
    para valores positivos.
    """
    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dLdA):
        """
        Derivada: 1 si z > 0, de lo contrario 0.
        """
        # Multiplicación elemento a elemento por la máscara booleana
        return dLdA * (self.Z > 0)


class GeLU:
    """
    Gaussian Error Linear Unit: f(z) = z * P(X <= z)
    Variante moderna de ReLU usada en BERT y Transformers. 
    Es una curva suave que permite gradientes pequeños para valores negativos.
    """
    def forward(self, Z):
        self.Z = Z
        # Usamos la función de error (erf) para aproximar la CDF de la Normal
        self.Phi = 0.5 * (1 + erf(Z / np.sqrt(2)))
        self.A = Z * self.Phi
        return self.A

    def backward(self, dLdA):
        """
        Derivada compleja: f'(z) = Phi(z) + z * phi(z)
        donde phi es la PDF de la normal standard.
        """
        # PDF de la distribución normal
        phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * self.Z**2)
        dAdZ = self.Phi + self.Z * phi
        return dLdA * dAdZ


class SoftMax:
    """
    Softmax: Multi-clase Probabilidad.
    Convierte un vector de 'logits' en una distribución de probabilidad que suma 1.
    """
    def forward(self, Z):
        # Truco de estabilidad numérica: restamos el máximo para evitar exp(inf)
        E = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.A = E / np.sum(E, axis=1, keepdims=True)
        return self.A

    def backward(self, dLdA):
        """
        NOTA DEL PROFESOR: El Jacobiano de Softmax.
        Softmax es una función acoplada (todas las salidas dependen de todas las entradas).
        La derivada dL/dZ_i = A_i * (dL/dA_i - sum_j(dL/dA_j * A_j))
        Aquí implementamos esta fórmula vectorizada para eficiencia en batches.
        """
        # Producto punto dLdA * self.A sumado por filas
        dot_product = np.sum(dLdA * self.A, axis=1, keepdims=True)
        return self.A * (dLdA - dot_product)
