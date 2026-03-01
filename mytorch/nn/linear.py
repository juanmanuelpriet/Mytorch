import numpy as np

# ==============================================================================
# CAPÍTULO: CAPAS LINEALES (FULLY CONNECTED / DENSE LAYERS)
# ==============================================================================
# 
# Esta clase representa la unidad fundamental de transformación en redes neuronales.
# 
# Problema que resuelve: 
#   En la naturaleza, los datos rara vez están alineados. Necesitamos proyectar 
#   nuestros inputs a nuevos espacios dimensionales donde las características 
#   puedan ser separadas o combinadas. La capa lineal realiza una transformación 
#   afín (rotación, escalado y traslación).
# 
# Rol en el Pipeline:
#   Actúa como el "músculo" que aprende. Es aquí donde residen los parámetros 
#   W (pesos) y b (sesgos) que el optimizador ajustará.
# 
# Conexión con el sistema:
#   Recibe activaciones de la capa anterior y entrega una suma ponderada (Z) 
#   a la función de activación.
# ==============================================================================

class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        Inicialización de la capa. 
        
        Args:
            in_features (int): Dimensión de entrada (ej: 784 para MNIST plano).
            out_features (int): Número de neuronas/unidades de salida.
            debug (bool): Flag para imprimir estados internos.
            
        Atributos Clave:
            self.W (Matrix): Matriz de pesos de forma (out_features, in_features).
            self.b (Vector): Vector de sesgo de forma (out_features, 1).
        """
        
        # ----------------------------------------------------------------------
        # NOTA DEL PROFESOR: Inicialización de Pesos (He / Xavier)
        # ----------------------------------------------------------------------
        # Si inicializamos con ceros, todas las neuronas aprenderán lo mismo 
        # (simetría). Si usamos valores muy grandes, las activaciones explotarán.
        # Usamos np.sqrt(2.0 / in_features) para mantener la varianza de las 
        # señales constante a través de las capas (ideal para ReLU/GeLU).
        # ----------------------------------------------------------------------
        limit = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(out_features, in_features) * limit
        self.b = np.zeros((out_features, 1))

        # Gradientes: Aquí guardaremos las derivadas para que el optimizador las use
        self.dLdW = None # Derivada del Error respecto a los Pesos
        self.dLdb = None # Derivada del Error respecto al Sesgo
        self.debug = debug

    def forward(self, A):
        """
        Paso hacia adelante (Forward Pass).
        Calcula la pre-activación Z = A · W^T + b^T
        
        Args:
            A (np.ndarray): Activaciones de la capa anterior. 
                            Shape: (Batch_Size, in_features)
                            
        Returns:
            Z (np.ndarray): Suma ponderada resultante.
                            Shape: (Batch_Size, out_features)
        """
        # Guardamos A para el cálculo del gradiente en el backward
        self.A = A
        # N es el tamaño del batch, útil para promediar gradientes
        self.N = A.shape[0]
        
        # ----------------------------------------------------------------------
        # INTUICIÓN GEOMÉTRICA:
        # Cada neurona en esta capa define un hiperplano en el espacio de entrada.
        # W controla la orientación del plano y b controla su desplazamiento.
        # El producto punto mide qué tanto se alinea la entrada con el "patrón"
        # que cada neurona busca.
        # ----------------------------------------------------------------------
        # Z = A @ W.T + b.T
        # Implementamos broadcasting para sumar b (vector) a cada fila de A@W.T
        Z = self.A @ self.W.T + self.b.T
        return Z

    def backward(self, dLdZ):
        """
        Paso hacia atrás (Backpropagation).
        Aplica la Regla de la Cadena para propagar el error y calcular gradientes.
        
        Args:
            dLdZ (np.ndarray): Gradiente del error respecto a la salida Z.
                               Shape: (Batch_Size, out_features)
                               
        Returns:
            dLdA (np.ndarray): Gradiente del error respecto a la entrada A.
                               Shape: (Batch_Size, in_features)
        """
        
        # 1. Gradiente respecto a las activaciones (para la capa anterior)
        # dL/dA = dL/dZ * dZ/dA = dL/dZ * W
        dLdA = dLdZ @ self.W
        
        # 2. Gradiente respecto a los Pesos W
        # dL/dW = (dL/dZ)^T * A / N
        # ----------------------------------------------------------------------
        # EXPLICACIÓN MATEMÁTICA:
        # Según la regla de la cadena, dL/dW = dL/dZ * dZ/dW.
        # Como Z = A*W^T, entonces dZ/dW es proporcional a A.
        # Hacemos el producto externo y promediamos por N para que el aprendizaje
        # sea independiente del tamaño del batch.
        # ----------------------------------------------------------------------
        self.dLdW = dLdZ.T @ self.A / self.N
        
        # 3. Gradiente respecto al Sesgo b
        # dL/db = sum(dL/dZ) / N
        # ----------------------------------------------------------------------
        # EL SESGO Y EL BROADCASTING:
        # Como b se sumó a todos los ejemplos del batch, el gradiente total
        # es la suma de los gradientes de cada ejemplo.
        # ----------------------------------------------------------------------
        self.dLdb = np.sum(dLdZ, axis=0, keepdims=True).T / self.N
        
        return dLdA
