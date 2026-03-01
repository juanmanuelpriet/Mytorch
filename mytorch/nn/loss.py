import numpy as np

# ==============================================================================
# CAPÍTULO: FUNCIONES DE PÉRDIDA (LOSS FUNCTIONS / CRITERIA)
# ==============================================================================
# 
# Problema que resuelve: 
#   ¿Cómo sabemos si la red está aprendiendo? Necesitamos una métrica escalar 
#   que cuantifique la "distancia" entre lo que la red predice (A) y la 
#   realidad (Y). Esta es la "brújula" que guía los gradientes.
# 
# Rol en el Pipeline:
#   Es el punto final del Forward Pass y el punto de partida (Génesis) del 
#   Backward Pass.
# ==============================================================================

class MSELoss:
    """
    Mean Squared Error (Error Cuadrático Medio).
    Ideal para problemas de REGRESIÓN (predecir valores continuos).
    
    Fórmula: L = (1 / 2NC) * sum( (A - Y)^2 )
    
    Donde N es el batch size y C es el número de componentes de salida.
    El factor 2 en el denominador es un "truco matemático" para que al 
    derivar se cancele con el exponente cuadrado.
    """

    def forward(self, A, Y):
        """
        Calcula el error.
        
        Args:
            A (np.ndarray): Predicciones del modelo (N, C).
            Y (np.ndarray): Etiquetas reales (N, C).
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]  
        self.C = A.shape[1]  
        
        # Error al cuadrado individual
        se = (A - Y) * (A - Y) 
        # Suma de errores al cuadrado (Sum of Squared Errors)
        sse = np.sum(se) 
        # Promedio total (Normalizado por N y C)
        mse = sse / (2 * self.N * self.C) 

        return mse

    def backward(self):
        """
        Derivada del MSE respecto a las predicciones A.
        dL/dA = (A - Y) / (N * C)
        
        EXPLICACIÓN:
        La derivada de (A-Y)^2 es 2(A-Y). Al dividir por 2NC, 
        el 2 se cancela y queda el término lineal.
        """
        dLdA = (self.A - self.Y) / (self.N * self.C)
        return dLdA


class CrossEntropyLoss:
    """
    Cross-Entropy Loss (Entropía Cruzada).
    La reina de la CLASIFICACIÓN. 
    
    Problema: 
    MSE no castiga fuertemente las predicciones que están "muy seguras pero erróneas".
    La Entropía Cruzada usa logaritmos para penalizar exponencialmente el error,
    acelerando el aprendizaje cuando la red está muy equivocada.
    
    Fórmula: L = - (1/N) * sum_batch( sum_clases( Y_ij * log(Softmax(A_ij)) ) )
    """

    def forward(self, A, Y):
        """
        Args:
            A (np.ndarray): Logits (salida lineal del modelo) (N, C).
            Y (np.ndarray): One-hot encoding de las etiquetas (N, C).
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]  
        self.C = A.shape[1]  

        # ----------------------------------------------------------------------
        # NOTA DEL PROFESOR: Estabilidad de Softmax
        # ----------------------------------------------------------------------
        # Aquí calculamos softmax internamente. En implementaciones de producción 
        # (como PyTorch), se suele usar LogSoftmax para mejorar la estabilidad 
        # numérica y evitar valores ínfimos que el computador redondee a cero.
        # ----------------------------------------------------------------------
        
        # Log-Sum-Exp trick implícito (E de A)
        expA = np.exp(A - np.max(A, axis=1, keepdims=True))
        self.softmax = expA / np.sum(expA, axis=1, keepdims=True)
        
        # Cálculo de la Entropía Cruzada: -Y * log(P)
        # Solo sumamos el logaritmo de la probabilidad de la clase correcta (donde Y=1)
        crossentropy = -Y * np.log(self.softmax + 1e-15) # +eps para evitar log(0)
        sum_crossentropy = np.sum(crossentropy)  
        L = sum_crossentropy / self.N

        return L

    def backward(self):
        """
        Derivada de CrossEntropy + Softmax.
        dL/dA = Softmax(A) - Y
        
        MAGIA MATEMÁTICA:
        Aunque las derivadas individuales de Softmax y Log son complejas,
        al combinarlas resulta en una resta elegantemente simple. 
        El gradiente es simplemente la "distancia" entre la probabilidad 
        predicha y el objetivo 1.
        """
        dLdA = (self.softmax - self.Y) / self.N
        return dLdA
