import numpy as np

# ==============================================================================
# CAPÍTULO: BATCH NORMALIZATION (NORMALIZACIÓN POR LOTES)
# ==============================================================================
# 
# Problema que resuelve: 
#   El "Internal Covariate Shift". A medida que los pesos cambian, la distribución 
#   de las entradas a las capas profundas cambia constantemente, obligando a 
#   la red a "perseguir un blanco móvil". 
# 
# Rol en el Pipeline:
#   Actúa como un regulador de flujo. Mantiene las activaciones centradas en cero 
#   y con varianza unitaria, lo que permite usar tasas de aprendizaje más altas 
#   y disminuye la dependencia de la inicialización.
# 
# Resumen Matemático:
#   1. Media: mu = 1/N * sum(Z)
#   2. Varianza: sigma^2 = 1/N * sum((Z - mu)^2)
#   3. Normalización: Z_hat = (Z - mu) / sqrt(sigma^2 + eps)
#   4. Escalamiento y Desplazamiento: Y = gamma * Z_hat + beta
# ==============================================================================

class BatchNorm1d:
    def __init__(self, num_features, alpha=0.9):
        """
        Inicialización de BatchNorm.
        
        Args:
            num_features (int): Número de neuronas de entrada (columnas).
            alpha (float): Factor de momentum para promedios móviles (0 a 1).
        """
        self.alpha = alpha # Momentum (Inercia para estadísticas de inferencia)
        self.eps = 1e-8    # Estabilidad numérica (evita división por cero)

        # ----------------------------------------------------------------------
        # PARÁMETROS ENTRENABLES (gamma y beta)
        # ----------------------------------------------------------------------
        # BW (gamma): Permite a la red recuperar la escala original si es necesario.
        # Bb (beta): Permite a la red recuperar el desplazamiento original.
        # Se inicializan en 1 y 0 respectivamente para que inicialmente no alteren Z_hat.
        # ----------------------------------------------------------------------
        self.BW = np.ones((1, num_features))   # Gamma
        self.Bb = np.zeros((1, num_features))  # Beta

        # Gradientes para el optimizador
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # ----------------------------------------------------------------------
        # ESTADÍSTICAS PARA INFERENCIA (Running Stats)
        # ----------------------------------------------------------------------
        # Durante el testeo no tenemos un "batch" representativo. Usamos la media
        # y varianza que fuimos calculando durante todo el entrenamiento.
        # ----------------------------------------------------------------------
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        Forward Pass: Normaliza las pre-activaciones.
        
        Args:
            Z (np.ndarray): Pre-activaciones (N, F).
            eval (bool): Si es True, usa estadísticas guardadas (Modo Inferencia).
        """
        self.Z = Z
        self.N = Z.shape[0] # Tamaño del Batch

        # 1. Calculamos la media y varianza del batch actual por columna
        self.M = np.mean(Z, axis=0) # Media mu
        self.V = np.var(Z, axis=0)  # Varianza sigma^2

        if not eval:
            # ------------------------------------------------------------------
            # MODO ENTRENAMIENTO (TRAINING)
            # ------------------------------------------------------------------
            
            # Paso A: Centrar y Escalar (Normalización Standard)
            # NZ = (Z - mu) / std
            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)

            # Paso B: Transformación Afín (Aprender la distribución ideal)
            # BZ = gamma * NZ + beta
            self.BZ = self.BW * self.NZ + self.Bb

            # Paso C: Actualización de estadísticas acumuladas (Momentum)
            # Utilizamos un promedio móvil exponencial para la posteridad.
            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V

        else:
            # ------------------------------------------------------------------
            # MODO EVALUACIÓN (INFERENCE)
            # ------------------------------------------------------------------
            # NO usamos el batch actual para normalizar, usamos lo que aprendimos.
            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            self.BZ = self.BW * self.NZ + self.Bb

        return self.BZ

    def backward(self, dLdBZ):
        """
        Backward Pass: Derivada de una normalización dependiente.
        
        NOTA DEL PROFESOR: Esta es la derivada más difícil de implementar.
        Debido a que mu y sigma dependen de todos los ejemplos del batch (Z),
        el gradiente fluye a través de mu, a través de sigma y directamente.
        
        Fórmula Simplificada:
        dL/dZ = (gamma / (N * sigma)) * (N * dL/dY - sum(dL/dY) - Z_hat * sum(dL/dY * Z_hat))
        """
        
        # 1. Gradiente del Sesgo Aprendido (Beta)
        # dL/dBeta = sum(dL/dY)
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)

        # 2. Gradiente del Escala Aprendida (Gamma)
        # dL/dGamma = sum(dL/dY * Z_hat)
        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)

        # 3. Gradiente respecto a la Entrada Z
        # Aplicamos la expansión de la regla de la cadena para BatchNorm.
        inv_std = 1.0 / np.sqrt(self.V + self.eps)
        
        # ----------------------------------------------------------------------
        # DESGLOSE DE LA FÓRMULA:
        # Term 1: N * dLdBZ -> Propagación directa a través de NZ.
        # Term 2: self.dLdBb -> Flujo a través de la MEDIA M.
        # Term 3: self.NZ * self.dLdBW -> Flujo a través de la VARIANZA V.
        # ----------------------------------------------------------------------
        dLdZ = (self.BW * inv_std / self.N) * (
            self.N * dLdBZ - self.dLdBb - self.NZ * self.dLdBW
        )

        return dLdZ
