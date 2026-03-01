import numpy as np

# ==============================================================================
# CAPÍTULO: DROPOUT (REGULARIZACIÓN POR ABANDONO)
# ==============================================================================
# 
# Problema que resuelve: 
#   El "Overfitting" (Sobreajuste). Cuando una red es muy grande, las neuronas 
#   pueden desarrollar una "co-dependencia" (co-adaptation), aprendiendo a 
#   corregir los errores de otras en lugar de aprender características útiles.
# 
# Rol en el Pipeline:
#   Actúa como un "entrenador exigente". Durante el entrenamiento, apaga 
#   aleatoriamente un porcentaje de neuronas, obligando a la red a ser redundante 
#   y robusta.
# 
# Intuición Biológica/Social:
#   Si quieres que un equipo sea resiliente, asegúrate de que cada miembro pueda 
#   hacer el trabajo del otro. Si un día alguien falta, el sistema sigue operando.
# ==============================================================================

class Dropout:
    def __init__(self, p=0.5):
        """
        Inicialización de Dropout.
        
        Args:
            p (float): Probabilidad de APAGAR una neurona (Rango 0 a 1).
                       p=0.5 significa que el 50% de las neuronas se ignoran.
        """
        self.p = p           # Tasa de abandono
        self.mask = None     # Máscara binaria (quién vive y quién muere)
        self.training = True # Estado: True en entrenamiento, False en test

    def forward(self, A):
        """
        Forward Pass: Aplica el filtrado aleatorio.
        
        Args:
            A (np.ndarray): Activaciones de entrada (Batch_Size, Features).
        """
        # ----------------------------------------------------------------------
        # NOTA DEL PROFESOR: Inverted Dropout
        # ----------------------------------------------------------------------
        # En inferencia (test), todas las neuronas están encendidas. 
        # Si en entrenamiento apagamos el 50%, la suma total en test será 
        # el doble de grande. Para evitar esto, escalamos las activaciones 
        # en entrenamiento por 1/(1-p). Así, el valor esperado de la suma 
        # se mantiene constante.
        # ----------------------------------------------------------------------
        
        if not self.training:
            # En modo evaluación, simplemente dejamos pasar la señal
            return A
        
        # Generar máscara de Bernoulli: 
        # 1. np.random.rand crea valores entre 0 y 1.
        # 2. Comparamos con p para decidir quién sobrevive.
        # 3. Dividimos por (1-p) para compensar la pérdida de energía (Inverted Dropout).
        self.mask = (np.random.rand(*A.shape) > self.p) / (1.0 - self.p)
        
        # Aplicamos la máscara elemento a elemento
        return A * self.mask

    def backward(self, dLdA_out):
        """
        Backward Pass: Solo fluye gradiente por donde pasó la señal.
        
        Args:
            dLdA_out (np.ndarray): Gradiente del error respecto a la salida.
        """
        if not self.training:
            return dLdA_out
        
        # ----------------------------------------------------------------------
        # REGLA DE LA CADENA EN DROPOUT:
        # Puesto que Y = A * mask, la derivada dY/dA es simplemente la máscara.
        # Por lo tanto, dL/dA = dL/dY * mask. 
        # Las neuronas que estuvieron apagadas no reciben ni transmiten error.
        # ----------------------------------------------------------------------
        return dLdA_out * self.mask
