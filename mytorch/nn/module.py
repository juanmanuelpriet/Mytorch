import numpy as np

# ==============================================================================
# CAPÍTULO: CONTENEDORES Y ORQUESTACIÓN (MODULES)
# ==============================================================================
# 
# Problema que resuelve: 
#   Una red neuronal real tiene decenas de capas. Gestionar manualmente el 
#   paso de datos de una a otra y revertir el orden para el gradiente es 
#   propenso a errores. Necesitamos una estructura que abstraiga la red 
#   como un "todo".
# 
# Rol en el Pipeline:
#   Es el "Director de Orquesta". Organiza el flujo de información secuencial.
# ==============================================================================

class Sequential:
    """
    Contenedor secuencial de capas.
    Permite tratar a una cascada de capas como un único objeto.
    """
    def __init__(self, *layers):
        """
        Args:
            *layers: Lista variable de objetos capa (Linear, ReLU, etc.)
        """
        self.layers = list(layers)

    def forward(self, x):
        """
        Paso adelante en cascada: Output_n = f_n(f_n-1(...f_1(x)...))
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        """
        Paso atrás en cascada: Aplicamos la Regla de la Cadena de fin a principio.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def __call__(self, x):
        """
        Azúcar sintáctico: permite llamar al modelo como una función.
        Ej: prediccion = modelo(input)
        """
        return self.forward(x)
