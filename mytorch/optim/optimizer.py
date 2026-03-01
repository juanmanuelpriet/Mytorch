import numpy as np

# ==============================================================================
# CAPÍTULO: LA BASE DE LA OPTIMIZACIÓN
# ==============================================================================
# 
# Problema que resuelve: 
#   Una vez que tenemos los gradientes (dirección del error), ¿cómo actualizamos 
#   los millones de parámetros de forma eficiente? El Optimizador encapsula 
#   la lógica de actualización (Update Rule).
# 
# Rol en el Pipeline:
#   Es el "mecánico" de la red. Entra en acción después del Backward Pass.
# ==============================================================================

class Optimizer:
    """
    Clase Base Abstracta para todos los optimizadores.
    Gestiona la referencia al modelo y sus parámetros entrenables.
    """
    def __init__(self, model, lr):
        """
        Args:
            model (Sequential): El modelo a optimizar.
            lr (float): Learning Rate (Tasa de aprendizaje / Tamaño del paso).
        """
        self.model = model
        self.lr = lr
        
        # ----------------------------------------------------------------------
        # NOTA DEL PROFESOR: Filtrado de Capas
        # ----------------------------------------------------------------------
        # No todas las capas tienen parámetros (ej: ReLU, Dropout). 
        # Aquí recolectamos solo aquellas que tienen el atributo 'W' (pesos)
        # para no perder tiempo iterando sobre activaciones puras.
        # ----------------------------------------------------------------------
        self.layers = [l for l in model.layers if hasattr(l, 'W')]

    def zero_grad(self):
        """
        Resetea los gradientes acumulados.
        En PyTorch y en esta librería, los gradientes se suman o sobrescriben.
        Es vital llamar a esto antes de un nuevo backward para no "arrastrar"
        información del batch anterior.
        """
        for layer in self.layers:
            if hasattr(layer, 'dLdW') and layer.dLdW is not None:
                layer.dLdW.fill(0.0)
            if hasattr(layer, 'dLdb') and layer.dLdb is not None:
                layer.dLdb.fill(0.0)

    def step(self):
        """
        Método abstracto que implementará cada descendiente (SGD, Adam, etc.)
        """
        raise NotImplementedError
