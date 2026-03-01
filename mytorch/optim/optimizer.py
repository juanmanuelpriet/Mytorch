import numpy as np


class Optimizer:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
        # Filtramos solo las capas que tienen parámetros entrenables
        self.layers = [l for l in model.layers if hasattr(l, 'W')]

    def zero_grad(self):
        # En esta librería, los gradientes se sobrescriben en cada backward,
        # pero es bueno tener la interfaz por si acaso.
        pass

    def step(self):
        raise NotImplementedError
