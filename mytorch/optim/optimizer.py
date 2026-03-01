import numpy as np


class Optimizer:
    def __init__(self, model, lr):
        self.model = model
        self.lr = lr
        # Filtramos solo las capas que tienen parámetros entrenables
        self.layers = [l for l in model.layers if hasattr(l, 'W')]

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, 'dLdW') and layer.dLdW is not None:
                layer.dLdW.fill(0.0)
            if hasattr(layer, 'dLdb') and layer.dLdb is not None:
                layer.dLdb.fill(0.0)

    def step(self):
        raise NotImplementedError
