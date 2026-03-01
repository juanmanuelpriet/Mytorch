import numpy as np

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, A):
        if not self.training:
            return A
        
        # Generar máscara bernoulli
        self.mask = (np.random.rand(*A.shape) > self.p) / (1.0 - self.p)
        return A * self.mask

    def backward(self, dLdA_out):
        if not self.training:
            return dLdA_out
        
        return dLdA_out * self.mask
