import numpy as np
import math as mt
from scipy.special import erf


class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ


class Sigmoid:
    """
    En las mismas líneas que arriba:
    Define la función 'forward'
    Define la función 'backward'
    Lee el documento/explicación para más detalles sobre la sigmoide.
    """
    def forward(self,Z):

        self.A = 1/(1+np.exp(-Z)) # uso funcion sigmoide

        return self.A
    def backward(self):

        dAdZ = self.A * (1-self.A) # calculos en cuaderno

        return dAdZ


class Tanh:
    """
     En las mismas líneas que arriba:
    Define la función 'forward'
    Define la función 'backward'
    Lee el documento/explicación para más detalles sobre la Tanh.
    """
    def forward(self,Z):

        self.A = (np.exp(Z) - np.exp(-Z))/(np.exp(Z)+np.exp(-Z)) # uso funcion tanh tambien puede ser np.tanh

        return self.A
    def backward(self):

        dAdZ = 1-(self.A * self.A)

        return dAdZ


class ReLU:
    """
     En las mismas líneas que arriba:
    Define la función 'forward'
    Define la función 'backward'
    Lee el documento/explicación para más detalles sobre la ReLU.
    """
    def forward(self,Z):
        
        self.A = np.maximum(0,Z)

        return self.A
    
    def backward(self):

        dAdZ = np.where(self.A > 0, 1, 0).astype('f')

        return dAdZ
    

class GeLU:

    def forward(self,Z):

        self.Z = Z
        self.Phi = 0.5 * (1 + erf(Z / np.sqrt(2)))   
        self.A = Z * self.Phi 

        return self.A
    
    def backward(self):

        phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * self.Z**2)

        dAdZ = self.Phi + self.Z * phi 

        return dAdZ
    

class SoftMax:

    def forward(self,Z):

        E = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        self.A = E / np.sum(E, axis=1, keepdims=True)

        return self.A
    
    def backward(self):
        # self.A tiene shape (batch_size, n)
        batch_size, n = self.A.shape
        
        # Inicializamos el array de salida
        dAdZ = np.zeros((batch_size, n, n))
    
        for i in range(batch_size):
            a = self.A[i]  # vector (n,)
            # J = diag(a) - a @ a.T
            dAdZ[i] = np.diag(a) - np.outer(a, a)
    
        return dAdZ


