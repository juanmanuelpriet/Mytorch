import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):

        """
        Inicializa los pesos (weights) y los sesgos (biases) con ceros.
        Revisa la función np.zeros.
        Lee el enunciado/la guía (writeup) para identificar las formas (shapes) correctas de cada uno.
        """

        self.W = np.zeros((out_features,in_features))  # matriz de pesos dim: Cout x Cin
        self.b = np.zeros((out_features,1)) # vector de sesgo dim: Cout x 1

        self.debug = debug

    def forward(self, A):
        """
        :param A: Entrada a la capa lineal con forma (N, C0)
        :return: Salida Z de la capa lineal con forma (N, C1)
        Lee el enunciado/la guía (writeup) para detalles de implementación.
        """

        self.A = A # Entradas dim: N x Cin
        self.N = A.shape[0] # almacena el tamaño de el batch

        self.Ones = np.ones((self.N,1)) # vector de unos para sumar el sesgo

        Z = self.A @ self.W.T + self.Ones @ self.b.T # salida de la red dim: N x Cout

        return Z

    def backward(self, dLdZ):
        
        #calculos cuaderno
        dZdA = self.W.T
        dZdW = self.A
        dZdb = self.Ones

        #remplazos con ecuaciones
        dLdA = dLdZ @ dZdA.T
        dLdW = dLdZ.T @ dZdW
        dLdb = dLdZ.T @ dZdb
        
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdb = dZdb
            self.dLdA = dLdA

        return dLdA
