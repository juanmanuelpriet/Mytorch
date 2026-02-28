import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU,Tanh,GeLU


class MLP0:
    # Define una clase "MLP0": una red MLP mínima (una capa lineal + una activación)

    def __init__(self, debug=False):
        # Constructor. Se ejecuta cuando creas MLP0().
        # debug controla si guardamos variables internas para inspección.

        self.layers = [Linear(2, 3)]
        # Lista de capas "paramétricas". Aquí solo hay UNA capa lineal:
        # Linear(in_features=2, out_features=3)
        # Si A0 tiene shape (N, 2), entonces Z0 tendrá shape (N, 3).

        self.f = [ReLU()]
        # Lista de funciones de activación. Aquí solo hay UNA activación ReLU.
        # Se aplicará después de la capa lineal. A1 tendrá shape (N, 3).

        self.debug = debug
        # Guarda la bandera debug para saber si debemos guardar intermedios.

    def forward(self, A0):
        # Forward pass: computa la salida de la red dado el input A0.
        # A0: activaciones/entrada con shape (N, 2)

        Z0 = self.layers[0].forward(A0)  # TODO
        # Aplica la capa lineal al input.
        # Matemáticamente: Z0 = A0 @ W^T + b
        # Z0 shape: (N, 3)

        A1 = self.f[0].forward(Z0)  # TODO
        # Aplica ReLU a Z0.
        # ReLU(x) = max(0, x) elemento a elemento
        # A1 shape: (N, 3)

        if self.debug:
            # Si debug está activo, guardamos intermedios para revisar luego.

            self.Z0 = Z0
            # Guarda el pre-activación (salida de la lineal antes de ReLU).

            self.A1 = A1
            # Guarda la salida final del forward.

        return A1
        # Devuelve la salida del modelo (predicción/representación).

    def backward(self, dLdA1):
        # Backward pass: propaga gradientes desde la salida hacia la entrada.
        # dLdA1: gradiente de la pérdida respecto a la salida A1.
        # shape: (N, 3)

        dA1dZ0 = self.f[0].backward()
        # Obtiene la derivada local de la activación: dA1/dZ0.
        # En ReLU típicamente es:
        # 1 si Z0 > 0, 0 si Z0 <= 0 (elemento a elemento)

        dLdZ0 = dLdA1 * dA1dZ0
        # Regla de la cadena:
        # dL/dZ0 = dL/dA1 ⊙ dA1/dZ0
        # ⊙ es multiplicación elemento a elemento.
        # shape: (N, 3)

        dLdA0 = self.layers[0].backward(dLdZ0)
        # Propaga el gradiente a través de la capa lineal.
        # Devuelve dL/dA0 con shape (N, 2).
        #
        # Internamente, Linear.backward(dLdZ0) típicamente:
        # - Calcula y guarda dL/dW y dL/db
        # - Devuelve dL/dA0 = dLdZ0 @ W   (según convención de shapes)

        if self.debug:
            # Si debug está activo, guardamos gradientes intermedios.

            self.dA1dZ0 = dA1dZ0
            # Guarda derivada local de ReLU.

            self.dLdZ0 = dLdZ0
            # Guarda gradiente respecto al pre-activación Z0.

            self.dLdA0 = dLdA0
            # Guarda gradiente respecto a la entrada A0.

        return dLdA0
        # Devuelve el gradiente respecto a la entrada.
        # Esto sirve si la red fuera parte de un pipeline mayor o para verificación.
class MLP1:
    # Define una MLP con 1 capa oculta: Linear(2→3) + ReLU + Linear(3→2) + ReLU

    def __init__(self, debug=False):
        """
        Initialize 2 linear layers. Layer 1 of shape (2,3) and Layer 2 of shape (3, 2).
        Use Relu activations for both the layers.
        Implement it on the same lines(in a list) as MLP0
        """
        # Docstring: explica que hay 2 capas lineales:
        #   - Capa 0: 2 -> 3 (entrada a hidden)
        #   - Capa 1: 3 -> 2 (hidden a salida)
        # y que se usa ReLU como activación en ambas etapas.

        self.layers = [Linear(2, 3), Linear(3, 2)]  # TODO
        # Lista con las capas lineales (capas con parámetros):
        # layers[0] transforma A0 de shape (N,2) a Z0 de shape (N,3)
        # layers[1] transforma A1 de shape (N,3) a Z1 de shape (N,2)

        self.f = [ReLU(), ReLU()]  # TODO
        # Lista con las activaciones:
        # f[0] se aplica a Z0 para producir A1 (shape (N,3))
        # f[1] se aplica a Z1 para producir A2 (shape (N,2))
        #
        # Nota: en algunos diseños, la última activación podría ser Identity/Softmax,
        # pero aquí te piden ReLU en ambas.

        self.debug = debug
        # Flag para guardar intermedios y gradientes y poder depurar.

    def forward(self, A0):
        # Forward pass: calcula la salida A2 del modelo dado el input A0.
        # A0 shape: (N, 2)

        Z0 = self.layers[0].forward(A0) # TODO
        # Primera capa lineal:
        # Z0 = A0 @ W0^T + b0
        # Z0 shape: (N, 3)

        A1 = self.f[0].forward(Z0) # TODO
        # Primera activación ReLU:
        # A1 = ReLU(Z0) = max(0, Z0) elemento a elemento
        # A1 shape: (N, 3)

        Z1 = self.layers[1].forward(A1) # TODO
        # Segunda capa lineal:
        # Z1 = A1 @ W1^T + b1
        # Z1 shape: (N, 2)

        A2 = self.f[1].forward(Z1) # TODO
        # Segunda activación ReLU:
        # A2 = ReLU(Z1)
        # A2 shape: (N, 2)
        #
        # Esta A2 es la salida final del modelo.

        if self.debug:
            # Si debug está activo, guardamos valores intermedios del forward
            # para inspección o para chequear gradientes.

            self.Z0 = Z0
            # Pre-activación de la primera capa (antes de ReLU)

            self.A1 = A1
            # Post-activación de la primera etapa (salida de ReLU0)

            self.Z1 = Z1
            # Pre-activación de la segunda capa (antes de ReLU)

            self.A2 = A2
            # Salida final (post-activación ReLU1)

        return A2
        # Devuelve la salida final del forward.

    def backward(self, dLdA2):
        # Backward pass: propaga gradientes desde la salida A2 hasta la entrada A0.
        # dLdA2 = ∂L/∂A2 (gradiente de la pérdida respecto a la salida)
        # dLdA2 shape: (N, 2)

        dA2dZ1 = self.f[1].backward()            # derivada local de la activación
        # Calcula derivada local de la activación final:
        # dA2/dZ1 (para ReLU: 1 si Z1>0, 0 si Z1<=0)
        #
        # OJO: como no le pasas Z1, ReLU.backward() debe usar Z1 guardado internamente
        # (por ejemplo, guardó una máscara en forward).

        dLdZ1 = dLdA2 * dA2dZ1                # regla de la cadena
        # Regla de la cadena:
        # ∂L/∂Z1 = (∂L/∂A2) ⊙ (∂A2/∂Z1)
        # ⊙ es multiplicación elemento a elemento
        # dLdZ1 shape: (N, 2)

        dLdA1 = self.layers[1].backward(dLdZ1)   # propagar a capa lineal
        # Backprop a través de la segunda capa lineal (Linear(3→2)):
        # Devuelve ∂L/∂A1 (gradiente respecto a la entrada de esta capa)
        # dLdA1 shape: (N, 3)
        #
        # Internamente suele calcular y guardar gradientes:
        # - dLdW1, dLdb1 (para actualizar parámetros)
        # - y devuelve dLdA1

        dA1dZ0 = self.f[0].backward()            # derivada local de la activación
        # Derivada local de la primera ReLU:
        # dA1/dZ0 (1 si Z0>0, 0 si Z0<=0)
        # Igual que antes: requiere que ReLU0 haya guardado Z0 o una máscara en forward.

        dLdZ0 = dLdA1 * dA1dZ0                # regla de la cadena
        # Regla de la cadena para la primera etapa:
        # ∂L/∂Z0 = (∂L/∂A1) ⊙ (∂A1/∂Z0)
        # dLdZ0 shape: (N, 3)

        dLdA0 = self.layers[0].backward(dLdZ0)   # propagar a capa lineal
        # Backprop a través de la primera capa lineal (Linear(2→3)):
        # Devuelve ∂L/∂A0 (gradiente respecto a la entrada original)
        # dLdA0 shape: (N, 2)
        #
        # Internamente calcula/guarda:
        # - dLdW0, dLdb0
        # - y devuelve dLdA0

        if self.debug:
            # Si debug está activo, guardamos intermedios del backward.

            self.dA2dZ1 = dA2dZ1
            # Guarda derivada local de ReLU final.

            self.dLdZ1 = dLdZ1
            # Guarda gradiente respecto a Z1 (pre-activación de la 2da capa).

            self.dLdA1 = dLdA1
            # Guarda gradiente respecto a A1 (salida de la 1ra ReLU / entrada a 2da lineal).

            self.dA1dZ0 = dA1dZ0
            # Guarda derivada local de ReLU0.

            self.dLdZ0 = dLdZ0
            # Guarda gradiente respecto a Z0 (pre-activación de la 1ra capa).

            self.dLdA0 = dLdA0
            # Guarda gradiente final respecto a la entrada A0.

        return dLdA0
        # Devuelve ∂L/∂A0.
        # Esto sirve para encadenar con capas anteriores o para pruebas/chequeos.


class MLP4:
    def __init__(self, debug=False):
        """
        Initialize 4 hidden layers and an output layer of shape below:
        Layer1 (2, 4),
        Layer2 (4, 8),
        Layer3 (8, 8),
        Layer4 (8, 4),
        Output Layer (4, 2)

        Refer the diagrmatic view in the writeup for better understanding.
        Use ReLU activation function for all the layers.)
        """
        # List of Hidden Layers
        self.layers = None  # TODO

        # List of Activations
        self.f = None  # TODO

        self.debug = debug

    def forward(self, A):

        if self.debug:

            self.Z = []
            self.A = [A]

        L = len(self.layers)

        for i in range(L):

            Z = None  # TODO
            A = None  # TODO

            if self.debug:

                self.Z.append(Z)
                self.A.append(A)

        return NotImplemented

    def backward(self, dLdA):

        if self.debug:

            self.dAdZ = []
            self.dLdZ = []
            self.dLdA = [dLdA]

        L = len(self.layers)

        for i in reversed(range(L)):

            dAdZ = None  # TODO
            dLdZ = None  # TODO
            dLdA = None  # TODO

            if self.debug:

                self.dAdZ = [dAdZ] + self.dAdZ
                self.dLdZ = [dLdZ] + self.dLdZ
                self.dLdA = [dLdA] + self.dLdA

        return NotImplemented
