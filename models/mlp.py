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
    # MLP con 4 capas ocultas + 1 capa de salida.
    # Arquitectura (lineales):
    #   (2→4) → (4→8) → (8→8) → (8→4) → (4→2)
    # Y una ReLU después de CADA capa lineal (según tu docstring).

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
        # Docstring: te especifica exactamente los tamaños de cada capa lineal (in_features, out_features).
        # "4 hidden layers + output" ⇒ 5 capas lineales en total.

        # List of Hidden Layers
        self.layers = [Linear(2, 4), Linear(4, 8), Linear(8, 8), Linear(8, 4), Linear(4, 2)]  # TODO
        # Lista de capas lineales (capas con parámetros W, b).
        # Cada Linear(in, out) transforma activaciones A de shape (N, in) a pre-activaciones Z de shape (N, out):
        #   Z = A @ W^T + b
        #
        # Shapes por etapa (N = batch size):
        #   A0: (N,2)
        #   Z0: (N,4)  por Linear(2→4)
        #   Z1: (N,8)  por Linear(4→8)
        #   Z2: (N,8)  por Linear(8→8)
        #   Z3: (N,4)  por Linear(8→4)
        #   Z4: (N,2)  por Linear(4→2)

        # List of Activations
        self.f = [ReLU(), ReLU(), ReLU(), ReLU(), ReLU()]
        # Lista de activaciones, una por cada capa lineal.
        # ReLU se aplica elemento a elemento: ReLU(x)=max(0,x).
        #
        # Nota: en muchos modelos, la última capa NO lleva ReLU (se deja Identity o se aplica Softmax aparte),
        # pero tu enunciado/docstring dice "ReLU para todas", así que aquí hay 5 ReLU.

        self.debug = debug
        # Flag que activa guardado de intermedios (útil para depuración y para revisar gradientes).

    def forward(self, A):
        # Forward: calcula la salida del modelo a partir de la entrada A.
        # Aquí "A" empieza siendo A0 (la entrada), y se va actualizando capa por capa.

        if self.debug:
            self.Z = []
            # Guardará cada pre-activación Zi (salida de cada Linear).

            self.A = [A]
            # Guardará activaciones Ai.
            # A[0] = entrada A0, A[1] = salida después de la 1ra ReLU, ..., A[5] = salida final.

        L = len(self.layers)
        # Número de capas lineales (aquí L=5).
        # Como tienes self.f del mismo tamaño, f[i] corresponde a layers[i].

        for i in range(L):
            # Recorre las 5 etapas: (Linear i) seguido de (ReLU i).

            Z = self.layers[i].forward(A)
            # Aplica la capa lineal i.
            # Matemáticamente: Zi = Ai @ Wi^T + bi
            # Shape cambia según la capa:
            #   i=0: (N,2)->(N,4)
            #   i=1: (N,4)->(N,8)
            #   i=2: (N,8)->(N,8)
            #   i=3: (N,8)->(N,4)
            #   i=4: (N,4)->(N,2)

            A = self.f[i].forward(Z)
            # Aplica ReLU a Zi para obtener Ai+1:
            #   Ai+1 = ReLU(Zi)
            # ReLU no cambia la forma: A tiene la misma shape que Z en esa etapa.

            if self.debug:
                self.Z.append(Z)
                # Guarda Zi (pre-activación) para inspección o para chequear backward.

                self.A.append(A)
                # Guarda Ai+1 (post-activación) para inspección.

        return A
        # Devuelve la última activación (A5), que es la salida final del modelo.
        # En este caso shape (N,2).

    def backward(self, dLdA):
        # Backward: propaga gradientes desde la salida hacia la entrada.
        # dLdA entra como ∂L/∂A_last (aquí A5). Shape: (N,2).

        if self.debug:
            self.dLdA = [dLdA]
            # Guardará una lista de gradientes ∂L/∂Ai alineados (eventualmente) con self.A.

            self.dAdZ = []
            # Guardará derivadas locales de activaciones: dAi+1/dZi (máscara de ReLU).

            self.dLdZ = []
            # Guardará gradientes respecto a Zi: ∂L/∂Zi.

        L = len(self.layers)
        # Número de capas lineales (5). Recorremos en reversa: 4→0.

        for i in reversed(range(L)):
            # Recorremos desde la última capa hacia la primera:
            # i=4,3,2,1,0

            # Pass through activation first (backward)
            dAdZ = self.f[i].backward()
            # Derivada local de ReLU en la etapa i: dAi+1/dZi.
            # Para ReLU:
            #   dAdZ = 1 donde Zi > 0
            #   dAdZ = 0 donde Zi <= 0
            #
            # IMPORTANTE:
            # Tu ReLU.backward() NO recibe Z, así que ReLU.forward() debió guardar internamente
            # (por ejemplo una máscara basada en Zi) para poder devolver dAdZ aquí.

            dLdZ = dLdA * dAdZ
            # Regla de la cadena para pasar de gradiente en activación a gradiente en pre-activación:
            #   ∂L/∂Zi = (∂L/∂Ai+1) ⊙ (∂Ai+1/∂Zi)
            # Multiplicación elemento a elemento.
            # Shape de dLdZ coincide con Zi.

            # Then through linear layer
            dLdA = self.layers[i].backward(dLdZ)
            # Propaga el gradiente a través de la capa lineal i.
            # Devuelve ∂L/∂Ai (gradiente respecto a la entrada de esa capa lineal).
            #
            # Internamente Linear.backward(dLdZ) típicamente:
            #   - Calcula y guarda gradientes de parámetros: dLdW_i y dLdb_i
            #   - Devuelve dLdA_prev con shape igual al input de esa capa
            #
            # Ejemplo en i=4 (Linear 4→2):
            #   dLdZ shape: (N,2)  ->  dLdA shape: (N,4)

            if self.debug:
                self.dAdZ = [dAdZ] + self.dAdZ
                # Prepend para que el orden quede 0..4 (igual que en forward).

                self.dLdZ = [dLdZ] + self.dLdZ
                # Guarda ∂L/∂Zi por etapa, alineado por índice.

                self.dLdA = [dLdA] + self.dLdA
                # Guarda ∂L/∂Ai (va quedando alineado con self.A si haces prepend).

        return dLdA
        # Devuelve el gradiente respecto a la entrada original A0: ∂L/∂A0.
        # Shape final: (N,2).