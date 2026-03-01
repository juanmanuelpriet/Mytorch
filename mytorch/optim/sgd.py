import numpy as np


class SGD:
    # Implementa el optimizador Stochastic Gradient Descent (SGD),
    # con opción de momentum (si momentum != 0).

    def __init__(self, model, lr=0.1, momentum=0):
        # Constructor del optimizador.
        # model: tu red (tiene sus capas lineales dentro de model.layers)
        # lr: learning rate (tamaño del paso)
        # momentum: μ (si es 0 → SGD normal; si >0 → SGD con momentum)

        self.l = model.layers
        # Guarda referencia directa a la lista de capas del modelo.
        # OJO: esto asume que model.layers contiene capas con parámetros (W, b).

        self.L = len(model.layers)
        # Número total de capas en el modelo.

        self.lr = lr
        # Learning rate (η o ϛ en el PDF): escala cuánto te mueves en dirección del gradiente.

        self.mu = momentum
        # Tasa de momentum (μ en el PDF). Si μ=0, no hay inercia; si μ≈0.9, sí.

        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f")
                    for i in range(self.L)]
        # Lista de "velocidades" v_W, una por capa.
        # Cada v_W[i] tiene la MISMA shape que W de esa capa.
        # dtype="f" crea floats (como pide el enunciado), para evitar ints.

        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f")
                    for i in range(self.L)]
        # Lista de velocidades v_b, una por capa.
        # Cada v_b[i] tiene la misma shape que b de esa capa.

    def step(self):
        # step() aplica UNA actualización de parámetros usando los gradientes ya calculados.
        # Esto se llama DESPUÉS de:
        #   model.forward(...)
        #   loss.forward(...)
        #   model.backward(...)
        # Porque backward es el que llena dLdW y dLdb en cada capa.

        for i in range(self.L):
            # Recorre cada capa del modelo y actualiza sus parámetros.

            if self.mu == 0:
                # Caso 1: SGD "puro" (sin momentum).
                # Fórmulas del PDF:
                #   W := W - lr * dL/dW
                #   b := b - lr * dL/db

                self.l[i].W = self.l[i].W - self.lr * self.l[i].dLdW  # TODO
                # Actualiza los pesos W de la capa i restando lr * gradiente.
                # dLdW debe haber sido calculado en Linear.backward(...).
                # Si dLdW es grande, el paso es grande; lr controla la magnitud.

                self.l[i].b = self.l[i].b - self.lr * self.l[i].dLdb  # TODO
                # Actualiza el bias b de la capa i restando lr * gradiente del bias.

            else:
                # Caso 2: SGD con momentum.
                # El PDF define:
                #   v_W := μ v_W + dL/dW
                #   W := W - lr * v_W
                # Similar para b.

                self.v_W[i] = self.mu * self.v_W[i] + self.lr * self.l[i].dLdW  # TODO
                # Aquí estás manteniendo una "velocidad" para W.
                # Interpretación:
                #  - μ * v_W[i]: conserva parte de la dirección anterior (inercia).
                #  - + lr * dLdW: agrega el gradiente actual escalado.
                #
                # OJO IMPORTANTE:
                # Según el PDF típico, la velocidad suele ser:
                #   v_W = μ v_W + dLdW    (SIN lr aquí)
                # y luego:
                #   W = W - lr * v_W
                # En TU código, metiste lr dentro de v_W y luego restas v_W directo abajo.
                # Eso también es válido (solo cambia dónde multiplicas por lr),
                # pero debes ser consistente (y lo eres porque luego haces W -= v_W).

                self.v_b[i] = self.mu * self.v_b[i] + self.lr * self.l[i].dLdb  # TODO
                # Igual que arriba, pero para el bias:
                # v_b acumula historial + gradiente actual (escalado por lr).

                self.l[i].W = self.l[i].W - self.v_W[i]  # TODO
                # Actualiza W usando la "velocidad" acumulada.
                # Como en v_W ya metiste lr, aquí no multiplicas por lr otra vez.

                self.l[i].b = self.l[i].b - self.v_b[i]  # TODO
                # Actualiza b usando su velocidad.
