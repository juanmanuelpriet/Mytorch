import numpy as np


class BatchNorm1d:
    # BatchNorm1d normaliza features (columnas) de un batch 2D:
    #   Z tiene shape (N, F) donde:
    #   N = batch size (número de ejemplos)
    #   F = num_features (número de neuronas/features)
    #
    # Hace esto en TRAIN:
    #   1) calcula media M y varianza V por feature usando el batch
    #   2) normaliza: NZ = (Z - M) / sqrt(V + eps)
    #   3) escala y desplaza: BZ = BW * NZ + Bb   (gamma y beta)
    #   4) actualiza running_M, running_V (promedios móviles)
    #
    # En INFERENCE:
    #   usa running_M y running_V en vez de las estadísticas del batch.

    def __init__(self, num_features, alpha=0.9):
        # num_features = F (número de columnas/features)
        # alpha = factor de "momentum" para running stats:
        #   running = alpha*running + (1-alpha)*batch_stat

        self.alpha = alpha
        # Peso para el promedio móvil (más alto = cambia más lento)

        self.eps = 1e-8
        # Epsilon para evitar dividir por 0 y mejorar estabilidad numérica

        self.BW = np.ones((1, num_features))
        # BW = gamma (parámetro entrenable) para ESCALAR cada feature.
        # Shape (1, F) para broadcast a (N, F)

        self.Bb = np.zeros((1, num_features))
        # Bb = beta (parámetro entrenable) para DESPLAZAR cada feature.
        # Shape (1, F)

        self.dLdBW = np.zeros((1, num_features))
        # Gradiente de la loss respecto a BW (gamma)

        self.dLdBb = np.zeros((1, num_features))
        # Gradiente de la loss respecto a Bb (beta)

        # Running mean and variance, updated during training, used during inference
        self.running_M = np.zeros((1, num_features))
        # running_M: media "acumulada" por feature para usar en inference

        self.running_V = np.ones((1, num_features))
        # running_V: varianza "acumulada" por feature para usar en inference
        # inicia en 1 para evitar dividir por 0 antes de entrenar

    def forward(self, Z, eval=False):
        """
        Z: entrada a batchnorm (normalmente pre-activaciones) shape (N, F)
        eval:
          - False -> TRAINING: usa estadísticas del batch y actualiza running stats
          - True  -> INFERENCE: usa running stats (no depende del batch actual)
        """
        self.Z = Z
        # Guarda Z para backward (por si lo necesitas; aquí se usa NZ y V)

        self.N = Z.shape[0]
        # N = batch size (número de filas)

        self.M = np.mean(Z, axis=0)
        # M = media por feature (por columna)
        # axis=0: promedio sobre filas -> shape (F,)
        # Ej: M[j] = promedio de Z[:, j]

        self.V = np.var(Z, axis=0)
        # V = varianza por feature (por columna)
        # shape (F,)
        # Ej: V[j] = varianza de Z[:, j]

        if eval == False:
            # =========================
            # TRAINING MODE
            # =========================

            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)
            # NZ = normalized Z (Z-hat del PDF)
            # Restas la media (centras) y divides por std (escalas)
            # sqrt(V + eps) es la desviación estándar por feature
            # NZ shape: (N, F) (por broadcasting)

            self.BZ = self.BW * self.NZ + self.Bb
            # BZ = salida de BatchNorm (Z-tilde del PDF):
            # gamma * NZ + beta
            # BW y Bb son (1,F) pero se "expanden" a (N,F)

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            # Actualiza media acumulada:
            # running_M := alpha*running_M + (1-alpha)*M_batch
            # OJO de shape:
            # - running_M es (1,F)
            # - M es (F,)
            # Numpy broadcast -> lo convierte compatible

            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V
            # Actualiza varianza acumulada:
            # running_V := alpha*running_V + (1-alpha)*V_batch

        else:
            # =========================
            # INFERENCE MODE
            # =========================
            # Aquí NO quieres que la salida dependa del batch actual,
            # por eso usas las estadísticas acumuladas durante training.

            self.NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            # Normaliza usando running mean/var
            # running_M y running_V ya son (1,F)

            self.BZ = self.BW * self.NZ + self.Bb
            # Igual que training: escala y shift con parámetros aprendidos

        return self.BZ
        # Devuelve salida batchnorm: shape (N, F)

    def backward(self, dLdBZ):
        # dLdBZ: gradiente de la loss respecto a la salida BZ (Z-tilde)
        # shape: (N, F)
        #
        # Objetivos:
        # 1) calcular gradientes de gamma (BW) y beta (Bb)
        # 2) devolver dLdZ (gradiente respecto a la entrada Z) para seguir backprop

        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)
        # Gradiente respecto a BW (gamma):
        # BZ = BW * NZ + Bb
        # ∂BZ/∂BW = NZ
        # Entonces dLdBW = sum_over_batch(dLdBZ ⊙ NZ)
        # axis=0 suma sobre N -> queda (1,F)

        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)
        # Gradiente respecto a Bb (beta):
        # ∂BZ/∂Bb = 1
        # Entonces dLdBb = sum_over_batch(dLdBZ)
        # shape (1,F)

        # =========================
        # Gradiente respecto a Z (entrada)
        # =========================
        # Calcular dLdZ "a mano" desde la definición completa de BN es largo,
        # pero existe una fórmula simplificada muy usada (y es la que pones):
        #
        # dL/dZ = (gamma / (N * std)) * (N*dL/dBZ - sum(dL/dBZ) - NZ*sum(dL/dBZ * NZ))
        #
        # Donde:
        # - gamma = BW
        # - std = sqrt(V + eps)
        # - sum(dL/dBZ) es por feature (sobre N)
        # - sum(dL/dBZ * NZ) también es por feature
        #
        # Nota: tu comentario dice:
        # "sum(dL/dB * NZ) is self.dLdBW (without BW factor)"
        # En tu código, self.dLdBW = sum(dLdBZ*NZ) (ya está perfecto)

        inv_std = 1.0 / np.sqrt(self.V + self.eps)
        # inv_std = 1/std por feature
        # shape (F,)

        dLdZ = (self.BW * inv_std / self.N) * (
            self.N * dLdBZ - self.dLdBb - self.NZ * self.dLdBW
        )
        # Aquí aplicas la fórmula simplificada:
        #
        # - (self.BW * inv_std / self.N): factor común por feature
        # - self.N * dLdBZ: el término principal
        # - self.dLdBb = sum(dLdBZ) por feature (1,F) -> broadcast a (N,F)
        # - self.NZ * self.dLdBW: NZ (N,F) ⊙ sum(dLdBZ*NZ) (1,F) -> (N,F)
        #
        # Resultado dLdZ shape: (N, F)
        # Eso se devuelve para que la capa anterior reciba su gradiente.

        return dLdZ
        # Gradiente de la loss respecto a la entrada Z, para seguir backprop.