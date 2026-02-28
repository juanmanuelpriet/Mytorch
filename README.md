# Mytorch

Un framework de aprendizaje profundo ligero y educativo, inspirado en PyTorch. Desarrollado desde cero para comprender los fundamentos de las arquitecturas de redes neuronales, la retropropagación y la optimización.

## 🚀 Características

Mytorch implementa los componentes centrales de un stack de deep learning moderno:

- **Capas de Redes Neuronales**:
  - `Linear`: Capas completamente conectadas estándar.
  - `BatchNorm1d` / `BatchNorm2d`: Normalización por lotes para estabilidad en el entrenamiento.
- **Funciones de Activación**:
  - `ReLU`, `Sigmoid`, `Tanh`, `Identity`, `GeLU` y `SoftMax`.
- **Funciones de Pérdida**:
  - `MSELoss` para regresión.
  - `CrossEntropyLoss` para clasificación.
- **Modelos**:
  - `MLP`: Implementación modular de Perceptrón Multicapa.
- **Motor Autograd**: Implementación personalizada de pasos forward (hacia adelante) y backward (hacia atrás).

## 📁 Estructura del Proyecto

```text
HW1P1/
├── mytorch/            # Lógica central del framework
│   ├── nn/             # Módulos de red neuronal (Linear, BatchNorm, etc.)
│   └── ...            
├── models/             # Arquitecturas de modelos predefinidas (MLP)
├── README.md           # Documentación del proyecto
└── .gitignore          # Reglas estrictas de exclusión
```

## 🛠 Instalación y Uso

### Prerrequisitos
- Python 3.8+
- NumPy

### Configuración
1. Clona el repositorio:
   ```bash
   git clone https://github.com/juanmanuelpriet/Mytorch.git
   cd Mytorch
   ```
2. (Opcional) Crea un entorno virtual:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

### Ejemplo Básico
```python
from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU

# Definir una capa simple
capa = Linear(128, 64)
activacion = ReLU()

# Pase hacia adelante
salida = activacion(capa(input_tensor))
```

## 🧪 Pruebas

La infraestructura de pruebas está optimizada para verificación local.
- Pendiente: Implementación de suite de pruebas estándar.
- Verificación vía scripts de validación locales.

## 🗺 Hoja de Ruta (Roadmap)

- [x] Capas Lineales y de Activación básicas.
- [x] Normalización por Lotes (Batch Normalization).
- [x] Arquitectura MLP.
- [ ] Implementar optimizadores avanzados (Adam, RMSProp).
- [ ] Soporte para Capas Convolucionales (CNNs).
- [ ] Documentación avanzada.

---
*Desarrollado con fines educativos para el curso de Fundamentos de Redes Neuronales.*
