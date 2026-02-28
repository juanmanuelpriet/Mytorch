# Mytorch

A lightweight, educational deep learning framework inspired by PyTorch. Built from scratch to understand the fundamentals of neural network architectures, backpropagation, and optimization.

## 🚀 Features

Mytorch implements core components of a modern deep learning stack:

- **Neural Network Layers**:
  - `Linear`: Standard fully connected layers.
  - `BatchNorm1d` / `BatchNorm2d`: Batch normalization for training stability.
- **Activation Functions**:
  - `ReLU`, `Sigmoid`, `Tanh`, `Identity`, `GeLU`, and `SoftMax`.
- **Loss Functions**:
  - `MSELoss` for regression.
  - `CrossEntropyLoss` for classification.
- **Models**:
  - `MLP`: Modular Multi-Layer Perceptron implementation.
- **Autograd Engine**: Custom implementation of forward and backward passes.

## 📁 Project Structure

```text
HW1P1/
├── mytorch/            # Core framework logic
│   ├── nn/             # Neural network modules (Linear, BatchNorm, etc.)
│   └── ...            
├── models/             # Pre-defined model architectures (MLP)
├── README.md           # Project documentation
└── .gitignore          # Strict exclusion rules
```

## 🛠 Installation & Usage

### Prerequisites
- Python 3.8+
- NumPy

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/juanmanuelpriet/Mytorch.git
   cd Mytorch
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

### Basic Example
```python
from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU

# Define a simple layer
layer = Linear(128, 64)
activation = ReLU()

# Forward pass
output = activation(layer(input_tensor))
```

## 🧪 Testing

Testing infrastructure is currently optimized for local verification. To run the internal tests (if available):
- TODO: Add specific test runner commands if a standard test suite is implemented.
- Pendiente: Verificación vía scripts de validación locales.

## 🗺 Roadmap

- [x] Basic Linear and Activation layers.
- [x] Batch Normalization.
- [x] MLP Architecture.
- [ ] TODO: Implement advanced optimizers (Adam, RMSProp).
- [ ] TODO: Support for Convolutional Layers (CNNs).
- [ ] TODO: Documentation for advanced usage.

---
*Developed for educational purposes in the Neural Networks Fundamentals course.*
