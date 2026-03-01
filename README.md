# 🧠 MyTorch: A Deep Learning Framework from Scratch

**MyTorch** is a foundational deep learning framework implemented entirely in **NumPy**. It was designed as a capstone project to master the internal mechanics of modern neural network architectures, gradient descent optimization, and the mathematical intuition behind backpropagation.

---

## 🏛️ Project Architecture

The framework is organized into modular components that mirror the structure of industrial frameworks like PyTorch, enabling deep understanding through direct implementation.

### 1. Neural Network Modules (`mytorch.nn`)
- **`Linear`**: Fully connected layers with optimized weight initialization (Xavier/He).
- **`BatchNorm1d`**: Mitigates "Internal Covariate Shift" using running means and variances for stable training.
- **`Dropout`**: Includes "Inverted Dropout" scaling to ensure consistent output magnitude during inference.
- **`Activations`**: A complete suite of non-linearities: `ReLU`, `GeLU`, `Sigmoid`, `Tanh`, and `SoftMax`.
- **`Sequential`**: A container to orchestrate complex model pipelines concisely.

### 2. Optimization Engine (`mytorch.optim`)
- **`SGD`**: Implementation of Gradient Descent with advanced **Momentum** support.
- **`Adam`**: Adaptive Moment Estimation, combining adaptive learning rates for each parameter.
- **`RMSProp`**: Normalizes gradients using moving averages of squared gradients.

### 3. Loss Functions
- **`CrossEntropyLoss`**: Optimized classification loss with numerical stability tricks (LogSumExp).
- **`MSELoss`**: Standard Mean Squared Error for regression tasks.

---

## 📚 Educational Annotations
What makes this repository unique is its **Professor-Level Documentation**. Every file in `mytorch/` has been annotated with:
- **Mathematical Derivations**: Step-by-step breakdowns of forward and backward gradients (Chain Rule applications).
- **Geometric Intuition**: Visualizations of how decision boundaries warp the data space.
- **Numerical Stability**: Explanations of epsilon terms, bias correction, and overflow prevention.

---

## 🧪 Experimental Showcase: `Example.ipynb`
The repository includes a comprehensive **Benchmark Matrix** that visualizes:
- **5 Diverse Datasets**: Moons, Circles, Blobs, Noisy Moons, and a custom Spiral.
- **Visual Matrix (60 Experiments)**: A massive 3x4 grid per dataset comparing every combination of **Optimizer** vs **Activation Function**.
- **Convergence Analysis**: Direct observation of how Adam outperforms SGD in complex topologies like the spiral.

### How to Run the Experiments
1. Install dependencies: `pip install numpy matplotlib pandas sklearn`
2. Open `Example.ipynb` in your preferred Jupyter environment (VS Code, JupyterLab).
3. Execute "Run All" to generate the visual atlas of neural network convergence.

---

## 🛠 Installation & Setup

```bash
# Clone the repository
git clone https://github.com/juanmanuelpriet/Mytorch.git
cd Mytorch

# Install requirements
pip install -r requirements.txt
```

---

## 🎓 Final Project Significance
This repository represents the culmination of a rigorous Deep Learning curriculum. It demonstrates not just the ability to use AI tools, but the **engineering capability to build them from first principles**.

Developed by **Juan Manuel Prieto** as a comprehensive study of neural foundations.
