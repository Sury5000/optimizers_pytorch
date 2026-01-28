# Optimization Algorithms in Deep Learning (PyTorch)

This project documents my hands-on exploration of **optimization algorithms used in deep learning**, focusing on understanding how different optimizers update model parameters. Instead of only using built-in PyTorch optimizers, I implemented key optimization steps manually to study their internal behavior.

---

## Objective

The objective of this work is to:
- Understand how different optimizers update weights
- Compare classical and adaptive optimization methods
- Study the role of momentum, variance, and weight decay
- Implement optimizer update rules step by step

---

## Base Model Setup

- A simple linear model (`nn.Linear`) is created
- Model parameters are passed to different optimizers
- This model serves as a reference point to study optimizer behavior

---

## 1. Stochastic Gradient Descent (SGD)

### Vanilla SGD
- Implemented using `torch.optim.SGD`
- Uses only the current gradient for weight updates
- Constant learning rate applied to all updates

### SGD with Momentum
- Momentum term added to accelerate convergence
- Past gradients influence the current update
- Helps reduce oscillations during training

### SGD with Nesterov Momentum
- Uses a lookahead gradient calculation
- Adjusts direction before reaching the update point
- Improves convergence stability compared to standard momentum

---

## 2. Adagrad Optimizer

- Implemented using `torch.optim.Adagrad`
- Learning rate adapts based on parameter importance
- Frequently updated parameters receive smaller learning rates
- Demonstrates adaptive learning behavior for sparse features

---

## 3. RMSProp – Manual Implementation

To understand RMSProp internally:
- A custom `rms_prop_step()` function is implemented
- Running variance of gradients is computed using exponential decay
- Learning rate is normalized using the accumulated variance
- Weight updates depend on both past and current gradients

Multiple gradient steps are applied to show how learning rates adapt over time.

---

## 4. Adam Optimizer – Manual Implementation

A custom `adam_step()` function is implemented to study Adam in detail:
- First moment (momentum) is computed
- Second moment (variance) is computed
- Bias correction is applied to both moments
- Final update combines momentum and variance normalization

This demonstrates how Adam combines the benefits of Momentum and RMSProp.

---

## 5. NAdam Optimizer

- Implemented using `torch.optim.NAdam`
- Combines Adam optimization with Nesterov momentum
- Uses both current and future gradient estimates
- Provides faster convergence in some scenarios

---

## 6. AdamW – Decoupled Weight Decay

A custom `adamw_step()` function is implemented to study AdamW:
- Adam update is computed without weight decay interference
- Weight decay is applied independently of gradient updates
- Demonstrates why AdamW is preferred over Adam with L2 regularization

A zero-gradient example is used to clearly show the effect of weight decay.

---

## Key Observations

- Momentum accelerates convergence by accumulating past gradients
- Adaptive optimizers adjust learning rates automatically
- RMSProp stabilizes updates using gradient variance
- Adam combines momentum and adaptive learning rates
- AdamW separates regularization from optimization dynamics

---

## Conclusion

This work provides a detailed understanding of optimization algorithms beyond high-level API usage. By implementing optimizer update rules manually and comparing them with PyTorch implementations, I developed a deeper intuition for how learning rates, momentum, variance, and weight decay influence model training.

These concepts form a critical foundation for training stable and efficient deep learning models.

---
