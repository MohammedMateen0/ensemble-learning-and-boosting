# Gradient Boosting Foundations

This module focuses on the mathematical intuition and implementation of Gradient Boosting using residual learning, weak learners, and additive modeling.

The project includes manual boosting implementation, learning rate experiments, tree count analysis, and visualization of model behavior.

---

# Topics Covered

## Core Concepts

- Boosting vs Bagging
- Weak Learners
- Residual Learning
- Additive Models
- Gradient Descent Intuition
- Learning Rate (Shrinkage)
- Bias Reduction
- Overfitting in Boosting

---

# Mathematical Foundation

Gradient Boosting builds models sequentially:

\[
F(x) = F0​(x)+ηh1​(x)+ηh2​(x)+⋯+ηhn​(x)
\]

Where:
- \(F0​(x)\) = initial prediction
- \(hn(x)\) = weak learner
- \(η\) = learning rate

Residuals:

\[
r_i = y_i - \hat{y}_i
\]

Each new tree learns the current prediction errors.

---

# Project Structure

```text
day1_gradient_boosting/
│
├── notebooks/
│   └── gradient_boosting_basics.ipynb
│
├── src/
│   ├── residual_learning.py
│   ├── boosting_loop.py
│   ├── experiments.py
│   ├── visualization.py
│   └── main.py
│
├── images/
│
└── README.md