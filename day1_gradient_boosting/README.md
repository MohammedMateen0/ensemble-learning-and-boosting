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
ri = yi - ŷi
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

```
---

# Implementations

## 1. Manual Residual Learning

Implemented:

* initial prediction using mean
* residual computation
* weak learner training
* additive prediction updates

---

## 2. Manual Gradient Boosting Loop

Custom boosting loop includes:

* iterative residual fitting
* shallow decision trees
* learning rate scaling
* MSE tracking

---

## 3. Learning Rate Experiments

Compared:

* 0.01
* 0.1
* 0.5

Observed:

* convergence speed differences
* generalization behavior
* shrinkage effects

---

## 4. Tree Count Experiments

Compared:

* 50 trees
* 100 trees
* 300 trees

Observed:

* gradual improvement from additive learning
* effect of boosting iterations

---

## 5. Learning Curve Visualization

Visualized:

* train error
* test error
* boosting convergence behavior
* overfitting dynamics

---

# Algorithms Used

* DecisionTreeRegressor
* GradientBoostingRegressor

Libraries:

* NumPy
* Matplotlib
* scikit-learn

---

# Key Learnings

* Boosting trains models sequentially
* Each tree learns residual errors
* Learning rate controls update magnitude
* Smaller learning rates usually require more trees
* Boosting reduces bias through additive corrections
* Excessive depth or large learning rates can overfit

---

# Results

## Learning Rate Experiment

| Learning Rate | Test MSE |
| ------------- | -------- |
| 0.01          | 0.6643   |
| 0.1           | 0.2940   |
| 0.5           | 0.2459   |

---

## Tree Count Experiment

| Trees | Test R² |
| ----- | ------- |
| 50    | 0.7435  |
| 100   | 0.7756  |
| 300   | 0.8102  |

---

# How to Run

## Install Dependencies

```bash
pip install -r requirements_day1.txt
```

---

## Run Project

From the `src/` directory:

```bash
python main.py
```

---

# Future Extensions

Next modules in this repository:

* XGBoost
* LightGBM
* CatBoost
* Stacking Ensembles
* SHAP Explainability
* Apartment Price Prediction Project

---


Machine Learning roadmap project focused on:

* ensemble learning
* boosting systems
* production ML engineering
* explainable AI


