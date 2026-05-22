# Optuna Hyperparameter Optimization

## Overview

This module focuses on automated hyperparameter optimization using Optuna.

Instead of manually tuning machine learning models, Optuna intelligently searches the parameter space and finds high-performing configurations using adaptive optimization.

The experiments in this module use:
- XGBoost
- LightGBM
- Regression workflows
- RMSE optimization

---

# Topics Covered

- Optuna studies
- Objective functions
- Trial optimization
- Search spaces
- Hyperparameter tuning
- XGBoost optimization
- LightGBM optimization
- Log-scaled parameter search
- Automated ML experimentation

---

# Algorithms Used

- XGBRegressor
- LGBMRegressor

---

# Files

## `basic_optuna.py`

Introduction to:
- studies
- trials
- objective functions
- parameter suggestions

---

## `xgboost_optuna.py`

Hyperparameter optimization for XGBoost.

Tuned parameters:
- learning_rate
- max_depth
- n_estimators
- subsample
- colsample_bytree
- reg_alpha
- reg_lambda

---

## `lightgbm_optuna.py`

LightGBM hyperparameter optimization.

Tuned parameters:
- num_leaves
- learning_rate
- feature_fraction
- min_data_in_leaf

---

# Key Concepts

## Objective Function

Defines what Optuna tries to optimize.

Example:
- minimize RMSE
- maximize accuracy

---

## Trials

Each trial:
- selects parameter values
- trains model
- evaluates performance

---

## Search Space

Defines:
- parameter ranges
- categorical options
- optimization boundaries

---

# Why Optuna?

Compared to GridSearchCV:
- faster
- more efficient
- adaptive search
- better scalability

Optuna focuses on promising regions of parameter space instead of exhaustively testing every combination.

---

# Key Learnings

- Automated hyperparameter optimization
- Intelligent parameter search
- Model tuning workflows
- Boosting model optimization
- Practical ML experimentation

---

# Installation

```bash
pip install optuna
Run Example
python src/xgboost_optuna.py
```

---
# Future Improvements
- Cross-validation optimization
- Pruning
- Visualization dashboards
- Ensemble optimization
- Multi-objective optimization