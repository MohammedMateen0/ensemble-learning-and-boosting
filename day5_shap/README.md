# SHAP Explainable AI

## Overview

This module focuses on explainable machine learning using SHAP (SHapley Additive exPlanations).

SHAP explains:
- how machine learning models make predictions
- how much each feature contributes
- why predictions increase or decrease

The experiments in this module use XGBoost regression models and SHAP visualizations.

---

# Topics Covered

- SHAP values
- TreeExplainer
- Global explanations
- Local explanations
- Feature attribution
- Explainable AI
- Model interpretability

---

# Algorithms Used

- XGBRegressor
- SHAP TreeExplainer

---

# Files

## `basic_shap.py`

Introduction to:
- SHAP values
- TreeExplainer
- feature contribution analysis

---

## `beeswarm_plot.py`

Global feature importance visualization.

Shows:
- most influential features
- feature impact distribution
- positive and negative contribution behavior

---

## `waterfall_plot.py`

Single prediction explanation.

Shows:
- how each feature pushes prediction up or down
- local prediction reasoning

---

## `dependence_plot.py`

Relationship between:
- feature value
- SHAP contribution

Useful for:
- understanding nonlinear behavior
- feature interaction analysis

---

## `force_plot.py`

Interactive visualization for:
- individual predictions
- contribution direction
- prediction decomposition

---

## `feature_importance_vs_shap.py`

Comparison between:
- traditional feature importance
- SHAP explainability

Highlights why SHAP provides deeper interpretability.

---

# Key Concepts

## SHAP Values

Each SHAP value represents:
- a feature's contribution to prediction

Positive SHAP:
- increases prediction

Negative SHAP:
- decreases prediction

---

## Global Explanations

Explain:
- overall model behavior
- important features across dataset

Example:
- beeswarm plots
- summary plots

---

## Local Explanations

Explain:
- one specific prediction

Example:
- waterfall plot
- force plot

---

# Why SHAP?

Traditional feature importance only explains:
- which features matter

SHAP explains:
- how features affect predictions
- why predictions happen
- contribution direction and magnitude

---

# Core SHAP Equation

Prediction = Base Value + Sum of SHAP Contributions

---

# Key Learnings

- Explainable AI workflows
- Feature attribution
- Interpretable boosting models
- Local vs global explanations
- Business-friendly ML interpretation

---

# Installation

```bash
pip install shap
Run Example
python src/beeswarm_plot.py
```
---
# Future Improvements
- SHAP dashboards
- Streamlit integration
- Real estate price explanations
- Classification explainability
- Model monitoring