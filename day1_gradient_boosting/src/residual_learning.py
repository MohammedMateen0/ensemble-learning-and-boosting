import numpy as np
from sklearn.tree import DecisionTreeRegressor


def run_residual_demo():

    print("\n===== RESIDUAL LEARNING DEMO =====\n")

    # Simple dataset
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([5, 7, 9, 11, 13])

    # Initial prediction
    F0 = np.mean(y)

    print("Initial Prediction (Mean):")
    print(F0)

    # Residuals
    residuals = y - F0

    print("\nResiduals:")
    print(residuals)

    # Weak learner
    tree = DecisionTreeRegressor(max_depth=1)

    tree.fit(X, residuals)

    # Predict corrections
    corrections = tree.predict(X)

    print("\nPredicted Corrections:")
    print(corrections)

    # Learning rate
    learning_rate = 0.1

    # Updated predictions
    updated_predictions = F0 + learning_rate * corrections

    print("\nUpdated Predictions:")
    print(updated_predictions)

    # New residuals
    new_residuals = y - updated_predictions

    print("\nNew Residuals:")
    print(new_residuals)