import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

def boosting_loop():
    # Dataset
    X = np.array([1,2,3,4,5]).reshape(-1,1)
    y = np.array([5,7,9,11,13])

    # Initial prediction
    F = np.full(len(y), np.mean(y))

    learning_rate = 0.1
    n_trees = 5

    models = []

    for i in range(n_trees):

        # Compute residuals
        residuals = y - F

        # Train weak learner
        tree = DecisionTreeRegressor(max_depth=1)
        tree.fit(X, residuals)

        # Predict residuals
        pred = tree.predict(X)

        # Update ensemble
        F += learning_rate * pred

        # Store model
        models.append(tree)

        # Metrics
        mse = mean_squared_error(y, F)

        print(f"Iteration {i+1}")
        print("Predictions:", F)
        print("MSE:", mse)
        print("-"*40)