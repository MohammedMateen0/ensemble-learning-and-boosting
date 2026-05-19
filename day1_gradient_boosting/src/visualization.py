from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split


def plot_learning_curve():

    print("\nGenerating Learning Curve Plot...\n")

    # Dataset
    data = fetch_california_housing()

    X = data.data
    y = data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Model
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    train_errors = []
    test_errors = []

    # Staged predictions
    for pred_train, pred_test in zip(
        model.staged_predict(X_train),
        model.staged_predict(X_test)
    ):

        train_errors.append(
            mean_squared_error(y_train, pred_train)
        )

        test_errors.append(
            mean_squared_error(y_test, pred_test)
        )
    output_path = Path(__file__).resolve().parent.parent / "images" / "learning_curve.png"
    # Plot
    plt.figure(figsize=(10, 6))

    plt.plot(train_errors, label="Train Error")

    plt.plot(test_errors, label="Test Error")

    plt.xlabel("Number of Trees")

    plt.ylabel("MSE")

    plt.title("Gradient Boosting Learning Curve")

    plt.legend()

    plt.savefig(output_path)

    plt.show()