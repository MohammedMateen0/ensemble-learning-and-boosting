from sklearn.datasets import fetch_california_housing

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error


def learning_rate_experiment():

    print("\n===== LEARNING RATE EXPERIMENT =====\n")

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

    learning_rates = [0.01, 0.1, 0.5]

    for lr in learning_rates:

        model = GradientBoostingRegressor(
            learning_rate=lr,
            n_estimators=100,
            max_depth=3,
            random_state=42
        )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)

        print(f"\nLearning Rate: {lr}")

        print(f"Train MSE: {train_mse:.4f}")

        print(f"Test MSE: {test_mse:.4f}")

        print("-" * 50)


def tree_experiment():

    print("\n===== TREE COUNT EXPERIMENT =====\n")

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

    tree_counts = [50, 100, 300]

    for n_trees in tree_counts:

        model = GradientBoostingRegressor(
            learning_rate=0.1,
            n_estimators=n_trees,
            max_depth=3,
            random_state=42
        )

        model.fit(X_train, y_train)

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        print(f"\nTrees: {n_trees}")

        print(f"Train R²: {train_score:.4f}")

        print(f"Test R²: {test_score:.4f}")

        print("-" * 50)