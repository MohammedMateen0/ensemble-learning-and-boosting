from lightgbm import LGBMRegressor

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split


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

fractions=[0.5,0.7,0.9,1.0]

for frac in fractions:
    model=LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        feature_fraction=frac,
        random_state=42
    )
    model.fit(X_train,y_train)

    score=model.score(X_test,y_test)

    print(f'''Feature Fraction: {frac}
Test R^2: {score:.4f}''')