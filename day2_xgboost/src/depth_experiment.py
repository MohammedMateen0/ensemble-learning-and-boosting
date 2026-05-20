from xgboost import XGBRegressor

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

data=fetch_california_housing()

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

depths=[2,4,6,10]

for depth in depths:
    model=XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=depth,
        random_state=42
    )
    model.fit(X_train,y_train)
    train_score=model.score(X_train,y_train)
    test_score=model.score(X_test,y_test)

    print(f'''\n Depth: {depth}
Train R²: {train_score}
Test R²: {test_score}''')