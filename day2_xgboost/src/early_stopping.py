from xgboost import XGBRegressor

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

data =fetch_california_housing()

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model=XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    early_stopping_rounds=20,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test,y_test)],
    verbose=True
    )
print("Best Iteration:",model.best_iteration)

