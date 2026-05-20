from xgboost import XGBRegressor

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV
)

from sklearn.metrics import mean_squared_error

data=fetch_california_housing()

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model=XGBRegressor(random_state=42)

params={
    "n_estimators":[100,200,300],
    "max_depth":[3,4,6,8],
    "learning_rate":[0.01,0.05,0.1],
    "subsample":[0.7,0.8,1.0],
    'colsample_bytree':[0.7,0.8,1.0]
}

search=RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    n_iter=10,
    scoring="neg_mean_squared_error",
    verbose=1,
    random_state=42
)
search.fit(X_train,y_train)

best_model=search.best_estimator_

predictions=best_model.predict(X_test)

mse=mean_squared_error(y_test,predictions)

print(f'''Best Parameters:
{search.best_params_}
Test MSE:{mse:.4f}''')