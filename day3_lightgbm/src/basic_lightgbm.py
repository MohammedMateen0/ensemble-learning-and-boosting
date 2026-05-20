from lightgbm import LGBMRegressor

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

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

model=LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)

model.fit(X_train,y_train)

prediction=model.predict(X_test)

mse=mean_squared_error(y_test,prediction)

print("MSE: ",mse)