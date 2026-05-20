import xgboost as xgb

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

dtrain=xgb.DMatrix(X_train,label=y_train)
dtest=xgb.DMatrix(X_test,label=y_test)

params={
    "objective":"reg:squarederror",
    "max_depth":4,
    "learning_rate":0.1
}
model=xgb.train(
    params,
    dtrain=dtrain,
    num_boost_round=100
)

predictions=model.predict(dtest)

mse=mean_squared_error(y_test,predictions)

print(f"MSE: {mse:.4f}")