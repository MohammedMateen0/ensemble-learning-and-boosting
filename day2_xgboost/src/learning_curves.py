import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

import sys
import os

sys.path.append(
    os.path.abspath("..")
)

data=fetch_california_housing()

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model=XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X_train,y_train)

train_errors=[]
test_errors=[]

for i in range(1,301):
    train_pred=model.predict(X_train,iteration_range=(0,i))
    test_pred=model.predict(X_test,iteration_range=(0,i))
    train_mse=mean_squared_error(y_train,train_pred)
    test_mse=mean_squared_error(y_test,test_pred)
    train_errors.append(train_mse)
    test_errors.append(test_mse)

plt.figure(figsize=(10,6))
plt.plot(train_errors,label="Train Error")
plt.plot(test_errors,label="Test Errors")
plt.xlabel("Number Of Trees")
plt.ylabel("MSE")
plt.title("XGBoost Learning Curve")
plt.legend()
plt.savefig("../images/xgboost_learning_curve.png")
plt.show()