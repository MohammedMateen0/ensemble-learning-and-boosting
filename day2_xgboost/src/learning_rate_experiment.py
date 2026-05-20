from xgboost import XGBRegressor

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

learning_rates=[0.01,0.05,0.1,0.3]

for lr in learning_rates:
    model=XGBRegressor(
        n_estimators=300,
        learning_rate=lr,
        max_depth=4,
        random_state=42
    )

    model.fit(X_train,y_train)

    prediction=model.predict(X_test)

    mse=mean_squared_error(y_test,prediction)

    print(f'''Learning rate: {lr}
MSE: {mse:.4f}''')