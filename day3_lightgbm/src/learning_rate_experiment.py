from lightgbm import LGBMRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data=fetch_california_housing()

X=data.data
y=data.target

X_tarin,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
learning_rates=[0.01,0.05,0.1,0.3]

for lr in learning_rates:
    model=LGBMRegressor(
        n_estimators=300,
        learning_rate=lr,
        num_leaves=31,
        random_state=42
    )
    model.fit(X_tarin,y_train)
    predictons=model.predict(X_test)

    mse=mean_squared_error(y_test,predictons)
    print(f'''Learning Rate: {lr}
Mean Squared Error: {mse:.4f}''')