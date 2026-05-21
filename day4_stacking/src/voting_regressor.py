from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    VotingRegressor
)

from xgboost import XGBRegressor

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
#Base Models
lr=LinearRegression()

rf=RandomForestRegressor(random_state=42)

xgb=XGBRegressor(random_state=42)

# Ensemble
ensemble=VotingRegressor(
    estimators=[
        ('lr',lr),
        ('rf',rf),
        ('xgb',xgb)
    ]
)

ensemble.fit(X_train,y_train)

predictions=ensemble.predict(X_test)

mse=mean_squared_error(y_test,predictions)

print(f'Mean Squared Error:{mse:.4f}')