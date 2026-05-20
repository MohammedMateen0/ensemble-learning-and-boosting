import time

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

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

xgb_model=XGBRegressor(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)
start=time.time()

xgb_model.fit(X_train,y_train)

xgb_time=time.time()-start

lgbm_model=LGBMRegressor(
    n_estimators=300,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)

start=time.time()

lgbm_model.fit(X_train,y_train)

lgbm_time=time.time()-start

print(f'''XGBoost Training Time: {xgb_time:.4f} seconds
LightGBM Training Time: {lgbm_time:.4f} seconds''')