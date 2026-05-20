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

model=XGBRegressor(
    n_estimators=2000,
    learning_rate=1,
    max_depth=15,
    subsample=1,
    colsample_bytree=1,
    random_state=42
)

model.fit(X_train,y_train)

train_score=model.score(X_train,y_train)
test_score=model.score(X_test,y_test)

print(f'''Train R^2: {train_score:.4f}
Test R^2: {test_score:.4f}''')