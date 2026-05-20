from xgboost import XGBRegressor

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split

data= fetch_california_housing()

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

configs=[
    {"reg_alpha":0,"reg_lambda":1},
    {"reg_alpha":1,"reg_lambda":1},
    {"reg_alpha":5,"reg_lambda":10}
]

for config in configs:
    model=XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        reg_alpha=config['reg_alpha'],
        reg_lambda=config['reg_lambda'],
        random_state=42
    )

    model.fit(X_train,y_train)

    train_score=model.score(X_train,y_train)

    test_score=model.score(X_test,y_test)

    print(f'''\n Configuration: {config}
Train R²: {train_score:.4f}
Test R²: {test_score:.4f}''')