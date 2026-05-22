import optuna

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
def objective(trial):
  params={
      "learning_rate":trial.suggest_float(
          "learning_rate",
          0.01,
          0.3
      ),
      "max_depth":trial.suggest_int(
          "max_depth",
          3,
          10
      ),
      "n_estimators":trial.suggest_int(
          "n_estimators",
          100,
          500
      ),
      "subsample":trial.suggest_float(
          "subsample",
          0.5,
          1.0
      ),
      "colsample_bytree":trial.suggest_float(
          "colsample_bytree",
          0.5,
          1.0
      ),
      "reg_alpha":trial.suggest_float(
          "reg_alpha",
          1e-4,
          10,
          log=True
      ),
      "reg_lambda":trial.suggest_float(
          "reg_labda",
          1e-4,
          10,
          log=True
      ),
      "random_state":42
  }
  model=LGBMRegressor(**params)

  model.fit(X_train,y_train)

  predictions=model.predict(X_test)

  rmse=mean_squared_error(y_test,predictions)**(1/2)
  return rmse

study=optuna.create_study(
    direction='minimize'
)

study.optimize(
    objective,
    n_trials=5
)
print(f'''Best Parameter:
{study.best_params}
Best RMSE:
{study.best_value}''')
