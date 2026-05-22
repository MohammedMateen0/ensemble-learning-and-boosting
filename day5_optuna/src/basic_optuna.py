from numpy import square
import optuna
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

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
  learning_rate=trial.suggest_float(
      "learning_rate",
      0.01,
      0.3
  )
  max_depth=trial.suggest_int(
      "max_depth",
      3,
      10,
)
  n_estimators=trial.suggest_int(
      "n_estimators",
      50,
      300
  )
  model=XGBRegressor(
      learning_rate=learning_rate,
      max_depth=max_depth,
      n_estimators=n_estimators,
      random_state=42 
  )
  model.fit(X_train,y_train)

  predictions=model.predict(X_test)

  mse=mean_squared_error(y_test,predictions)
  rmse=mse**(1/2)
  return rmse

study=optuna.create_study(
    direction="minimize"
)
study.optimize(
    objective,
    n_trials=20
)

print(f'''Best Trial:
{study.best_trial}
Best parameters:
{study.best_params}
Best RMSE:
{study.best_value}''')