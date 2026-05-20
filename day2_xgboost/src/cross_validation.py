import numpy as np
from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score

data=fetch_california_housing()

X=data.data
y=data.target

model=XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

scores=cross_val_score(
    model,
    X,

    y,
    cv=5,
    scoring="r2"
)
print(f'''Cross Validation Scores:
{scores}
Mean CV Score:{np.mean(scores):.4f}''')