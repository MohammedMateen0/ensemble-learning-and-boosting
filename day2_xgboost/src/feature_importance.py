import pandas as pd

import matplotlib.pyplot as plt

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
    n_estimators=200,
    random_state=42
)

model.fit(X_train,y_train)

importance=pd.DataFrame({
    "Feature":data.feature_names,
    "Importance":model.feature_importances_
})

importance=importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)

plt.figure(figsize=(10,6))

plt.bar(
    importance['Feature'],
    importance["Importance"]
)

plt.xticks(rotation=45)

plt.title("XGBoost Feature IMportance")

plt.tight_layout()

plt.show()