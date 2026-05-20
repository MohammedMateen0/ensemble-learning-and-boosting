from catboost import CatBoostRegressor

import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.DataFrame({
    "city": ["A", "B", "A", "C", "B", "C"],
    "rooms": [2, 3, 2, 4, 3, 5],
    "price": [100, 150, 120, 200, 170, 250]
})

X=df[['city','rooms']]
y=df['price']

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model=CatBoostRegressor(
    verbose=0
)
model.fit(X_train,y_train,cat_features=['city'])

predictions=model.predict(X_test)

print(predictions)