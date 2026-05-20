import pandas as pd

from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split

df=pd.DataFrame({
    "city":['A','B','A','C','B','C'],
    'rooms':[2,3,2,4,3,5],
    'price':[100,150,120,200,170,250]
})

df['city']=df['city'].astype('category')

X=df[['city','rooms']]
y=df['price']

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model=LGBMRegressor()

model.fit(X_train,y_train)

model.fit(X_train,y_train,categorical_feature=['city'])
predictions=model.predict(X_test)

print(predictions)