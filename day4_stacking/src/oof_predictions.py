import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score

data=load_breast_cancer()

X=data.data
y=data.target

kf=KFold(
    n_splits=5,

    shuffle=True,
    random_state=42
)

off_preds=np.zeros(len(y))

model=RandomForestClassifier(random_state=42)

for train_idx,val_idx in kf.split(X):
    X_train,X_val=X[train_idx],X[val_idx]

    y_tarin,y_val=y[train_idx],y[val_idx]

    model.fit(X_train,y_tarin)

    preds=model.predict(X_val)

    off_preds[val_idx]=preds

accuracy=accuracy_score(y,off_preds)

print(f"Accuracy: {accuracy:.4f}")