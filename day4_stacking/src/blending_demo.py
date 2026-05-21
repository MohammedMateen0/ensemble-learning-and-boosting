import numpy as np

from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import  LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

data=load_breast_cancer()

X=data.data
y=data.target

X_train,X_temp,y_train,y_temp=train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)
X_val,X_test,y_val,y_test=train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

rf=RandomForestClassifier(random_state=42)

xgb=XGBClassifier(
    eval_metric="logloss",
    random_state=42
)

rf.fit(X_train,y_train)
xgb.fit(X_train,y_train)

rf_val=rf.predict_proba(X_val)[:,1]
xgb_val=xgb.predict_proba(X_val)[:,1]

meta_X=np.column_stack((rf_val,xgb_val))

meta_model=LogisticRegression()

meta_model.fit(meta_X,y_val)

rf_test=rf.predict_proba(X_test)[:,1]
xgb_test=xgb.predict_proba(X_test)[:,1]

meta_test=np.column_stack((rf_test,xgb_test))

final_preds=meta_model.predict(meta_test)

accuracy=accuracy_score(y_test,final_preds)

print(f"Blending Accuracy:{accuracy:.4f}")