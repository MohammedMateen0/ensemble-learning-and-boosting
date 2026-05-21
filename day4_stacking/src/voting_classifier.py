from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    VotingClassifier
)

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score

data=load_breast_cancer()

X=data.data
y=data.target

X_train,X_test,y_train,y_test=train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Base Models

lr=LogisticRegression(max_iter=5000)

rf=RandomForestClassifier(random_state=42)

xgb=XGBClassifier(
    eval_matric="logloss",
    random_state=42
)

#Voting ensemble

ensemble=VotingClassifier(
    estimators=[
        ("lr",lr),
        ("rf",rf),
        ("xgb",xgb)
    ],
    voting="soft"
)

ensemble.fit(X_train,y_train)

predictoins=ensemble.predict(X_test)

accuracy=accuracy_score(y_test,predictoins)

print("Accuracy",accuracy)