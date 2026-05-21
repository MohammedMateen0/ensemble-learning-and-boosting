from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier
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

estimators=[
    ("rf",RandomForestClassifier(random_state=42)),
    ("xgb",XGBClassifier(
        eval_metric="logloss",
        random_state=42
    ))
]

meta_model=LogisticRegression(max_iter=5000)

stack=StackingClassifier(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5
)

stack.fit(X_train,y_train)

predictions=stack.predict(X_test)

accuracy=accuracy_score(y_test,predictions)

print(f"Stacking Accuracy: {accuracy:.4f}")