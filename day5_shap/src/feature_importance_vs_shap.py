import shap

import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# LOAD DATASET


data = fetch_california_housing()

X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

y = data.target

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



# TRAIN XGBOOST MODEL
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)


# NORMAL FEATURE IMPORTANCE
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n===== XGBOOST FEATURE IMPORTANCE =====\n")

print(importance_df)


# Plot Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("../images/xgboost_feature_importance.png")
plt.show()



# SHAP EXPLAINABILITY
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test)


shap.summary_plot(
    shap_values,
    X_test,
    feature_names=X.columns,
    show=False
)

plt.savefig("../images/shap_beeswarm.png")
plt.show()

shap_importance = pd.DataFrame({
    "Feature": X.columns,
    "Mean_Absolute_SHAP": abs(shap_values).mean(axis=0)
})

shap_importance = shap_importance.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)
print("\n===== SHAP FEATURE IMPORTANCE =====\n")
print(shap_importance)
print("\n===== IMPORTANT INSIGHT =====\n")
print(
    """
Traditional Feature Importance:
- Tells WHICH features matter

SHAP:
- Tells WHICH features matter
- Explains HOW they affect predictions
- Explains WHY predictions happen
- Works at both:
    1. Global level
    2. Individual prediction level
"""
)