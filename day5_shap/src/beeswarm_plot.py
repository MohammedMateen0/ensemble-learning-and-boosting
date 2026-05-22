import shap

import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.datasets import fetch_california_housing

from sklearn.model_selection import train_test_split


# Dataset
data = fetch_california_housing()

X = data.data
y = data.target

feature_names = data.feature_names

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = XGBRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Explainer
explainer = shap.TreeExplainer(model)

# SHAP values
shap_values = explainer.shap_values(X_test)

# Beeswarm plot
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    show=False
)

# Save image
plt.savefig("../images/beeswarm_plot.png")

plt.show()