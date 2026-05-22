import shap

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
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# SHAP Explainer
explainer = shap.TreeExplainer(model)

# SHAP values
shap_values = explainer.shap_values(X_test)

print("SHAP Values Shape:")

print(shap_values.shape)