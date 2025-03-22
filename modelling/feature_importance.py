import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load trained models
rf_model = joblib.load("modelling/random_forest_model.pkl")
xgb_model = joblib.load("modelling/xgboost_model.pkl")


# Load dataset to get feature names
data = pd.read_csv("data/balanced_loan_default_data.csv")
features = data.drop(columns=["has_default"])  # Exclude target variable

# Extract feature importance
rf_importance = pd.DataFrame({"Feature": features.columns, "Importance": rf_model.feature_importances_})
xgb_importance = pd.DataFrame({"Feature": features.columns, "Importance": xgb_model.feature_importances_})


# Sort and normalize
rf_importance = rf_importance.sort_values(by="Importance", ascending=False)
xgb_importance = xgb_importance.sort_values(by="Importance", ascending=False)


# Save importance scores
rf_importance.to_csv("data/results/random_forest_feature_importance.csv", index=False)
xgb_importance.to_csv("data/results/xgboost_feature_importance.csv", index=False)


# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=rf_importance["Importance"][:10], y=rf_importance["Feature"][:10], color="blue")
plt.title("Top 10 Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.savefig("Visualizations/Modelling/random_forest_feature_importance.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=xgb_importance["Importance"][:10], y=xgb_importance["Feature"][:10], color="green")
plt.title("Top 10 Feature Importance (XGBoost)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.savefig("Visualizations/Modelling/xgboost_feature_importance.png")
plt.show()

