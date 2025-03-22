import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import joblib

# Ensure directories exist
os.makedirs("results", exist_ok=True)
os.makedirs("Visualizations/Modelling", exist_ok=True)

# Load the dataset
data = pd.read_csv("data/balanced_loan_default_data.csv")

# Split features and target
X = data.drop(columns=["has_default"])
y = data["has_default"]

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load trained models
log_reg = joblib.load("modelling/logistic_regression_model.pkl")
random_forest = joblib.load("modelling/random_forest_model.pkl")
xgboost_model = joblib.load("modelling/xgboost_model.pkl")

# Ensure Logistic Regression model was trained with feature names
if hasattr(log_reg, "fit"):
    log_reg.fit(X_train, y_train)  # Retrain Logistic Regression with feature names

# Store models for evaluation
models = {
    "Logistic Regression": log_reg,
    "Random Forest": random_forest,
    "XGBoost": xgboost_model
}

results = []
roc_curves = {}

# Evaluate models
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for AUC-ROC
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    results.append([name, accuracy, precision, recall, f1, auc])
    
    # Store ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_curves[name] = (fpr, tpr)

# Convert results to DataFrame
results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC"])
results_df.to_csv("data/results/model_performance.csv", index=False)

# Print results
print(results_df)

# Plot Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    ax = axes[i]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix: {name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.savefig("Visualizations/Modelling/confusion_matrices.png")
plt.show()

# Plot ROC Curve
plt.figure(figsize=(8, 6))
for name, (fpr, tpr) in roc_curves.items():
    plt.plot(fpr, tpr, label=name)

plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Baseline
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("Visualizations/Modelling/roc_curve.png")
plt.show()
