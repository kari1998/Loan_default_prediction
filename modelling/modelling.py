import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure output directory exists
output_dir = "modelling"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
data = pd.read_csv("data/balanced_loan_default_data.csv")  # UPDATED FILE

# Define features and target variable
X = data.drop(columns=["has_default"])
y = data["has_default"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Logistic Regression ---
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)

log_accuracy = accuracy_score(y_test, y_pred_log)
print(f"Logistic Regression Accuracy: {log_accuracy:.4f}")
print(classification_report(y_test, y_pred_log))

# Save model & confusion matrix
joblib.dump(log_reg, os.path.join(output_dir, "logistic_regression_model.pkl"))
pd.DataFrame(confusion_matrix(y_test, y_pred_log)).to_csv(os.path.join(output_dir, "logistic_regression_conf_matrix.csv"), index=False)

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
print(classification_report(y_test, y_pred_rf))

# Save model & confusion matrix
joblib.dump(rf, os.path.join(output_dir, "random_forest_model.pkl"))
pd.DataFrame(confusion_matrix(y_test, y_pred_rf)).to_csv(os.path.join(output_dir, "random_forest_conf_matrix.csv"), index=False)

# --- XGBoost ---
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")
print(classification_report(y_test, y_pred_xgb))

# Save model & confusion matrix
joblib.dump(xgb, os.path.join(output_dir, "xgboost_model.pkl"))
pd.DataFrame(confusion_matrix(y_test, y_pred_xgb)).to_csv(os.path.join(output_dir, "xgboost_conf_matrix.csv"), index=False)

# --- Neural Network ---
nn_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, verbose=1)

y_pred_nn = (nn_model.predict(X_test_scaled) > 0.5).astype(int)

nn_accuracy = accuracy_score(y_test, y_pred_nn)
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")
print(classification_report(y_test, y_pred_nn))

# Save model & confusion matrix
nn_model.save(os.path.join(output_dir, "neural_network_model.h5"))
pd.DataFrame(confusion_matrix(y_test, y_pred_nn)).to_csv(os.path.join(output_dir, "neural_network_conf_matrix.csv"), index=False)

# Print summary of results
print("\nModel Performance Summary:")
print(f"Logistic Regression: {log_accuracy:.4f}")
print(f"Random Forest: {rf_accuracy:.4f}")
print(f"XGBoost: {xgb_accuracy:.4f}")
print(f"Neural Network: {nn_accuracy:.4f}")
