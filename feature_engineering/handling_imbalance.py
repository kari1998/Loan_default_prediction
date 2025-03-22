import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load processed dataset
df = pd.read_csv("processed_loan_default_data.csv")

# Separate features and target variable
X = df.drop(columns=["has_default"])
y = df["has_default"]

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convert back to DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled["has_default"] = y_resampled  # Add target variable back

# Check class distribution after SMOTE
print("Class Distribution After SMOTE:", Counter(y_resampled))

# Save the balanced dataset
df_resampled.to_csv("balanced_loan_default_data.csv", index=False)

print("SMOTE applied successfully. Balanced dataset saved as 'balanced_loan_default_data.csv'.")
