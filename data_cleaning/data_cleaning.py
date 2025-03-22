import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "data/synthetic_loan_default_data.csv"  
df = pd.read_csv(file_path)

### Step 1: Check for Duplicates ###
num_duplicates = df.duplicated().sum()
if num_duplicates > 0:
    df = df.drop_duplicates()
    print(f"Removed {num_duplicates} duplicate rows.")
else:
    print("No duplicates found.")

### Step 2: Drop Unnecessary Columns ###
df.drop(columns=["loan_id"], inplace=True)
print("Dropped 'loan_id' column.")

### Step 3: Check for Inconsistent Categorical Values ###
print("\n🔹 Unique Values in Categorical Columns:")
for col in ["loan_purpose", "employment_status", "marital_status"]:
    print(f"{col}:\n{df[col].value_counts()}\n")

# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=["number"])

# Check correlation
plt.figure(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()


# Save cleaned data for further analysis
df.to_csv("data/cleaned_loan_default_data.csv", index=False)
print("Cleaned dataset saved successfully.")
