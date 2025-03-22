import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Load dataset
df = pd.read_csv("data/cleaned_loan_default_data.csv")

# 1️⃣ Handle Missing Values
df = df.dropna(subset=['age', 'income', 'loan_amount'])  # Ensure no missing values in key features

# 2️⃣ Feature Binning (Age, Income, Loan Amount)
# Age Bins
age_bins = [18, 25, 35, 45, 55, 65, float('inf')]
age_labels = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

# Income Bins
income_bins = [0, 40000, 80000, 120000, 160000, float('inf')]
income_labels = ['Low', 'Moderate', 'Middle', 'Upper-Middle', 'High']
df['income_group'] = pd.cut(df['income'], bins=income_bins, labels=income_labels, right=False)

# Loan Amount Bins (Fixed Issue)
loan_bins = [-1, 10000, 20000, 30000, 40000, 50000, float('inf')]  
loan_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High', 'Extreme']
df['loan_category'] = pd.cut(df['loan_amount'], bins=loan_bins, labels=loan_labels, right=False)

# 3️⃣ One-Hot Encoding for Categorical Variables
categorical_features = ['marital_status', 'employment_status', 'loan_purpose', 'age_group', 'income_group', 'loan_category']
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drop first category to avoid multicollinearity
encoded_data = encoder.fit_transform(df[categorical_features])
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_features))

# 4️⃣ Scaling Numerical Features
scaler = MinMaxScaler()
numerical_features = ['age', 'income', 'loan_amount']
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# 5️⃣ Merge Encoded Data & Drop Original Categorical Columns
df = df.drop(columns=categorical_features).reset_index(drop=True)
df = pd.concat([df, encoded_df], axis=1)

# Save Processed Data
df.to_csv("data/processed_loan_default_data.csv", index=False)

print("Feature Engineering Completed Successfully! 🚀")
