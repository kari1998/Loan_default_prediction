import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("data/cleaned_loan_default_data.csv")

# Define a function to detect outliers using IQR
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    return outliers

# Check outliers for each numerical column
outliers_age = detect_outliers(df, "age")
outliers_income = detect_outliers(df, "income")
outliers_loan_amount = detect_outliers(df, "loan_amount")

# Print results
print(f"Outliers in Age: {len(outliers_age)}")
print(f"Outliers in Income: {len(outliers_income)}")
print(f"Outliers in Loan Amount: {len(outliers_loan_amount)}")
