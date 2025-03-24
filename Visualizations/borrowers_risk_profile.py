import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = 'data/balanced_loan_default_data.csv'  # Adjust to your file path
df = pd.read_csv(file_path)

# Filter defaulters
defaulters = df[df['has_default'] == 1]

# Define age group columns (modify this if needed based on the exact column names in your dataset)
age_columns = [col for col in df.columns if 'age_group' in col]

# Define marital status columns
marital_status_columns = [col for col in df.columns if 'marital_status' in col]

# Define loan purpose columns
loan_purpose_columns = [col for col in df.columns if 'loan_purpose' in col]

# Create the likely defaulter profile based on key features
def defaulter_profile():
    # 1. Age Group Analysis
    age_group_defaulter = defaulters[age_columns].sum().idxmax().replace('age_group_', '').replace('-', ' to ')
    
    # 2. Marital Status Analysis
    marital_status_defaulter = defaulters[marital_status_columns].sum().idxmax().replace('marital_status_', '').capitalize()
    
    # 3. Loan Amount Analysis
    avg_loan_amount = defaulters['loan_amount'].mean()
    
    # 4. Loan Purpose Analysis
    loan_purpose_defaulter = defaulters[loan_purpose_columns].sum().idxmax().replace('loan_purpose_', '').capitalize()
    
    profile = {
        'Likely Defaulter Age Group': age_group_defaulter,
        'Likely Defaulter Marital Status': marital_status_defaulter,
        'Average Loan Amount for Defaulters': round(avg_loan_amount, 2),
        'Most Common Loan Purpose for Defaulters': loan_purpose_defaulter
    }
    
    return profile

# Get defaulter profile
defaulter_profile_output = defaulter_profile()
print(defaulter_profile_output)

# Visualizations

# 1. Age Group Visualization
if len(age_columns) > 0:
    defaulters_age_group = defaulters[age_columns].sum()
    defaulters_age_group.plot(kind='bar', color='skyblue', figsize=(8,6))
    plt.title('Defaulters by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Number of Defaulters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No age group columns found.")

# 2. Marital Status Visualization
if len(marital_status_columns) > 0:
    defaulters_marital_status = defaulters[marital_status_columns].sum()
    defaulters_marital_status.plot(kind='bar', color='lightgreen', figsize=(8,6))
    plt.title('Defaulters by Marital Status')
    plt.xlabel('Marital Status')
    plt.ylabel('Number of Defaulters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No marital status columns found.")

# 3. Loan Amount Visualization (Histogram)
if 'loan_amount' in defaulters.columns:
    plt.figure(figsize=(8,6))
    sns.histplot(defaulters['loan_amount'], bins=20, color='coral')
    plt.title('Distribution of Loan Amounts for Defaulters')
    plt.xlabel('Loan Amount')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
else:
    print("Loan amount column not found.")

# 4. Loan Purpose Visualization
if len(loan_purpose_columns) > 0:
    defaulters_loan_purpose = defaulters[loan_purpose_columns].sum()
    defaulters_loan_purpose.plot(kind='bar', color='lightblue', figsize=(8,6))
    plt.title('Defaulters by Loan Purpose')
    plt.xlabel('Loan Purpose')
    plt.ylabel('Number of Defaulters')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("No loan purpose columns found.")
