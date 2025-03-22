#Import Required Libraries
import pandas as pd
import numpy as np

#Define the Number of Samples
num_samples = 10000

#Generate Random Data
np.random.seed(42)  # Ensure reproducibility

data = pd.DataFrame({
    "loan_id": np.arange(1, num_samples + 1),
    "age": np.random.randint(18, 70, num_samples),
    "income": np.random.randint(20000, 150000, num_samples),
    "loan_amount": np.random.randint(1000, 50000, num_samples),
    "loan_purpose": np.random.choice(["personal", "business", "education", "medical"], num_samples),
    "employment_status": np.random.choice(["employed", "unemployed", "self-employed"], num_samples),
    "marital_status": np.random.choice(["single", "married", "divorced"], num_samples),
    "has_default": np.random.choice([0, 1], num_samples, p=[0.8, 0.2])  # 20% default rate
})

# Save the Dataset as CSV
data.to_csv("D:\Project_portfolio\Loan_default_prediction(Python)\data", index=False)
print("Dataset saved successfully!")
