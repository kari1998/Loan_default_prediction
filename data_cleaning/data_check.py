import pandas as pd
import os

file_path = r"D:\Project_portfolio\Loan_default_prediction(Python)\data\synthetic_loan_default_data.csv"

# Check if the file exists
if os.path.exists(file_path):
    print(f"File found: {file_path}")
    
    # Load dataset
    df = pd.read_csv(file_path)

    # Step 1: Preview Data
    print("\n🔹 First 5 Rows of Data:")
    print(df.head())

    # Step 2: Check Data Types & Missing Values
    print("\n🔹 Dataset Info:")
    print(df.info())

    print("\n🔹 Missing Values per Column:")
    print(df.isnull().sum())

    # Step 3: Explore Data Distribution
    print("\n🔹 Summary Statistics:")
    print(df.describe())

    print("\n🔹 Default Distribution:")
    print(df["has_default"].value_counts())

 