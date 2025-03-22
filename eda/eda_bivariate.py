import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_loan_default_data.csv")

### Step 1: Numerical Variables vs. Default Status ###
num_vars = ["age", "income", "loan_amount"]

for var in num_vars:
    plt.figure(figsize=(8, 5))
    
    # Boxplot with mean/median lines
    ax = sns.boxplot(x=df["has_default"], y=df[var], palette=["#1f77b4", "#d62728"])
    
    # Add mean lines
    mean_values = df.groupby("has_default")[var].mean()
    for i, mean_val in enumerate(mean_values):
        plt.axhline(mean_val, linestyle="dashed", color="black", alpha=0.7)
        plt.text(i, mean_val, f"Mean: {mean_val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    plt.title(f"{var.capitalize()} vs Loan Default")
    plt.xlabel("Loan Default (0 = No, 1 = Yes)")
    plt.ylabel(var.capitalize())
    plt.show()

### Step 2: Categorical Variables vs. Default Status ###
cat_vars = ["loan_purpose", "employment_status", "marital_status"]

for var in cat_vars:
    plt.figure(figsize=(10, 6))
    
    # Compute default rates for sorting
    default_rates = df.groupby(var)["has_default"].mean().sort_values()
    sorted_categories = default_rates.index

    # Create count plot with sorted order
    ax = sns.countplot(data=df, x=var, hue="has_default", palette=["#1f77b4", "#d62728"], order=sorted_categories)

    # Add percentage labels on bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom', fontsize=10, fontweight="bold")

    plt.title(f"Loan Default by {var.replace('_', ' ').capitalize()}")
    plt.xlabel(var.replace('_', ' ').capitalize())
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title="Has Default", labels=["No Default", "Default"])
    plt.show()

print("Bivariate analysis completed. Plots displayed.")
