import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("data/cleaned_loan_default_data.csv")

# Set Seaborn style
sns.set_style("whitegrid")
plt.rcParams.update({'axes.titlesize': 14, 'axes.labelsize': 12})  # Improve text readability

### ---- Histogram for Numerical Variables ---- ###
numerical_features = ["age", "income", "loan_amount"]

plt.figure(figsize=(12, 6))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(1, 3, i)
    sns.histplot(df[col], bins=30, kde=True, color="skyblue", edgecolor="black")
    plt.axvline(df[col].mean(), color="red", linestyle="dashed", linewidth=2)  # Mean indicator
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {col}")

plt.suptitle("Distribution of Numerical Features", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

### ---- Boxplots for Numerical Variables ---- ###
plt.figure(figsize=(12, 5))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(y=df[col], color="lightcoral", showfliers=False)  # Hides outliers for better visualization
    plt.ylabel(col)
    plt.title(f"Boxplot of {col}")

plt.suptitle("Boxplots of Numerical Features", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()

### ---- Bar Charts for Categorical Variables ---- ###
categorical_features = ["loan_purpose", "employment_status", "marital_status"]

plt.figure(figsize=(14, 6))
for i, col in enumerate(categorical_features, 1):
    plt.subplot(1, 3, i)
    ax = sns.countplot(x=df[col], palette="pastel", order=df[col].value_counts().index)
    
    # Annotate bars with percentages
    total = len(df[col])
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        ax.annotate(percentage, (p.get_x() + p.get_width() / 2, p.get_height()), 
                    ha='center', va='bottom', fontsize=10, color='black', fontweight='bold')

    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Distribution of {col}")

plt.suptitle("Distribution of Categorical Features", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.show()