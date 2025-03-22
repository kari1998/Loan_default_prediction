import pandas as pd

# Load processed data
df = pd.read_csv("processed_loan_default_data.csv")


# Check class distribution
class_counts = df['has_default'].value_counts(normalize=True) * 100
print("Class Distribution:\n", class_counts)

# Visualize imbalance
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(x=df['has_default'], palette='coolwarm')
plt.title("Loan Default Class Distribution")
plt.xlabel("Has Default (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()
