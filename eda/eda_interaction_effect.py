# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned dataset
df = pd.read_csv("data/cleaned_loan_default_data.csv")

# Setting style for the visualizations
sns.set_style("whitegrid")
plt.rcParams.update({'axes.titlesize': 14, 'axes.labelsize': 12})

# Display the first few rows to confirm dataset is loaded
df.head()

### ---- Visualizing Interaction Effects ---- ###

# Step 1: Pairplot to visualize interactions between numerical features
numerical_features = ['age', 'income', 'loan_amount']

# Pairplot shows pairwise relationships in the dataset
sns.pairplot(df[numerical_features], diag_kind='kde', corner=True, plot_kws={"s": 25, "alpha": 0.6})
plt.suptitle("Pairwise Relationships between Numerical Features", fontsize=16, fontweight="bold", y=1.02)
plt.show()

# Step 2: Interaction plot for loan amount and age, showing loan default as a hue
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='age', y='loan_amount', hue='has_default', palette="coolwarm", s=50, alpha=0.7)
plt.title('Interaction between Age and Loan Amount by Loan Default Status', fontsize=14, fontweight='bold')
plt.xlabel('Age')
plt.ylabel('Loan Amount')
plt.show()

### ---- Feature Engineering: Creating Interaction Terms ---- ###

# PolynomialFeatures helps to create interaction terms
interaction = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
interaction_terms = interaction.fit_transform(df[['age', 'income', 'loan_amount']])

# Converting interaction terms into a DataFrame
interaction_df = pd.DataFrame(interaction_terms, columns=["age", "income", "loan_amount", "age*income", "age*loan_amount", "income*loan_amount"])

# Combine with original dataset for modeling
df_interaction = pd.concat([df, interaction_df], axis=1)

### ---- Modeling: Logistic Regression with Interaction Terms ---- ###

# Define feature set including interaction terms and target variable
X = df_interaction[["age", "income", "loan_amount", "age*income", "age*loan_amount", "income*loan_amount"]]
y = df_interaction["has_default"]

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression to model interaction effects
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = logreg.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

### ---- Interaction Term Coefficients ---- ###
# Coefficients from the model will help to see the effect size of interaction terms
coef_df = pd.DataFrame(logreg.coef_, columns=["age", "income", "loan_amount", "age*income", "age*loan_amount", "income*loan_amount"])
print("\nLogistic Regression Coefficients for Interaction Terms:")
print(coef_df.T)

### ---- Heatmap of Interaction Correlation ---- ###

# Step 3: Heatmap to understand how variables correlate, especially interaction terms
plt.figure(figsize=(10, 6))
corr_matrix = df_interaction[["age", "income", "loan_amount", "age*income", "age*loan_amount", "income*loan_amount"]].corr()

# Plot heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap with Interaction Terms", fontsize=16, fontweight="bold")
plt.show()

### ---- Conclusion ---- ###
# The pairplots and interaction plots provide a visual sense of how the features relate.
# The logistic regression model shows the statistical importance of these interaction terms.
