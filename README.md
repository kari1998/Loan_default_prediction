# **📊 AI-Powered Loan Default Prediction**

## **Project Overview**

This project aims to develop an **AI-driven loan default prediction model** using Python, advanced machine learning techniques, and interactive tools. The objective is to enhance financial decision-making by identifying high-risk borrowers and improving loan risk assessment strategies. The results will be showcased through an **interactive dashboard** and a **LinkedIn case study**, making the insights both accessible and actionable.

## **💼 Business Problem**

Lending institutions face significant financial risks due to loan defaults, leading to **revenue losses** and **operational challenges**. Traditional models that rely on **historical financial data and credit scores** may not fully capture the complexity of borrowers' financial behaviors. This project leverages **AI-driven predictive modeling** to provide a **more accurate, data-driven approach** to assessing loan default risks, reducing default rates, and optimizing the loan approval process.

## **🔑 Key Questions**

This project seeks to answer the following critical questions:

1️⃣ **What factors significantly impact loan defaults?**  
2️⃣ **How do AI models compare to traditional models in predicting loan defaults?**  
3️⃣ **How can borrowers' risk profiles be visualized to improve decision-making?**  
4️⃣ **How can an interactive dashboard provide real-time loan risk insights to stakeholders?**  

## **🗂 Data Review**

### **📌 Dataset Overview**

The dataset consists of borrower attributes, loan information, and default indicators that are key to predicting loan repayment outcomes.

### **📊 Key Features**

- **Borrower Information:** Age, employment status, marital status  
- **Loan Details:** Loan amount, loan purpose, loan status  
- **Default Indicator:** Whether the borrower defaulted (1) or not (0)  

## **🔧 Data Cleaning & Preprocessing**

### **Step 1: Load the Dataset**

- Read the dataset (`synthetic_loan_default_data.csv`) using pandas.
- Checked for file existence before loading to avoid errors.

### **Step 2: Check for Duplicates**

- Found **0 duplicate rows**, so no removals were needed.

### **Step 3: Drop Unnecessary Columns**

- Removed the `loan_id` column since it's just an identifier and does not add predictive value.

### **Step 4: Check Unique Values in Categorical Columns**

- **Loan Purpose:** 4 unique values (personal, medical, education, business).
- **Employment Status:** 3 unique values (self-employed, employed, unemployed).
- **Marital Status:** 3 unique values (married, divorced, single).

### **Step 5: Outlier Detection**

**Objective:** Identify and handle extreme values in key numerical variables (`age`, `income`, `loan_amount`) to ensure data quality.  
**Method:** Used the **Interquartile Range (IQR)** method to detect outliers.  
**Results:**  
- **Outliers in Age:** 0  
- **Outliers in Income:** 0  
- **Outliers in Loan Amount:** 0  
  📌 **No outliers were detected, so no further action was needed.**  

### **Step 6: Correlation Analysis (Numerical Features)**

A **correlation heatmap** was generated to analyze relationships between numerical features.

#### **Findings from the Heatmap:**

- **No strong correlations exist** between `age`, `income`, `loan_amount`, and `has_default`.
- **Low correlation (~0.00)** suggests that default behavior may depend on categorical factors or non-linear patterns better captured by AI models.

📌 **The cleaned dataset was saved as** `cleaned_loan_default_data.csv` **for further analysis.**

## **📊 Exploratory Data Analysis (EDA)**

### **📊 Feature Distribution Analysis**

#### **📊 Distribution of Numeric Features**

📂 **Visual Output:** `Distribution_of_Numeric_Features.png`![Loan Default by Age Group](Visualizations/borrower_risk/Defaulters_by_Age_Group.png)


- **Loan Amount:** Right-skewed distribution indicating a few high-value loans.
- **Income Levels:** Right-skewed distribution with a small proportion of borrowers having very high incomes.
- **Credit Scores:** Concentration in the mid-to-high range, indicating moderate-to-good creditworthiness.

📌 **Key Insight:** Higher loan amounts and lower credit scores may increase default risk, requiring further investigation.

#### **📊 Distribution of Categorical Features**

📂 **Visual Output:** `Distribution_of_Categorical_Features.png`

- **Loan Purpose:** Education loans have a higher share of defaults compared to other purposes.
- **Employment Type:** Self-employed borrowers show slightly higher default rates.
- **Marital Status:** Married borrowers exhibit a slightly higher default tendency.

📌 **Key Insight:** Loan purpose and employment type significantly influence default risk. Borrowers with unstable income sources may be at higher risk.

## **📊 Model Training & Evaluation**

### **🔬 Models Used**

- **Traditional Models:** Logistic Regression, Random Forest, XGBoost  
- **AI Models:** Neural Networks  

### **📈 Performance Metrics**

| Model               | Accuracy   | Precision | Recall | F1-score | AUC-ROC |
| ------------------- | ---------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 52.89%     | 53.00%    | 53.00% | 53.00%   | -       |
| Random Forest       | **84.63%** | 85.00%    | 85.00% | 85.00%   | **-**   |
| XGBoost             | **83.00%** | 85.00%    | 83.00% | 83.00%   | **-**   |
| Neural Network      | 56.13%     | 56.00%    | 56.00% | 56.00%   | -       |

📌 **Random Forest and XGBoost delivered the best performance, likely due to their ability to handle non-linear relationships and feature interactions.**

## **📊 Model Insights & Business Recommendations**

1️⃣ **Borrowers aged 35-44 and 55-64 are at higher risk of default.**  
2️⃣ **Education loans have the highest default rates, indicating financial strain for borrowers in this category.**  
3️⃣ **Married individuals exhibit higher default rates, potentially due to additional financial commitments.**  
4️⃣ **AI models, particularly ensemble methods, provide superior prediction accuracy, making them valuable for risk assessment.**  
5️⃣ **Lenders should implement targeted risk mitigation strategies for high-risk borrower segments.**  

📌 **Next step: Deploy the model and dashboard for real-world application.** 🚀


## **📂 Project Structure**  
- `data/` → Contains raw and cleaned datasets , models perfomance files 
- `data_cleaning/` → contains data cleaning scripts 
- `eda/` → Saved eda phase of my analysis  
- `feature_engineering/` → Contains all the scripts that handled the processing of the data in preparation of modeling (checking and handling imbalance, feature_engineering) 
- `modelling/` →  Trained and saved all the models, feature importance and model evaluation


## **📢 Conclusion**  
This project successfully built an **AI-powered loan default prediction model**, providing **data-driven insights** into borrower risk. The findings highlight the impact of **age, marital status, and loan purpose** on default rates. With **Random Forest and XGBoost performing best**, this model is well-positioned to assist financial institutions in making informed lending decisions.  

📌 **Next step: Deploy the model and dashboard for real-world application.** 🚀  

