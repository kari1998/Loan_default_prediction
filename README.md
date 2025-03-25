# **📊 AI-Powered Loan Default Prediction**  

## **Project Overview**  
This project aims to develop an **AI-driven loan default prediction model** using Python, advanced machine learning techniques, and interactive tools. The objective is to enhance financial decision-making by identifying high-risk borrowers and improving loan risk assessment strategies. The results will be showcased through an **interactive dashboard** and a **LinkedIn case study**, making the insights both accessible and actionable.  

## **💼 Business Problem**  
Lending institutions face significant financial risks due to loan defaults, which can result in **revenue losses** and **operational challenges**. Traditional models that rely on **historical financial data and credit scores** may not fully capture the complexity of borrowers' financial behaviors. This project leverages **AI-driven predictive modeling** to offer a **more accurate, data-driven approach** to assessing loan default risks, reducing default rates, and optimizing the loan approval process.  

## **🔑 Key Questions**  
This project seeks to answer the following critical questions:  
1️⃣ **What factors significantly impact loan defaults?**  
2️⃣ **How do AI models compare to traditional models in predicting loan defaults?**  
3️⃣ **How can borrowers' risk profiles be visualized to improve decision-making?**  
4️⃣ **How can an interactive dashboard provide real-time loan risk insights to stakeholders?**  

## **🗂 Data Review**  

### **📌 Dataset Overview**  
The dataset used in this project consists of borrower attributes, loan information, and default indicators that are key to predicting the likelihood of loan repayment default.  

### **📊 Key Features**  
- **Borrower Information:** Age, employment status, marital status  
- **Loan Details:** Loan amount, loan purpose, loan status  
- **Default Indicator:** Whether the borrower defaulted (1) or not (0)  

## **🔧 Data Cleaning & Preprocessing**  

### **Step 1: Handling Missing Values**  
- Missing values were identified and imputed where appropriate (e.g., median imputation for numerical features and mode imputation for categorical features).  

### **Step 2: Encoding Categorical Variables**  
- **One-hot encoding** was applied to categorical variables to enable machine learning model training.  

### **Step 3: Feature Engineering**  
- Created new features such as **Debt-to-Income Ratio (DTI)** and **Credit Score Risk Levels** to improve model interpretability.  

### **Step 4: Normalization & Scaling**  
- Numerical features were standardized using **MinMaxScaler** for models sensitive to feature scaling.  

## **📊 Exploratory Data Analysis (EDA)**  
As part of the project, we conducted both univariate and bivariate analyses to explore key attributes in the dataset and their relationship to loan default rates. The following summarizes the most significant findings from the EDA:

#### Key Findings:  


 
   ### **Correlation Analysis**  
A **correlation heatmap** was generated to explore relationships between numerical variables.  

#### **Findings from the Correlation Heatmap:**  
- **No strong correlations** were observed between `age`, `income`, `loan_amount`, and `has_default`.  
- The **low correlation (~0.00)** between numerical features and `has_default` indicates that default behavior might be influenced more by categorical variables or complex, non-linear relationships, reinforcing the use of AI models.  


## **📊 Model Training & Evaluation**  

### **🔬 Models Used**  
- **Traditional Models:** Logistic Regression, Random Forest, XGBoost  
- **AI Models:** Neural Networks  

### **📈 Performance Metrics**  
| Model               | Accuracy | Precision | Recall | F1-score | AUC-ROC |  
|--------------------|----------|-----------|--------|----------|---------|  
| Logistic Regression | 52.89%    | 53.00%     | 53.00%  | 53.00%    | -    |  
| Random Forest      | **84.63%** | 85.00%     | 85.00%  | 85.00%    | **-** |  
| XGBoost           | **83.00%** | 85.00%     | 83.00%  | 83.00%    | **-** |  
| Neural Network     | 56.13%    | 56.00%     | 56.00%  | 56.00%    | -    |  

📌 **Random Forest and XGBoost emerged as the most reliable models.**  

### **Loan Default by Age Group**  
📂 **Visual Output:** `Visualiation/borrower_risk/Defaulters_by_Age_Group.png`  
- The highest default rates are observed in the **35-44 and 55-64** age groups.  
- **Younger and older borrowers show lower default rates**.  

📌 **This suggests that middle-aged borrowers may be at higher risk of default.**  

### **Loan Default by Loan Purpose**  
📂 **Visual Output:** `Visualiation/borrower_risk/Defaulters_by_Loan_Purpose.png`  
- **Education loans show a significant number of defaults**.  
- Other categories like **personal and business loans** also exhibit moderate risk.  

📌 **Loan purpose influences default risk, with education loans being particularly risky.**  

### **Loan Default by Marital Status**  
📂 **Visual Output:** `Visualiation/borrower_risk/Defaulters_by_Marriage_Status.png`  
- **Married individuals have the highest default count**, closely followed by **single borrowers**.  

📌 **Marital status plays a role, but further analysis is needed to determine whether income stability or family responsibilities contribute to this trend.**  


## **📊 Model Insights & Business Recommendations**  
1️⃣ **Borrowers aged 35-44 and 55-64 are at higher risk of default.**  
2️⃣ **Education loans show the highest default rates, suggesting financial constraints for borrowers in this category.**  
3️⃣ **Married individuals exhibit higher default rates, possibly due to financial commitments.**  
4️⃣ **AI models provide superior prediction accuracy, making them valuable for risk assessment.**  
5️⃣ **Lenders should implement targeted risk mitigation strategies for high-risk borrower groups.**  

## **📊 Next Steps**  
- **Deploy the model via an interactive dashboard for real-time predictions.**  
- **Conduct further analysis on income stability and repayment behavior.**  
- **Optimize loan approval strategies based on high-risk borrower segments.**  

## **📂 Project Structure**  
- `data/` → Contains raw and cleaned datasets , models perfomance files 
- `data_cleaning/` → contains data cleaning scripts 
- `eda/` → Saved eda phase of my analysis  
- `feature_engineering/` → Contains all the scripts that handled the processing of the data in preparation of modeling (checking and handling imbalance, feature_engineering) 
- `modelling/` →  Trained and saved all the models, feature importance and model evaluation


## **📢 Conclusion**  
This project successfully built an **AI-powered loan default prediction model**, providing **data-driven insights** into borrower risk. The findings highlight the impact of **age, marital status, and loan purpose** on default rates. With **Random Forest and XGBoost performing best**, this model is well-positioned to assist financial institutions in making informed lending decisions.  

📌 **Next step: Deploy the model and dashboard for real-world application.** 🚀  

