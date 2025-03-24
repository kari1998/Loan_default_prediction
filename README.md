
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
The dataset used in this project is **synthetically generated** to simulate real-world loan default data. It contains various borrower attributes, loan information, and default indicators that are key to predicting the likelihood of loan repayment default.  

### **📊 Key Features**  
The dataset consists of 10,000 rows, with the following critical features:  
- **Borrower Information:** Age, employment status, marital status  
- **Loan Details:** Loan amount, loan purpose, loan status  
- **Default Indicator:** Whether the borrower defaulted (1) or not (0)  



## **🔧 Data Cleaning Process**  

### **Step 1: Loading the Dataset**  
- The dataset (`synthetic_loan_default_data.csv`) was loaded using **pandas**, with a file existence check to prevent loading errors.  

### **Step 2: Removing Duplicate Records**  
- **Result:** No duplicate rows were found, so no removals were necessary.  

### **Step 3: Dropping Unnecessary Columns**  
- The `loan_id` column was dropped as it is a unique identifier and does not add predictive value.  

### **Step 4: Checking Unique Values in Categorical Columns**  
The following unique values were identified and documented:  

| **Column**            | **Unique Values**                             |  
|-----------------------|-----------------------------------------------|  
| `loan_purpose`        | Personal, Medical, Education, Business       |  
| `employment_status`   | Self-employed, Employed, Unemployed          |  
| `marital_status`      | Married, Divorced, Single                    |  

No inconsistencies or unexpected values were found.  

### **Step 5: Correlation Analysis (Numerical Features)**  
A **correlation heatmap** was generated to explore relationships between numerical variables.  

#### **Findings from the Correlation Heatmap:**  
- **No strong correlations** were observed between `age`, `income`, `loan_amount`, and `has_default`.  
- The **low correlation (~0.00)** between numerical features and `has_default` indicates that default behavior might be influenced more by categorical variables or complex, non-linear relationships, reinforcing the use of AI models.  

📌 **Visual Output:**  
- A correlation heatmap was generated and saved in the EDA folder for reference.  



## **📂 Cleaned Data Output**  
The cleaned dataset was saved for further analysis and modeling as **`cleaned_loan_default_data.csv`**.  
📁 **File Path:** `data/cleaned_loan_default_data.csv`  



## **Summary of Data Cleaning**  
✅ **No duplicate records** detected.  
✅ Removed the `loan_id` column.  
✅ Documented and verified **unique categorical values**.  
✅ Generated a **correlation heatmap** to explore relationships between numerical features.  
✅ Prepared and saved a **cleaned dataset** for the next project stages (EDA, feature engineering, and modeling).  



## **🔍 Data Quality Considerations**  
To ensure data readiness for AI modeling, the following checks and preprocessing tasks were completed:  
✔ **Handling Missing Values:** Addressed missing data.  
✔ **Categorical Variable Encoding:** Prepared categorical columns for AI models.  
✔ **Feature Engineering:** Developed engineered features (e.g., grouped loan purposes).  
✔ **Dataset Consistency Check:** Ensured alignment between cleaned data and AI modeling needs.  



