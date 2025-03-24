
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


### **Outlier Detection**  
**Objective:** Identify and handle extreme values in key numerical variables (`age`, `income`, `loan_amount`) to ensure data quality.  

**Method:** Used the **Interquartile Range (IQR)** method to detect outliers.  

**Results:**  
- **Outliers in Age:** 0  
- **Outliers in Income:** 0  
- **Outliers in Loan Amount:** 0  

📌 No outliers were detected, so no further action was needed.  

---

### **Boxplots for Numerical Variables**  
📂 **Visual Output:** `Boxplot.png`  

Boxplots provide insights into the spread, central tendency, and presence of outliers.  

- **Age:** Median age is around **40 years**, ranging from early **20s to about 70**. A few potential outliers exist in the **lower range**.  
- **Income:** Median income is approximately **$80,000–$100,000**, with some **high-income outliers above $140,000**.  
- **Loan Amount:** Median loan amount is **around $25,000**, ranging from a few thousand dollars to **$50,000**, with **no extreme outliers**.  

📌 Income and loan amounts show **broad distributions**, while age is **fairly symmetric** with a wider middle-aged range.  

---

### **Categorical Variable Distributions**  
📂 **Visual Output:** `Distribution_of_categorical_features.png`  

Bar plots display how frequently each category appears.  

- **Loan Purpose:** Even distribution across **personal, medical, education, and business loans**, with **slightly fewer business loans**.  
- **Employment Status:** **Balanced representation** across self-employed, employed, and unemployed individuals.  
- **Marital Status:** Fair distribution among **married, divorced, and single**, with a **slight dominance of married applicants**.  

📌 The dataset is **well-balanced across categories**, reducing bias.  

---

### **Numerical Feature Distribution**  
📂 **Visual Output:** `Distribution_of_numerical_features.png`  

Histograms provide a detailed view of numerical data distributions.  

- **Age:** Uniformly distributed, meaning **all age groups are fairly represented**.  
- **Income:** Fairly uniform across salary ranges, indicating **a well-distributed applicant base**.  
- **Loan Amount:** Evenly spread across different loan sizes, with **no extreme spikes**.  

📌 The dataset **fairly represents different financial profiles**, reducing the likelihood of bias.  

---

### **Overall Insights & Business Implications**  
- The dataset includes a **diverse applicant pool** across employment status, marital status, and loan purpose.  
- **Loan sizes vary widely**, requiring different risk models.  
- **Income variance is high**, making **income a key factor in loan approval decisions**.  
- Further analysis should explore **income-default correlation, loan purpose repayment trends, and borrower risk profiles**.  

---

### **Loan Default by Marital Status**  
📂 **Visual Output:** `Loan_Default_by_Marital_Status.png`  

- Default counts are similar across groups:  
  - **Married:** 685 defaults  
  - **Single:** 663 defaults  
  - **Divorced:** 697 defaults  
- The **total number of loan holders is highest for married individuals**, but their **default count is not significantly higher**.  

📌 **Marital status does not strongly influence loan default risk**.  

---

### **Loan Default by Loan Purpose**  
📂 **Visual Output:** `Loan_Default_by_Loan_Purpose.png`  

- **Personal loans** (526 defaults) have the highest count.  
- **Medical (492 defaults)** and **business (515 defaults)** loans show slightly lower defaults.  

📌 **Personal and business loans may carry a higher default risk**.  

---

### **Loan Default by Employment Status**  
📂 **Visual Output:** `Loan_Default_by_Employment_Status.png`  

- **Self-employed borrowers** have the highest default count (**703 defaults**).  
- **Employed (669 defaults) and unemployed (673 defaults)** show slightly lower numbers.  

📌 **Self-employed borrowers have the highest risk, likely due to income instability**.  

---

### **Age vs Loan Default**  
📂 **Visual Output:** `Age_vs_Loan_Default.png`  

- **Mean age of defaulters (43.77) and non-defaulters (43.48) is nearly identical**.  
- **No significant trend** between age and default.  

📌 **Age does not strongly impact loan default risk**.  

---

### **Income vs Loan Default**  
📂 **Visual Output:** `Income_vs_Loan_Default.png`  

- **Mean income for defaulters ($84,415) and non-defaulters ($84,757) is nearly the same**.  
- **No strong correlation between income and default behavior**.  

📌 **Income alone does not determine loan default risk**.  

---

### **Loan Amount vs Loan Default**  
📂 **Visual Output:** `Loan_Amount_vs_Loan_Default.png`  

- **Defaulters have a slightly higher average loan amount ($25,377) than non-defaulters ($25,318), but the difference is small**.  
- **No strong distinction between defaulters and non-defaulters based on loan amount**.  

📌 **Loan amount alone does not predict default risk**.  

---

### **Final Takeaways**  
✅ **Marital status does not strongly impact default risk**.  
✅ **Personal and business loans show slightly higher default rates**.  
✅ **Self-employed borrowers have the highest risk, suggesting income stability is a key factor**.  
✅ **Age, income, and loan amount do not show strong individual influence on default risk**.  

📌 **AI models may need to capture complex interactions between variables for better predictions**.



