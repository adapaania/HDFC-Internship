
# Credit Risk Modeling & Loan Default Prediction

## 📌 Project Overview

This project implements an **end-to-end credit risk modeling pipeline** to predict borrower loan default risk using customer enquiry behavior and account-level credit data. The work was carried out as part of a **Data Science Internship at HDFC Capital Advisors Ltd.**, focusing on building interpretable and robust models suitable for real-world underwriting decisions.

The pipeline covers **data ingestion, feature engineering, model training, evaluation, and business validation**, following industry-standard practices used in BFSI credit analytics.

---

## 🗂️ Data Description

The modeling dataset is constructed by integrating multiple borrower-level data sources:

* **Enquiry Data (JSON)**

  * Borrower credit enquiries across different credit types
  * Enquiry amounts and timestamps

* **Account Data (CSV)**

  * Loan amounts, repayment behavior, delinquencies, and credit types

* **Flag Data (CSV)**

  * Binary target variable (`TARGET`):

    * `0` → Good Loan
    * `1` → Bad Loan

Each borrower is represented by a **single consolidated record (UID-level aggregation)** to support supervised learning.

## Data Privacy Notice
Due to confidentiality constraints, raw datasets and intermediate data files
used in this project are not included. The repository focuses on demonstrating
feature engineering, modeling, and evaluation methodology.


---

## ⚙️ Feature Engineering

Extensive feature engineering was performed to capture borrower credit behavior:

### 🔹 Enquiry-Based Features

* Total number of enquiries per borrower
* Unique enquiry types per borrower
* Time-windowed enquiry counts:

  * Last **1, 3, 6, 9, and 12 months**
* Monetary aggregates per credit type:

  * Mean, median, minimum, maximum, and total enquiry amount
* First and last enquiry dates per credit type

### 🔹 Account-Level Features

* Loan amount statistics (average, median, min, max)
* Credit-type–wise total exposure
* Repayment behavior indicators (on-time payments, delinquencies)

### 🔹 Feature Reduction

* Recursive Feature Elimination (**RFE**) using **LightGBM**
* Removal of:

  * High-missing columns
  * Near-zero variance features
  * High-cardinality identifiers

---

## 🧪 Exploratory Data Analysis (EDA)

* Target class distribution (Good vs Bad loans)
* Enquiry frequency and amount distributions
* Risk trends across:

  * Enquiry recency
  * Loan amount buckets
  * Credit types
* Visualizations using **Matplotlib** and **Seaborn**

---

## 🔄 Preprocessing Pipeline

* Train–test alignment to avoid feature mismatch
* Categorical encoding:

  * Label Encoding
  * One-Hot Encoding (via `ColumnTransformer`)
* Numerical preprocessing:

  * Mean imputation
  * Min–Max scaling
* Class imbalance handling using **SMOTE**

---

## 🤖 Models Implemented

Multiple models were trained and benchmarked:

* **Logistic Regression** (primary model)
* **XGBoost**
* **LightGBM**

### Hyperparameter Tuning

* GridSearchCV with **ROC-AUC** as the optimization metric

### Final Model Selection

* **Logistic Regression** was selected due to its:

  * Strong balance between **interpretability and performance**
  * Alignment with credit underwriting and regulatory requirements

---

## 📊 Model Evaluation

Evaluation followed **industry-standard credit risk metrics**:

* ROC-AUC
* Gini Coefficient
* KS Statistic
* Confusion Matrix
* Classification Report

### Business Validation

* **Decile analysis**
* **Lift and Gains charts**
* Feature-wise decile plots to assess monotonic risk separation

These analyses ensured that the model provides **stable and interpretable risk ranking**, suitable for underwriting use cases.

---

## 📈 Key Results

* Logistic Regression achieved **AUC-ROC ≈ 0.65** on the test set
* Clear risk separation observed in top deciles
* Model demonstrated stable performance across train, test, and holdout datasets

---

## 🛠️ Tech Stack

* **Programming:** Python
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Modeling:** scikit-learn, XGBoost, LightGBM
* **Imbalance Handling:** imbalanced-learn (SMOTE)

---
## 📂 Project Structure
📂 Project Structure
├── notebooks/
│   ├── Monsoon - Enquiry Data.ipynb
│   ├── Monsoon - Enquiry Data Test.ipynb
│   ├── Monsoon - Accounts Data.ipynb
│   ├── Monsoon - Accounts Data Test.ipynb
│   ├── Monsoon - Merged Train Set.ipynb
│   ├── Monsoon - Merged Test Set.ipynb
│   ├── Monsoon - Logistic Regression Model.ipynb
│   ├── Monsoon - LGBM Model.ipynb
│   └── Monsoon - XGB MODEL.ipynb
└── Readme.md

📝 Note:
- Raw datasets are intentionally NOT included in this repository (confidential).
- To run the notebooks, place the required files locally (e.g., under a private `data/` folder on your machine)
  and update file paths accordingly.


---

## 📌 Author

**Aania Adap**
Data Science | Machine Learning | Credit Risk Analytics

