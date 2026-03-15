# TS-ACADEMY-GROUP-12-CAPSTONE-PROJECT

### Project Overview
The capstone project is a TS Academy Group 12 data science project that analyzes financial transaction data to identify patterns associated with fraudulent activities. This project demonstrates a supervised machine learning workflow, including data preparation, exploratory data analysis, visualization, and model optimization using Random Forest. 
With over 5.4 million synthetic simulator (PaySim) transaction data, our goal was to understand patterns within transaction data and build a machine learning model capable of identifying potentially fraudulent transactions.
The objective was to transition from a linear baseline model to an advanced ensemble architecture in order to achieve maximum detection sensitivity (Recall) while effectively handling high-dimensional financial data.

### Dataset
The dataset used for this analysis is the PaySim Fraud Detection Dataset, which simulates mobile money transactions and includes both legitimate and fraudulent activities.

Dataset: PaySim (With_Aggregated_Version) - https://www.kaggle.com/datasets/chendoytshman/fraud-detection-paysim

Scale: **5,420,481 rows** | **27 columns**

Target: **isFraud** (Highly imbalanced: <1% Fraud)

**Dataset Structure:**

PaySim Fraud Detection Dataset on Kaggle contains transaction information such as:

Transaction type

Transaction amount

Sender balance before and after transaction

Receiver balance before and after transaction  

### Workflow
**Data Preparation**

The first stage of the project focused on preparing the dataset for analysis using Pandas.

The following tasks were carried out:\
* Previewing the dataset using .head(), .tail(), and .sample()

Inspecting dataset structure using .info()

Checking dataset dimensions using .shape

Listing column names

Generating descriptive statistics using .describe()

Identifying zero values in the dataset

Checking for missing values

Exploring unique values in categorical columns

Computing value counts for categorical and integer columns

Filtering relevant columns for analysis

Renaming columns to improve readability

The preparation stage ensured that the dataset was well understood before further analysis was performed.


**Data Preprocessing and Exploratory Data Analysis**

In the preprocessing stage, the dataset was transformed and explored to understand relationships between variables.

Encoding Categorical Data

The Transaction_type column was the only categorical variable in text format. It was encoded using LabelEncoder from the scikit-learn library.

A new column named transaction_type_encoded was created where each category was assigned a numerical label:

0 – CASH_IN

1 – CASH_OUT

2 – DEBIT

3 – PAYMENT

4 – TRANSFER

This transformation allowed the machine learning model to process the categorical feature.

Fraud Distribution Analysis

The distribution of the fraud label was examined using:

df['fraud_label'].value_counts(normalize=True)

The analysis showed that fraudulent transactions represented a small proportion of the dataset, confirming that the data is highly imbalanced, which is common in fraud detection problems.

Transaction Type vs Fraud

A cross-tabulation analysis was performed:

pd.crosstab(df['Transaction_type'], df['fraud_label'])

This analysis helped identify which transaction types were more commonly associated with fraudulent behavior.
Fraud label indicating whether the transaction is fraudulent
