# TS-ACADEMY-GROUP-12-CAPSTONE-PROJECT

### Project Overview
The capstone project is a TS Academy Group 12 data science project that analyzes financial transaction data to identify patterns associated with fraudulent activities. This project demonstrates a supervised machine learning workflow, including data preparation, exploratory data analysis, visualization, and model optimization using Random Forest. 
With over 5.4 million synthetic simulator (PaySim) transaction data, our goal was to understand patterns within transaction data and build a machine learning model capable of identifying potentially fraudulent transactions.
The objective was to transition from a linear baseline model to an advanced ensemble architecture in order to achieve maximum detection sensitivity (Recall) while effectively handling high-dimensional financial data.

### Dataset
The dataset used for this analysis is the PaySim Fraud Detection Dataset, which simulates mobile money transactions and includes both legitimate and fraudulent activities.

Dataset: PaySim (With_Aggregated_Version)

Scale: **5,420,481 rows** | **27 columns**

Target: **isFraud** (Highly imbalanced: <1% Fraud)

🔗 Dataset Link  
https://www.kaggle.com/datasets/chendoytshman/fraud-detection-paysim


**Data Preprocessing and Exploratory Data Analysis\**

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



## 3. Technical Methodology

### A. Data Cleaning

To ensure efficient processing of the large dataset, the following preprocessing steps were performed:

**Cleaning & Audit**

- Verified **zero missing values**
- Checked for possible **data leakage**

**Feature Encoding**

Categorical variables such as:

- `transaction_type` (CASH_OUT, TRANSFER, etc.)
- `week_group`

were converted into numerical format using **Label Encoding**.

**Normalization**

A **StandardScaler** was applied to:

- `amount`
- balance-related features

This prevented large transaction values from dominating the model’s coefficients.

**Memory Optimization**

Numeric data types were **downcasted** (for example `float64 → float32`), reducing the dataset’s **memory footprint by over 50%**.

### B. Feature Selection & Engineering

Instead of using all raw columns, the analysis focused on features with the **highest predictive value**.

**Aggregated Velocity Features**

Behavioral indicators were created by calculating **transaction counts over 7-day and 30-day windows**, allowing the model to detect bursts of suspicious activity.

**Gini Importance Ranking**

A **Random Forest–based feature importance analysis** was used to remove noise and retain the **top 10 predictive features**.

## 4. Machine Learning Implementation

### Baseline Model: Logistic Regression (LR)

**Purpose**

To establish a performance baseline and test whether the data could be separated using a **linear classification model**.

**Mechanism**

The model estimates the **probability of fraud** based on weighted combinations of input features.

### Advanced Model: Random Forest (RF)

**Purpose**

To capture **complex, non-linear relationships** between transaction time, amount, and historical account behavior.

**Mechanism**

Random Forest is an **ensemble learning algorithm** consisting of multiple decision trees. Each tree produces a classification, and the final prediction is determined through **majority voting**, improving robustness and reducing variance.

## 5. Evaluation Strategy

A structured validation framework was used to ensure reliable performance.

**Train-Test Split**

- **80% Training Data**
- **20% Testing Data**

The original class distribution was preserved.

**Cross-Validation**

Cross-validation ensured that the model did not **overfit specific portions of the dataset**.

**Primary Evaluation Metric**

**Recall (Sensitivity)** was prioritized because, in fraud detection systems, **missing fraudulent transactions (False Negatives)** is far more costly than generating **false alarms (False Positives)**.

## 6. Model Performance & Evolution

### Performance Comparison

| Metric | Logistic Regression (Baseline) | Random Forest (Final Model) |
|------|------|------|
| Accuracy | 99.46% | 99.13% |
| Recall (Catch Rate) | 99.99% | 97.50% |
| Precision | 0.68 | 0.56 |
| F1 Score | 0.81 | 0.71 |

### Elite Feature Discovery (Top Predictors)

Using **Gini Importance**, the Random Forest model identified the following **top predictors**:

1. **Transaction Amount (33.5%)** – The strongest indicator of suspicious activity.  
2. **Tx Count Last 7 Days (21.5%)** – Detects bursts of transaction activity.  
3. **Avg Amount Last 30 Days (17.6%)** – Captures deviations from normal spending patterns.  
4. **Transaction Type (9.9%)** – Certain channels, particularly transfers, carry higher fraud risk.

### Final Confusion Matrix (Test Data)

- **True Positives (Fraud Caught):** 11,830  
- **False Negatives (Missed Fraud):** 303  
- **True Negatives (Legitimate Verified):** 1,062,790  
- **False Positives (False Alarms):** 9,174

## 7. Conclusion

The **TS Academy Capstone Project – Group 12**  project demonstrates a complete fraud detection workflow that includes data preparation, exploratory analysis, visualization, and machine learning model optimization. The analysis revealed key behavioral patterns within transaction data and highlighted the challenge of detecting fraud within highly imbalanced datasets. Exploratory analysis revealed that fraudulent activities occur primarily in **TRANSFER** and **CASH-OUT** transactions, and that fraud cases are extremely rare compared to normal transactions. Visualization techniques further helped identify patterns and anomalies within the data. Through hyperparameter tuning, the **Random Forest model** achieved improved performance, demonstrating its effectiveness in identifying potentially fraudulent transactions within the dataset.

These findings demonstrate how data analysis, visualization and machine learning can support fraud detection systems in financial institutions. 
Such systems are essential for improving security in digital financial services and reducing the risk of financial fraud.

Overall, the results show that machine learning models play an important role in improving fraud detection systems and helping financial institutions identify suspicious transactions more effectively.

## 8. Acknowledgments & License

**Tutor:** Hart Ofigwe  
**Institution:** TS Academy  

## 9. References

### Data Source
- Lopez-Rojas, E. A., Elmir, A., & Axelson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection*. In 28th European Modeling and Simulation Symposium (EMSS).  
- chendoytshman. (2023). *Fraud Detection - PaySim (with aggregated) [Data set]*. Kaggle. https://www.kaggle.com/datasets/chendoytshman/fraud-detection-paysim

### Software & Libraries
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,

**License:** Apache License 2.0  
**Copyright:** © 2026 TS Academy Capstone Project – Group 12

pd.crosstab(df['Transaction_type'], df['fraud_label'])

This analysis helped identify which transaction types were more commonly associated with fraudulent behavior.
Fraud label indicating whether the transaction is fraudulent
