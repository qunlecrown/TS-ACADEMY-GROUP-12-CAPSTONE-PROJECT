# Machine Learning-Based Fraud Detection in Mobile Money Transactions
**TS-ACADEMY-GROUP-12-CAPSTONE-PROJECT**

### Project Overview
The capstone project is a TS Academy Group 12 data science project that analyzes financial transaction data to identify patterns associated with fraudulent activities. This project demonstrates a supervised machine learning workflow, including data preparation, exploratory data analysis, visualization, and model optimization using Random Forest. 
With over 5.4 million synthetic simulator (PaySim) transaction data, our goal was to understand patterns within transaction data and build a machine learning model capable of identifying potentially fraudulent transactions.
The objective was to transition from a linear baseline model to an advanced ensemble architecture in order to achieve maximum detection sensitivity (Recall) while effectively handling high-dimensional financial data.

#### Dataset
The dataset used for this analysis is the PaySim Fraud Detection Dataset, which simulates mobile money transactions and includes both legitimate and fraudulent activities.

Dataset: PaySim (With_Aggregated_Version)

Scale: **5,420,481 rows** | **27 columns**

Target: **isFraud** (Highly imbalanced: <1% Fraud)

🔗 Dataset Link  
https://www.kaggle.com/datasets/chendoytshman/fraud-detection-paysim

## Data Preparation
The project analysis began with an exploration of the dataset to understand its structure and characteristics. The dataset contains over five million records and multiple features describing transaction behavior. Initial inspection involved **viewing sample records**, **checking the dataset shape**, and **reviewing data types** to understand the format of each variable. **Descriptive statistics** were generated to summarize numerical features, while checks for **missing and zero values** ensured the dataset was suitable for further analysis.  
Categorical columns were explored to identify unique values and frequency distributions. Some columns were filtered and renamed where necessary to improve readability and prepare the dataset for analysis


## Preprocessing and Exploratory Data Analysis (EDA)  
In the preprocessing stage, the dataset was transformed and explored to understand relationships between variables.  
**Encoding Categorical Data**  
The Transaction_type column was the only categorical variable in text format. It was encoded using LabelEncoder from the scikit-learn library.
During preprocessing, the categorical transaction type variable was transformed into numerical form using label encoding from scikit-learn library so that it could be used in machine learning models. The encoded values represented different transaction types such as:  
- `transaction_type` (CASH_IN, CASH_OUT, TRANSFER, DEBIT etc.)

**Exploratory analysis** was then conducted to understand the distribution of fraudulent and non-fraudulent transactions. The results showed that fraudulent transactions represent a very small proportion of the dataset, indicating a strong class imbalance. Additional analysis examined relationships between transaction types, transaction amounts, and account balances to determine patterns associated with fraudulent activity.

**Data Visualization** was then used to better understand the structure of the dataset and highlight the relationsip between variables. Correlation Analysis was then performed to identify relationships between numerical features such as - `transaction_amount`.

## Model Development and Evaluation

**Machine Learning Models**  
The Logistic Regression model and Random Forest Classification model were used in the evaluation stage.

**Logistic Regression**  
Logistic Regression served as the **baseline model** to determine whether fraudulent transactions could be separated using a linear classification approach.
The model estimates the probability that a transaction is fraudulent based on weighted combinations of input features.

**Random Forest**  
The advanced model used in the project was Random Forest, an ensemble learning algorithm composed of multiple decision trees. Each tree produces a prediction, and the final classification is determined through majority voting.
Random Forest was selected because it can capture complex non-linear relationships between transaction amount, transaction timing, and account behavior.

To further improve the model’s performance, **hyperparameter tuning** was performed using **Grid Search Cross-Validation (GridSearchCV)** to evaluate different combinations of hyperparameters. The search used a parameter grid consisting of **162 unique combinations**, calculated from the possible values for five hyperparameters. Each combination was evaluated using **3-fold cross-validation** on the balanced training dataset containing **97,066 samples**, resulting in a total of **486 model training runs**.

The evaluation metric used during the tuning process was `ROC–AUC`, which measures the model’s ability to distinguish between fraudulent and non-fraudulent transactions across different classification thresholds.The best parameters were `max_depth = 10`, `max_features = 'sqrt'`, `min_samples_leaf = 25`, `min_samples_split = 50`, and `n_estimators = 100`, achieving a `ROC-AUC of 0.9993`. The tuned model slightly improved performance: precision increased to **0.60**, recall to 0.98, and `F1 score to 0.75`, indicating better fraud detection with fewer missed cases and false alarms.



## Conclusion and Recommendation  

**Recommendation**  

Based on the findings of the analysis, the organisation should strengthen its fraud prevention framework by focusing on the most significant risk indicators identified in the dataset. First, real-time velocity monitoring should be implemented so that transactions associated with unusually high values of total_sent_last_1hr, particularly those above the 75th percentile threshold of 304,943, trigger immediate secondary authentication. This is necessary because rapid spending within a short period emerged as the strongest short-term fraud signal.  
In addition, `TRANSFER` and `CASH_OUT` transactions should be subjected to stricter verification procedures. Given their fraud rates of 8.3% and 2.7% respectively, these transaction types present greater exposure to fraudulent activity, especially where the transaction amount is close to the sender’s available balance. Measures such as one-time passwords or biometric confirmation would help reduce this risk.  
The organisation should also adopt end-of-month fraud response measures. Since Week 4 recorded the highest fraud concentration at 29.78%, compared with 22–24% in earlier weeks, fraud monitoring systems should be more sensitive and investigation teams better prepared during the final seven days of each month.  
Furthermore, repeat receiver accounts linked to fraud should be continuously monitored. The 4,937 mule accounts identified in Stage 3 should be flagged in real time, and any transaction involving them should automatically prompt a fraud review.

Finally, future versions of the fraud detection model should include additional variables such as sender account age, device fingerprint consistency, and geographic velocity. These features would improve the model’s ability to detect more complex fraud patterns not fully captured by the current six-feature model.  

**Conclusion**  

The **TS Academy Capstone Project – Group 12**  project demonstrates a comprehensive fraud detection workflow that includes data preparation, exploratory analysis, visualization, and machine learning model optimization. The analysis revealed key behavioral patterns within transaction data and highlighted the challenge of detecting fraud within highly imbalanced datasets. Exploratory analysis revealed that fraudulent activities occur primarily in **TRANSFER** and **CASH-OUT** transactions, and that fraud cases are extremely rare compared to normal transactions. Visualization techniques further helped identify patterns and anomalies within the data. Through hyperparameter tuning, the **Random Forest model** achieved improved performance, demonstrating its effectiveness in identifying potentially fraudulent transactions within the dataset.

These findings highlights the value of combining data analysis, visualization and machine learning to strengthen fraud detection systems in financial institutions. 
Such systems are essential for improving security in digital financial services and reducing the risk of financial fraud.

Overall, the results show that machine learning models play an important role in improving fraud detection systems and helping financial institutions identify suspicious transactions more effectively.

## Acknowledgments

**Tutor:** Hart Ofigwe  
**Institution:** TS Academy  
**Group 12:** Members

## References

### Data Source
- Lopez-Rojas, E. A., Elmir, A., & Axelson, S. (2016). *PaySim: A financial mobile money simulator for fraud detection*. In 28th European Modeling and Simulation Symposium (EMSS).  
- chendoytshman. (2023). *Fraud Detection - PaySim (with aggregated) [Data set]*. Kaggle. https://www.kaggle.com/datasets/chendoytshman/fraud-detection-paysim

### Software & Libraries
- Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V.,

**License:** Apache License 2.0  
**Copyright:** © 2026 TS Academy Capstone Project – Group 12
