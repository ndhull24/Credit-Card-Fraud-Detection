# Credit-Card-Fraud-Detection

1. Project Overview
This project aims to detect fraudulent transactions using machine learning techniques. Credit card fraud detection is a vital task in financial industries to identify suspicious activities and prevent unauthorized transactions. The dataset contains anonymized transaction features, which are analyzed to classify whether a transaction is legitimate or fraudulent.

2. Objectives
Build a machine learning model to detect fraudulent transactions.
Analyze the dataset to understand trends and patterns.
Handle class imbalance, as fraudulent transactions are rare.
Evaluate the performance of different models to ensure reliable detection.
3. Dataset Overview
The dataset used in this project contains:

Transaction Features (V1, V2, V3, ..., V28): Anonymized features due to confidentiality.
Amount: The transaction amount.
Class: The target variable indicating if a transaction is fraudulent (1) or legitimate (0).
4. Workflow
Step 1: Data Preprocessing
Handling Missing Data: Ensures the dataset is free from missing or NaN values.
Scaling Features: Uses techniques like StandardScaler to normalize features, especially Amount.
Handling Class Imbalance: Applies methods like undersampling or SMOTE (Synthetic Minority Oversampling Technique) to manage the skewed distribution of fraud and non-fraud transactions.
Step 2: Exploratory Data Analysis (EDA)
Distribution Analysis: Visualizes the distribution of fraudulent vs. legitimate transactions.
Correlation Matrix: Identifies correlations between different features.
Transaction Amount Analysis: Examines how the amount varies between fraud and non-fraud transactions.
Step 3: Model Building
Train-Test Split: Splits the data into training and testing sets to evaluate the model performance.
Machine Learning Models Used:
Logistic Regression: A simple, interpretable model.
Random Forest: A more robust and ensemble model.
XGBoost: An optimized model for high performance.
Step 4: Evaluation Metrics
Confusion Matrix: Visualizes True Positives, False Positives, True Negatives, and False Negatives.
Accuracy: Measures the overall performance of the model.
Precision, Recall, and F1-Score: Evaluates the model’s ability to handle imbalanced data accurately.
AUC-ROC Curve: Examines the trade-off between True Positive and False Positive rates.
Step 5: Hyperparameter Tuning
GridSearchCV or RandomizedSearchCV: Optimizes the model by finding the best parameters.
5. Code Snippets Used
Data Scaling:

python
Copy code
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
Handling Imbalanced Data with SMOTE:

python
Copy code
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
Training a Random Forest Model:

python
Copy code
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
Model Evaluation:

python
Copy code
from sklearn.metrics import classification_report, confusion_matrix
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
6. Results
Confusion Matrix: Indicates how well the model identifies frauds while minimizing false positives.
AUC-ROC Score: Helps assess the model’s ability to separate fraudulent transactions from legitimate ones.
Precision & Recall Trade-off: Balances minimizing false alarms (high precision) with capturing most frauds (high recall).
7. Conclusion
This project demonstrates how to build a credit card fraud detection system using machine learning techniques. Handling class imbalance, optimizing models, and evaluating performance are critical steps to ensure accurate detection. This model can help financial institutions reduce fraud by flagging suspicious transactions effectively.
