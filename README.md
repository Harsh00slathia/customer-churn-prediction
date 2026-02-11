📊 Customer Churn Prediction
📌 Problem Statement

  Predict whether a telecom customer will churn using demographic and service usage data.

📂 Dataset

  Customer Churn Dataset (Kaggle)

🔎 Exploratory Data Analysis

  Churn is imbalanced

  Month-to-month contracts have higher churn

  Higher monthly charges increase churn risk

  Customers with shorter tenure churn more

🤖 Models Used

  Logistic Regression (with class balancing)
 
  Random Forest Classifier

📈 Final Model

  Logistic Regression with class_weight='balanced'

 Performance:

  Accuracy: 74%

  Recall (Churn=1): 79%

  Precision (Churn=1): 50%

🔥 Key Business Insights

  TotalCharges, MonthlyCharges, and Tenure are top churn drivers

  Improving recall helps reduce customer loss
