import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("data/churn.csv")

# Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_scaled, y)

# Streamlit UI
st.title("📊 Customer Churn Prediction App")

st.write("Adjust customer details:")

# Numeric inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.slider("Monthly Charges", 0, 150, 50)
total_charges = tenure * monthly_charges

# Categorical inputs
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
payment = st.selectbox("Payment Method", 
                       ["Electronic check", "Mailed check", 
                        "Bank transfer (automatic)", 
                        "Credit card (automatic)"])

senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.selectbox("Has Dependents?", ["No", "Yes"])

if st.button("Predict"):

    # Start from mean values
    input_data = pd.DataFrame(X.mean()).T

    # Update numeric features
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges

    # Update categorical features (dummy encoded columns)
    for col in input_data.columns:
        if contract in col:
            input_data[col] = 1
        if internet in col:
            input_data[col] = 1
        if payment in col:
            input_data[col] = 1
        if senior == "Yes" and "SeniorCitizen" in col:
            input_data[col] = 1
        if partner == "Yes" and "Partner_Yes" in col:
            input_data[col] = 1
        if dependents == "Yes" and "Dependents_Yes" in col:
            input_data[col] = 1

    input_scaled = scaler.transform(input_data)

    probability = model.predict_proba(input_scaled)[0][1]

    st.write(f"Churn Probability: {probability:.2f}")

    if probability > 0.35:
        st.error("⚠ Customer is likely to churn!")
    else:
        st.success("✅ Customer is likely to stay.")
