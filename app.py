import streamlit as st
import pandas as pd
import joblib
import os
from src.data_processing.data_preprocessing import (
    generate_risk_features, drop_unnecessary_features, encode_categorical_features, scale_features
)

st.set_page_config(page_title="Insurance Premium Prediction", layout="centered")
st.title("💰 Health Insurance Premium Prediction")

# Paths
MODEL_PATH = "models/best_model.pkl"
FEATURE_COLS_PATH = "models/feature_columns.csv"
SCALER_PATH = "data/processed/scaler.pkl"

# --- UI Form (no sidebar, all in main body) ---
st.header("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    number_of_dependants = st.number_input("Number of Dependants", min_value=0, max_value=10, value=0)
    income_lakhs = st.number_input("Income (Lakhs)", min_value=0, max_value=100, value=10)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Unmarried"])

with col2:
    region = st.selectbox("Region", ["Northwest", "Northeast", "Southeast", "Southwest"])
    bmi_category = st.selectbox("BMI Category", ["Normal", "Overweight", "Obesity", "Underweight"])
    smoking_status = st.selectbox("Smoking Status", ["No Smoking", "Occasional", "Regular"])
    employment_status = st.selectbox("Employment Status", ["Salaried", "Self-Employed", "Freelancer"])
    income_level = st.selectbox("Income Level", ["<10L", "10L - 25L", "25L - 40L", "> 40L"])
    insurance_plan = st.selectbox("Insurance Plan", ["Bronze", "Silver", "Gold"])

medical_history = st.selectbox("Medical History", ["No Disease", "Diabetes", "High Blood Pressure", "Thyroid", "Heart Disease"])

# --- Prediction Button ---
if st.button("Predict Premium"):
    if not os.path.exists(MODEL_PATH):
        st.error("Trained model not found. Please run `scripts/train_model.py` first.")
    else:
        # Create single-row dataframe
        input_dict = {
            "age": age,
            "number_of_dependants": number_of_dependants,
            "income_lakhs": income_lakhs,
            "gender": gender,
            "region": region,
            "marital_status": marital_status,
            "bmi_category": bmi_category,
            "smoking_status": smoking_status,
            "employment_status": employment_status,
            "income_level": income_level,
            "insurance_plan": insurance_plan,
            "medical_history": medical_history,
            "annual_premium_amount": 0  # placeholder target
        }
        df = pd.DataFrame([input_dict])

        # # --- Preprocessing ---
        # df = generate_risk_features(df)
        # df = drop_unnecessary_features(df, ["medical_history","disease1","disease2","total_risk"])
        # df = encode_categorical_features(df)

        # Align with training feature columns
        feature_cols = pd.read_csv(FEATURE_COLS_PATH).squeeze().tolist()
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_cols]

        # Load scaler & model
        scaler = joblib.load(SCALER_PATH)
        model = joblib.load(MODEL_PATH)

        # Predict
        pred = model.predict(df)[0]

        # --- Display Result ---
        st.success(f"🏦 Predicted Annual Premium: **₹ {pred:,.2f}**")

