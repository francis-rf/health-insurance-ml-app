
# 💰 Health Insurance Premium Prediction App

This is a Streamlit-based web application that predicts annual health insurance premiums based on customer details. The app uses machine learning models trained on historical insurance data and includes dynamic preprocessing based on user inputs.

## 🚀 Features
- Predict insurance premiums using customer demographics and health history
- Dynamic risk scoring based on medical conditions
- Age-based model selection for improved accuracy
- Interactive UI built with Streamlit

## 🧠 Model Logic
- Risk scores are calculated using `calculate_normalized_risk()` based on selected medical conditions.
- Input data is processed using `preprocess_input()` which encodes categorical variables and applies scaling.

## 🛠 Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/insurance-premium-app.git
   cd insurance-premium-app
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📊 Input Fields
- Age
- Number of Dependants
- Income in Lakhs
- Gender
- Marital Status
- Region
- BMI Category
- Smoking Status
- Employment Status
- Income Level
- Insurance Plan
- Medical History (multi-select)

## 📁 Artifacts
- `model_young.joblib` and `model_rest.joblib`: trained models
- `scaler_young.joblib` and `scaler_rest.joblib`: scalers for preprocessing

## 📬 Contact
For questions or support, contact:
- **Name**: Francis
- **Email**: francis@example.com


