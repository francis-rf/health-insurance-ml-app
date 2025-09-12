
import os
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing.data_preprocessing import load_data, generate_risk_features, drop_unnecessary_features, encode_categorical_features, scale_features, split_and_save_data

RAW_PATH = 'data/processed/cleaned.xlsx'

def main():
    df = load_data(RAW_PATH)
    df = generate_risk_features(df)
    df = drop_unnecessary_features(df, ['medical_history','disease1','disease2','total_risk'])
    df = encode_categorical_features(df)
    cols_to_scale = ['age','number_of_dependants','income_level','income_lakhs','insurance_plan']
    X, y, scaler = scale_features(df, target_col='annual_premium_amount', cols_to_scale=cols_to_scale)
    split_and_save_data(X, y, scaler, output_dir='data/processed/')

if __name__ == "__main__":
    main()
