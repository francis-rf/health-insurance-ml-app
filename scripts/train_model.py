
import pandas as pd
import joblib
import os
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.train import train_linear_models, train_xgboost_model, tune_xgboost_model, evaluate_model, save_model

def main():
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

    # Train all models
    models = train_linear_models(X_train, y_train)
    xgb_model = train_xgboost_model(X_train, y_train)
    tuned_xgb = tune_xgboost_model(xgb_model, X_train, y_train)

    # Evaluate all
    for name, model in models.items():
        evaluate_model(model, X_test, y_test)
    evaluate_model(xgb_model, X_test, y_test)
    evaluate_model(tuned_xgb, X_test, y_test)

    # Select best model: LinearRegression
    best_model = models['LinearRegression']
    save_model(best_model, 'models/best_model.pkl')

    # Save feature columns
    pd.Series(X_train.columns).to_csv('models/feature_columns.csv', index=False)

if __name__ == "__main__":
    main()
