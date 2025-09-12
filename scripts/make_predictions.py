import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import joblib

def main(input_csv='data/sample/sample_input.csv'):
    model = joblib.load('models/best_model.pkl')
    X = pd.read_csv(input_csv)
    preds = model.predict(X)
    out = pd.DataFrame({'prediction': preds})
    out.to_csv('outputs/predictions/predictions.csv', index=False)
    print("✅ Predictions saved to outputs/predictions/predictions.csv")

if __name__ == "__main__":
    main()
