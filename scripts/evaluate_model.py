
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.evaluation import load_model, evaluate_model, plot_results, error_distribution, business_criteria_check

def main():
    X_test, y_test, model = load_model('data/processed/X_test.csv', 'data/processed/y_test.csv', 'models/best_model.pkl')
    y_pred, mae, rmse, r2 = evaluate_model(model, X_test, y_test)
    plot_results(y_test, y_pred)
    error_distribution(y_test, y_pred)
    business_criteria_check(y_test, y_pred)

if __name__ == "__main__":
    main()
