# # ===========================
# # Imports
# # ===========================
# # Standard library imports
# import os
# from typing import Tuple
# import warnings
# from IPython.display import display

# # Third-party imports
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.stats.outliers_influence import variance_inflation_factor
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# import joblib

# # Display options
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 120)
# sns.set_theme(style='whitegrid')
# warnings.filterwarnings('ignore')

# # Constants
# DATA_PATH = r"../data/processed/cleaned.xlsx"  # update path if file is located elsewhere
# # ===========================
# # Load Data
# # ===========================
# def load_data(path: str) -> pd.DataFrame:
#     """
#     Load dataset from CSV or Excel file.
#     Args:
#         path (str): File path of dataset.
#     Returns:
#         pd.DataFrame: Loaded dataset.
#     """
#     if path.endswith('.csv'):
#         return pd.read_csv(path)
#     elif path.endswith('.xlsx'):
#         return pd.read_excel(path,engine='openpyxl')
#     else:
#         raise ValueError('Unsupported file format. Use CSV or XLSX.')
# try:
#     df = load_data(DATA_PATH) # update path if if necessary
#     print('Data loaded Successfully.')
#     if not df.empty:
#         display(df.head())
# except Exception as exec:
#     print('Data failed to load:', exec)
#     df = pd.DataFrame()  # Prevent NameError

# # ===========================
# # Featrue Engineering
# # ===========================
# def create_risk_features(data: pd.DataFrame) -> pd.DataFrame:
#     """
#     Generating a new feature based on existing features.
#     Args:
#         data: pd.DataFrame: dataset
#     Returns:
#         pd.DataFrame: Loaded dataset.
#     """
#     risk_scores = {
#         'diabetes': 6,
#         'high blood pressure': 6,
#         'no disease': 0,
#         'thyroid': 5,
#         'heart disease': 8
#     }

#     data[['disease1', 'disease2']] = data['medical_history'].str.split(' & ', expand=True)
#     data = data.fillna({'disease1': np.nan, 'disease2': np.nan})
#     data['total_risk'] = 0
#     for col in ['disease1', 'disease2']:
#         data['total_risk'] += data[col].str.lower().map(risk_scores).fillna(0)


#     min_ = data['total_risk'].min()
#     max_ = data['total_risk'].max()


#     data['normalized_risk'] = data['total_risk'].apply(lambda row : (row - min_)/(max_ - min_))
    
#     return data

# df = create_risk_features(df)

# # ===========================
# # Featrue Selection
# # ===========================
# def feature_selection(data: pd.DataFrame,columns: list) -> pd.DataFrame:
#     """
#     Selects necessary features by droping unnecessry ones.
#     Args:
#         data: pd.DataFrame: dataset
#         columns: list: columns want to drop

#     Returns:
#         pd.DataFrame: Loaded dataset.
#     """
#     data = data.drop(columns,axis=1)

#     return data

# df = feature_selection(df,['medical_history','disease1', 'disease2', 'total_risk'])

# # ===========================
# # Featrue Encoding
# # ===========================
# def encoding_cat_vars(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Encoding categorical variables:
#       - Map ordinal categorical variables to numbers
#       - Apply one-hot encoding to nominal categorical variables
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input insurance dataframe
    
#     Returns
#     -------
#     pd.DataFrame
#         Preprocessed dataframe
#     """

#     # Make a copy to avoid modifying original
#     df = df.copy()

#     # Ordinal mappings
#     df['insurance_plan'] = df['insurance_plan'].map({
#         'Bronze': 1, 
#         'Silver': 2, 
#         'Gold': 3
#     })

#     df['income_level'] = df['income_level'].map({
#         '<10L': 1, 
#         '10L - 25L': 2, 
#         '25L - 40L': 3, 
#         '> 40L': 4
#     })

#     # One-hot encoding for nominal categorical columns
#     df = pd.get_dummies(
#         df,
#         columns=['gender','bmi_category','region','marital_status','smoking_status','employment_status'],
#         drop_first=True,
#         dtype=int
#     )

#     return df

# df =encoding_cat_vars(df)
# # ===========================
# # Correlation Matrix
# # ===========================

# def plot_correlation_matrix(df: pd.DataFrame,figsize: Tuple[int, int] = (24, 16)): 
#     """
#     visualization of correlation matrix for all the numerical columns
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input insurance dataframe
#     figsize: Tuple[int, int] 
#         plot size 
#     Returns
#     -------
#     None
#     """
#     corr = df.corr()
#     plt.figure(figsize=figsize)
#     sns.heatmap(corr, annot=True, cmap='vlag', center=0,fmt='.2f')
#     plt.title('Correlation matrix')
#     plt.show()

# plot_correlation_matrix(df)

# # ===========================
# # Featrue Scaling
# # ===========================

# def scale_features(df: pd.DataFrame, target_col: str, cols_to_scale: list) -> tuple:
#     """
#     Split dataset into X and y, and apply MinMax scaling to selected columns.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         Input dataframe
#     target_col : str
#         The target column to predict
#     cols_to_scale : list
#         List of feature columns to scale
    
#     Returns
#     -------
#     X : pd.DataFrame
#         Features dataframe with scaled selected columns
#     y : pd.Series
#         Target variable
#     scaler : MinMaxScaler
#         Fitted scaler (useful for transforming new/unseen data)
#     """
    
#     # Separate features and target
#     X = df.drop(target_col, axis=1).copy()
#     y = df[target_col].copy()

#     # Initialize and fit scaler
#     scaler = MinMaxScaler()
#     X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

#     return X, y, scaler

# cols_to_scale = ['age','number_of_dependants','income_level','income_lakhs','insurance_plan']

# X, y, scaler = scale_features(df, target_col='annual_premium_amount', cols_to_scale=cols_to_scale) 

# # ===========================
# # Variance Inflation Factor
# # ===========================

# def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculate Variance Inflation Factor (VIF) for each feature in a DataFrame.
    
#     Parameters
#     ----------
#     df : pd.DataFrame
#         DataFrame containing only the features (no target column).
    
#     Returns
#     -------
#     pd.DataFrame
#         DataFrame with two columns:
#         - 'Feature': feature names
#         - 'VIF': variance inflation factor values
#     """
#     # Ensure only numeric columns are used
#     numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()

#     vif_data = pd.DataFrame()
#     vif_data['Feature'] = numeric_df.columns
#     vif_data['VIF'] = [
#         variance_inflation_factor(numeric_df.values, i)
#         for i in range(numeric_df.shape[1])
#     ]

#     return vif_data

# vif_results = calculate_vif(X)
# vif_results
# vif_results = calculate_vif(X.drop('income_level',axis=1))
# vif_results

# # ===========================
# # Train-Test Split & Save Processed Data
# # ===========================

# def split_and_save_data(X: pd.DataFrame, y: pd.Series, scaler,
#                         output_dir: str = '../data/processed/',
#                         test_size: float = 0.2, random_state: int = 42):
#     """
#     Split dataset into train/test sets and save them along with scaler.
#     """

 
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=test_size, random_state=random_state
#     )
 
#     os.makedirs(output_dir, exist_ok=True)
#     X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
#     X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
#     y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
#     y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
 
#     joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
 
#     print('✅ Data split and saved successfully.')
#     return X_train, X_test, y_train, y_test

# X_train, X_test, y_train, y_test = split_and_save_data(X, y, scaler)

import os
import logging
from typing import Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
sns.set_theme(style='whitegrid')

# ===========================
# Load Data
# ===========================
def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV or Excel file.
    Args:
        path (str): File path of dataset.
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    if path.endswith('.csv'):
        return pd.read_csv(path)
    elif path.endswith('.xlsx'):
        return pd.read_excel(path, engine='openpyxl')
    else:
        raise ValueError('Unsupported file format. Use CSV or XLSX.')

# ===========================
# Feature Engineering
# ===========================
def generate_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate normalized risk score from medical history.
    """
    risk_scores = {
        'diabetes': 6,
        'high blood pressure': 6,
        'no disease': 0,
        'thyroid': 5,
        'heart disease': 8
    }
    df[['disease1', 'disease2']] = df['medical_history'].str.split(' & ', expand=True)
    df['total_risk'] = sum(df[col].str.lower().map(risk_scores).fillna(0) for col in ['disease1', 'disease2'])
    min_, max_ = df['total_risk'].min(), df['total_risk'].max()
    df['normalized_risk'] = (df['total_risk'] - min_) / (max_ - min_)
    return df

# ===========================
# Feature Selection
# ===========================
def drop_unnecessary_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drop specified columns from the dataset.
    """
    return df.drop(columns, axis=1)

# ===========================
# Feature Encoding
# ===========================
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode ordinal and nominal categorical features.
    """
    df = df.copy()
    df['insurance_plan'] = df['insurance_plan'].map({'Bronze': 1, 'Silver': 2, 'Gold': 3})
    df['income_level'] = df['income_level'].map({'<10L': 1, '10L - 25L': 2, '25L - 40L': 3, '> 40L': 4})
    df = pd.get_dummies(df, columns=['gender','bmi_category','region','marital_status','smoking_status','employment_status'], drop_first=True, dtype=int)
    return df

# ===========================
# Correlation Matrix
# ===========================
def plot_correlation_matrix(df: pd.DataFrame, figsize: Tuple[int, int] = (24, 16)) -> None:
    """
    Plot correlation matrix for numerical features.
    """
    corr = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap='vlag', center=0, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

# ===========================
# Feature Scaling
# ===========================
def scale_features(df: pd.DataFrame, target_col: str, cols_to_scale: list) -> Tuple[pd.DataFrame, pd.Series, MinMaxScaler]:
    """
    Scale selected features and separate target.
    """
    X = df.drop(target_col, axis=1).copy()
    y = df[target_col].copy()
    scaler = MinMaxScaler()
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])
    return X, y, scaler

# ===========================
# Variance Inflation Factor
# ===========================
def calculate_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate VIF for numeric features.
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
    vif_data = pd.DataFrame()
    vif_data['Feature'] = numeric_df.columns
    vif_data['VIF'] = [variance_inflation_factor(numeric_df.values, i) for i in range(numeric_df.shape[1])]
    return vif_data

# ===========================
# Train-Test Split & Save
# ===========================
def split_and_save_data(X: pd.DataFrame, y: pd.Series, scaler: MinMaxScaler,
                        output_dir: str = '../data/processed/',
                        test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data and save train/test sets and scaler.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
    logging.info('✅ Data split and saved successfully.')
    return X_train, X_test, y_train, y_test

logging.info("✅ data_preprocessing.py cleaned and improved.")
