# src/feature_engineering/process_data.py

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder

def main():
    """
    Main function to run the data processing and feature engineering pipeline.
    """
    # 1. DEFINE PATHS
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    raw_data_path = os.path.join(project_root, 'data', 'raw', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_data.csv')

    print(f"Reading raw data from: {raw_data_path}")

    # 2. LOAD DATA
    try:
        df = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {raw_data_path}")
        print("Please make sure you have run 'dvc pull' to get the raw data.")
        return

    # 3. DATA CLEANING
    print("Cleaning data...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    initial_count = len(df)
    df.dropna(inplace=True)
    final_count = len(df)
    print(f"Dropped {initial_count - final_count} rows with missing TotalCharges.")

    # 4. FEATURE ENGINEERING
    print("Engineering new features...")
    df['tenure_in_years'] = df['tenure'] / 12
    
    service_cols = [
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['num_services'] = df[service_cols].apply(lambda row: (row != 'No').sum(), axis=1)
    df['has_multiple_services'] = (df['num_services'] > 1).astype(int)
    df.drop('num_services', axis=1, inplace=True) # Clean up temporary column
    print("New features created.")

    # 5. SEPARATE FEATURES AND TARGET
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn']

    # 6. IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

    # 7. ONE-HOT ENCODING
    print("Applying One-Hot Encoding...")
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_data = encoder.fit_transform(X[categorical_cols])
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

    # 8. COMBINE PROCESSED DATA
    X_processed = pd.concat([X[numerical_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    final_df = pd.concat([X_processed, y.reset_index(drop=True)], axis=1)

    # 9. SAVE THE PROCESSED DATA
    print(f"Saving processed data to: {processed_data_path}")
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
    final_df.to_csv(processed_data_path, index=False)
    
    print("Data processing complete!")
    print(f"Final processed data shape: {final_df.shape}")


if __name__ == "__main__":
    main()