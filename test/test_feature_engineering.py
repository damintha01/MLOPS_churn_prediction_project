# tests/test_feature_engineering.py
import pandas as pd
import os
import sys

# Add the src directory to the Python path to import our script
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from feature_engineering.process_data import main as process_main

def test_processed_data_creation():
    """Test that the processing script runs and creates the output file."""
    # Define paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_data.csv')
    
    # Run the processing script
    process_main()
    
    # Check if the output file exists
    assert os.path.exists(processed_data_path), "Processed data file was not created."

def test_processed_data_shape():
    """Test that the processed data has the expected number of columns."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    processed_data_path = os.path.join(project_root, 'data', 'processed', 'processed_data.csv')
    
    # Run the processing script
    process_main()
    
    # Load the data and check its shape
    df = pd.read_csv(processed_data_path)
    # The expected number of columns might change, so check for a reasonable number
    # Original: 21 -> Drop customerID, Churn -> 19
    # One-Hot Encoding adds many columns. Let's just check it's > 30
    assert df.shape[1] > 30, f"Processed data has {df.shape[1]} columns, expected more than 30."