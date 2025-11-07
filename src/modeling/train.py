# src/modeling/train.py

import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(y_test, y_pred, y_pred_proba):
    """Calculate and return evaluation metrics."""
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba)
    }

def main():
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model for churn prediction.")
    parser.add_argument("--model_type", type=str, default="RandomForest", choices=["LogisticRegression", "RandomForest", "XGBoost"])
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees (for RF/XGB).")
    parser.add_argument("--max_depth", type=int, default=10, help="Max depth (for RF/XGB).")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate (for XGB).")
    args = parser.parse_args()

    # 2. Set up MLflow
    mlflow.set_experiment("Churn Prediction Experiment")

    # 3. Load and prepare data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    data_path = os.path.join(project_root, 'data', 'processed', 'processed_data.csv')
    df = pd.read_csv(data_path)
    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Train the model
    with mlflow.start_run(run_name=args.model_type):
        # Log parameters
        mlflow.log_param("model_type", args.model_type)
        if args.model_type == "LogisticRegression":
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif args.model_type == "RandomForest":
            model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
            mlflow.log_params({"n_estimators": args.n_estimators, "max_depth": args.max_depth})
        elif args.model_type == "XGBoost":
            model = XGBClassifier(n_estimators=args.n_estimators, learning_rate=args.learning_rate, use_label_encoder=False, eval_metric='logloss', random_state=42)
            mlflow.log_params({"n_estimators": args.n_estimators, "learning_rate": args.learning_rate})
        
        model.fit(X_train, y_train)
        
        # Evaluate and log metrics
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        mlflow.log_metrics(metrics)
        
        # Log the model
        if args.model_type == "XGBoost":
            mlflow.xgboost.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")
        
        print(f"Model {args.model_type} trained. AUC: {metrics['auc']:.4f}")

if __name__ == "__main__":
    main()