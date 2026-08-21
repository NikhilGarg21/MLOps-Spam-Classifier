import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score , f1_score
from logger import get_logger

logger = get_logger("model_evaluation")
logger.debug("Model_evaluation started")


def load_model(model_path: str) -> None:
    """Load the trained model from a file."""
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model loaded from %s", model_path)
        return model
    except FileNotFoundError:
        logger.error("File not found: %s", model_path)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the model: %s", e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """load data from a csv file"""
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data: %s", e)
        raise


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = model.predict(X_test)
        y_score = model.decision_function(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_score ),
        }

        logger.debug("Model evaluation metrics calculated")
        return metrics

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)
        logger.debug("Metrics saved to %s", file_path)

    except Exception as e:
        logger.error("Error occurred while saving the metrics: %s", e)
        raise


def main():
    try:
        model = load_model(r"C:\MLOPS\models\model.pkl")
        test_data = load_data(r"C:\MLOPS\data\processed\test_tfidf.csv")

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, "reports/metrics.json")
    except Exception as e:
        logger.error("Failed to complete the model evaluation process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
