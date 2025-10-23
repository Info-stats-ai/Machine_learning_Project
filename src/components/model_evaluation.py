import os
import sys
import json
import math
import pickle
import dill
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.exception import CustomException
from src.logged.logger import logging


@dataclass
class ModelEvaluationConfig:
    evaluation_report_path: str = os.path.join("artifacts", "evaluation_report.json")
    model_path: str = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")
    test_csv_path: str = os.path.join("artifacts", "test.csv")
    target_column_name: str = "math_score"


class ModelEvaluator:
    def __init__(self) -> None:
        self.config = ModelEvaluationConfig()

    def _load_model(self, path: str):
        try:
            with open(path, "rb") as f:
                return dill.load(f)
        except Exception as e:
            raise CustomException(e, sys)

    def _load_preprocessor(self, path: str):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_on_test(self) -> dict:
        try:
            logging.info("Starting model evaluation on test data")

            # Load artifacts
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model not found at {self.config.model_path}")
            if not os.path.exists(self.config.preprocessor_path):
                raise FileNotFoundError(f"Preprocessor not found at {self.config.preprocessor_path}")
            if not os.path.exists(self.config.test_csv_path):
                raise FileNotFoundError(f"Test CSV not found at {self.config.test_csv_path}")

            model = self._load_model(self.config.model_path)
            preprocessor = self._load_preprocessor(self.config.preprocessor_path)
            logging.info("Loaded model and preprocessor")

            # Read test data
            test_df = pd.read_csv(self.config.test_csv_path)
            if self.config.target_column_name not in test_df.columns:
                raise KeyError(
                    f"Target column '{self.config.target_column_name}' not found in test data"
                )

            X_test = test_df.drop(columns=[self.config.target_column_name], axis=1)
            y_test = test_df[self.config.target_column_name]

            # Transform and predict
            X_test_transformed = preprocessor.transform(X_test)
            y_pred = model.predict(X_test_transformed)

            # Metrics
            r2 = float(r2_score(y_test, y_pred))
            mae = float(mean_absolute_error(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(math.sqrt(mse))

            report = {
                "r2_score": r2,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "num_test_samples": int(len(y_test)),
            }

            # Persist report
            os.makedirs(os.path.dirname(self.config.evaluation_report_path), exist_ok=True)
            with open(self.config.evaluation_report_path, "w") as f:
                json.dump(report, f, indent=2)

            logging.info(
                f"Evaluation completed | R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f} | "
                f"Report saved to {self.config.evaluation_report_path}"
            )

            return report
        except Exception as e:
            logging.info("Error during model evaluation")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        evaluator = ModelEvaluator()
        evaluator.evaluate_on_test()
    except Exception as e:
        raise

