import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.exception import CustomException
from src.logged.logger import logging
from src.utils.util import save_object

# Optional dependencies
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:  # pragma: no cover
    XGBRegressor = None  # type: ignore

try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:  # pragma: no cover
    CatBoostRegressor = None  # type: ignore

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, object],
) -> Dict[str, float]:
    """Train each model and return a report of R2 scores on the test set."""
    report: Dict[str, float] = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            report[name] = float(r2)
            logging.info(f"Trained {name} | R2: {r2:.4f}")
        except Exception as train_err:
            logging.info(f"Skipping {name} due to error: {train_err}")
    return report


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(
        self, train_array: np.ndarray, test_array: np.ndarray
    ) -> Tuple[str, float, str]:
        try:
            logging.info(
                "Splitting dependent and independent variables from train and test data"
            )
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost": AdaBoostRegressor(),
            }
            if XGBoostRegressor := XGBRegressor:  # type: ignore
                models["XGBoost"] = XGBoostRegressor()
            if CatBoostRegressor is not None:  # type: ignore
                models["CatBoost"] = CatBoostRegressor(verbose=False)  # type: ignore

            model_report: Dict[str, float] = evaluate_models(
                X_train, y_train, X_test, y_test, models
            )
            if not model_report:
                raise CustomException("All models failed to train.", sys)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
            logging.info(
                f"Best model: {best_model_name} | R2: {best_model_score:.4f} | Saved to: {self.model_trainer_config.trained_model_file_path}"
            )

            return best_model_name, best_model_score, self.model_trainer_config.trained_model_file_path
        except Exception as e:
            raise CustomException(e, sys)