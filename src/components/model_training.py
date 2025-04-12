import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exceptions import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for model training.
    
    Attributes:
        model_trainer_path (str): Path to save the trained model.
    """
    model_trainer_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            
            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "LinearRegression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=0),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                                 models=models)
            
            # Get the best model name
            max_item = max(model_report.items(), key=lambda x: x[1])
            best_model_name, best_model_score = max_item

            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score greater than 0.6")
            
            logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.model_trainer_path,
                obj=models[best_model_name],
            )
            logging.info(f"Model saved at: {self.model_trainer_config.model_trainer_path}")

            return models[best_model_name], best_model_score


        except Exception as e:
            raise CustomException(e, sys)