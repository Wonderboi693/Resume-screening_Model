import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str=os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting training and test input data')

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
        
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Logistic Regression": LogisticRegression(),
                "XGBRegressor": XGBRegressor(),
            }

            params = {
                "Random Forest": {
                    "n_estimators": [10, 50, 100]
                },
                "Decision Tree": {
                    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [50, 100]
                },
                "Logistic Regression": {}, 
                "XGBClassifier": {
                    "learning_rate": [0.1, 0.01],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.7, 0.8, 0.9],
                    "n_estimators": [50, 100, 200, 300]
                }
            }

            model_report: dict=evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            

            best_model = models[best_model_name]

            
            logging.info(f"Best found {best_model_name} model on both training and testing dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)