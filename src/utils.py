import os
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import dill
from src.exceptions import CustomException

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    
    Parameters:
    file_path (str): The path where the object will be saved.
    obj: The object to be saved.
    
    Returns:
    None
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
            
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate a machine learning model using R2 score.
    
    Parameters:
    X_train (DataFrame): Training features.
    y_train (Series): Training target variable.
    X_test (DataFrame): Testing features.
    y_test (Series): Testing target variable.
    model: The machine learning model to be evaluated.
    
    Returns:
    float: The R2 score of the model on the test data.
    """
    try:
        report = {}

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            report[model_name] = r2_score(y_test, y_pred)

        return report
    
    except Exception as e:
        raise CustomException(e, sys)