import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.utils import save_object
from src.exceptions import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformer:
    '''
    This class is responsible for transforming the data using various preprocessing techniques.
    It handles both numerical and categorical features, applying appropriate transformations to each.
    '''
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['writing score', 'reading score']
            categorical_features = [
                'gender',
                'race/ethnicity',
                'parental level of education', 
                'lunch',
                'test preparation course'
            ]
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))  # Note: StandardScaler does not support sparse matrices
            ])

            logging.info(f"Numerical features: {numerical_features}")
            logging.info(f"Categorical features: {categorical_features}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Data loaded successfully")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math score'
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying transformations")

            input_features_train_df = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_df = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[input_features_train_df, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_df, np.array(target_feature_test_df)]

            logging.info("Saved preprocessor object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path) 

        except Exception as e:
            raise CustomException(e, sys)